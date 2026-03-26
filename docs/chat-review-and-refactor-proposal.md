# Chat 实现 Review 与重构建议

本文针对当前仓库内 `service/chat` 相关实现做架构 review，目标不是局限于现状修补，而是从可维护性、可扩展性、运行时一致性和未来演进成本出发，给出优化项与重构项。结论默认接受较大幅度重构，必要时可以推倒重来。

## 1. 结论摘要

当前实现已经具备一个雏形完整的 chat 平台骨架：

- 有会话、turn、step、data 的持久化模型
- 有 capability package、agent profile、agent mount 等平台化元件
- 有 tool、MCP、sub-agent、LLM response 等 action 类型
- 有 planner、prompt builder、timeline 查询、websocket 推送

但从系统形态上看，它仍然更接近：

- 一个“可配置的单轮执行编排器”

而不是：

- 一个“可演进的 agent runtime / chat execution platform”

最核心的问题不是局部代码风格，而是以下几类结构性问题：

- turn 生命周期是“数据库状态 + 进程内状态”混搭
- `ResourceSelection` 语义过载，贯穿请求、默认值、planner 输出和执行结果
- sub-agent 只是父执行链中的递归 action，不是 first-class runtime unit
- tool / function call / MCP 还没有真正进入 LLM 决策闭环
- 平台配置项有不少“存下来了，但没有真正参与调度和治理”

如果只做小修小补，系统会越来越难加能力；如果接受重构，建议优先重建 turn 状态机和 execution model。

## 2. 高优先级问题

### 2.1 显式运行时 bug

当前有两处实现会把 `ActionMetadata` 当成 `dict` 使用：

- `service/chat/runtime/tool_executor.py`
- `service/chat/runtime/mcp_executor.py`

其中：

- `execution_overview`
- `platform.capability_overview`

都通过 `item.metadata.get(...)` 取值，但 `metadata` 实际是 `ActionMetadata` 模型，不支持 `.get()`。这些诊断路径一旦被调用会直接失败。

这说明当前测试覆盖偏向主路径，平台自省和诊断路径还不够可靠。

### 2.2 turn 生命周期模型不稳定

当前 turn 的运行态分成两层：

- 数据库里的 `ChatTurn.status`
- 进程内的 `pending_turns / running_turns`

这会带来几个问题：

- 进程重启后，内存态消失，但数据库里可能仍是 `running`
- cancel 依赖当前进程持有 task
- 多实例部署下，哪个实例拥有 turn 不明确
- prepare 完成但 launch 前失败时，状态已经部分写入数据库

这不是简单的实现细节，而是 runtime ownership 没有被建模清楚。

### 2.3 prepare / launch 之间有僵尸窗口

当前流程中：

- 新会话在资源解析前就创建
- turn 在真正执行前就写成 `running`

所以会出现：

- planner 或资源解析失败，留下空 conversation
- turn 尚未启动就已经是 `running`
- websocket/应用层在 `prepare_turn` 和 `launch_prepared_turn` 之间异常时，留下半成品状态

这会直接影响数据清洁度和问题排查。

### 2.4 sub-agent 不是真正独立执行单元

当前 sub-agent 本质上是：

- 在父 turn 内部派生一个 `ChatSessionContext`
- 递归执行一组 nested actions

它不是独立 turn，也没有独立执行边界，因此缺失：

- 独立 trace
- 独立 usage/accounting
- 独立审批与权限边界
- 独立重试策略
- 独立调度和超时控制

这意味着现在的 “sub-agent” 更像“嵌套 workflow”，不是 agent runtime 中真正可运营的 agent task。

### 2.5 还没有真正的 function/tool-calling loop

现在的模式是：

- 先根据配置和启发式/planner 选择工具
- 执行工具/MCP
- 再让 LLM 生成最终文本

这不是现代 agent runtime 里常见的：

- model sees tools
- model chooses tool
- tool returns observation
- model decides next action
- final answer

因此当前系统虽有 `tool_call` / `mcp_call` / `sub_agent_call`，但并未形成真正的 agentic loop。扩展到 function call、多步工具使用、动态澄清、反思式规划时会比较受限。

## 3. 设计层面

### 3.1 当前设计判断

当前实现已经混合了四种角色：

- Conversation API
- Capability Registry
- Runtime Planner
- Execution Engine

这些角色并非没有拆文件，但边界仍不够硬，很多对象在不同层承担多重职责。

典型表现：

- `ResourceSelection` 同时承担请求表达、默认配置、planner 输入、planner 输出、执行动作集合
- `ChatSessionContext` 同时承载 request、conversation、agent、plan、artifacts、runtime state
- `ChatApplicationService` 直接 new 出整套系统并作为全局单例持有

### 3.2 建议的目标结构

建议从设计上明确拆成四层：

1. `Conversation Layer`
   - 负责 session、conversation、turn API
   - 只处理协议、权限、上下文绑定

2. `Resolution Layer`
   - 负责 agent 默认值、conversation 默认值、request overrides 合并
   - 输出稳定的 `ResolvedPlan`

3. `Execution Layer`
   - 负责 turn 状态机、step trace、operation dispatch、cancel/retry
   - 不感知 API transport

4. `Capability Layer`
   - 负责 capability registry、tool registry、MCP registry、agent registry
   - 负责 schema、governance、availability、health

### 3.3 turn 应升级为显式状态机

建议把 turn 状态从现在的隐式两段式，改成显式状态机，例如：

- `draft`
- `planned`
- `queued`
- `running`
- `completed`
- `failed`
- `canceled`

这样：

- `prepare` 只负责生成 `planned`
- `launch` 或调度器接手后进入 `queued/running`
- cancel/retry/recovery 都有稳定语义

### 3.4 websocket 不应绑定“当前会话”

当前 websocket session 维护了一个隐式当前 conversation。这个设计对 debug 很方便，但对长期演进不理想：

- transport state 与业务状态耦合
- 多标签页/多终端/断线重连语义模糊
- 回放、补发、恢复都更复杂

建议改成：

- 每个 `turn.start` 显式提供 `conversation_id`
- websocket 只是事件通道，不持有业务 session 主状态

## 4. 数据流转

### 4.1 `ResourceSelection` 语义过载

当前 `ResourceSelection` 包含：

- system defaults 开关
- conversation defaults 开关
- capability selections
- actions
- planner config

问题在于它被用于多个阶段：

- 用户请求输入
- 会话默认配置
- agent 默认配置
- planner 输入
- planner 输出之后的执行动作载体
- turn 持久化快照

一个对象贯穿这么多语义阶段，会导致：

- merge 逻辑越来越复杂
- 难判断 action 来源
- 排查 planner 问题时不容易知道是用户选的、agent 默认的还是 planner 自动加的

### 4.2 建议拆成四类对象

建议拆成以下模型：

- `RequestedResources`
  - 来自用户/API 的显式选择

- `ResolvedResources`
  - 合并 agent default、conversation default 后的资源候选集

- `ExecutionPlan`
  - planner 最终输出，明确本轮要执行什么

- `ExecutionTrace`
  - 运行过程中的 step、input/output、usage、errors

这样可以显著降低 merge/normalize 的复杂度。

### 4.3 step/data 模型建议 append-only

当前 step/data 已经有不错基础，但整体消费方式仍偏“实时拼 summary”。建议进一步明确：

- `turn/step/data` 是 append-only trace log
- conversation list、timeline、debug view 是 projection/read model

这样后续更容易支持：

- trace replay
- observability
- 异步索引
- analytics
- agent eval

### 4.4 prompt context、tool context、retrieval context 建议分 lane

目前最终 prompt 是把：

- history
- context_items
- prompt_context
- capability overlay

拼成文本。

这在早期可行，但后面会出现几个问题：

- retrieval 证据与 tool 输出没有严格优先级
- 子代理结果和普通文本上下文混在一起
- prompt 过长时难做裁剪策略
- 难以按 lane 做 token budgeting

建议把 prompt 输入显式拆 lane：

- conversation history lane
- retrieved evidence lane
- structured tool result lane
- delegated agent result lane
- system policy lane

然后在 prompt compiler 中做预算和裁剪。

## 5. 功能扩展

### 5.1 skill 目前更像 inline action bundle

当前 skill capability 的本质是：

- 一组 actions
- 一组 instructions
- 一些 preferred references

这适合作为“编排片段”，但还不算 workflow。

如果未来要把 skill 作为主要扩展单元，建议升级为：

- 支持前置条件
- 支持依赖 capability
- 支持并行段
- 支持终止条件
- 支持失败恢复策略
- 支持输入/输出 contract

也就是从 “action bundle” 升级成 “workflow template”。

### 5.2 tool / function call / MCP / sub-agent 应统一成 Operation 抽象

建议定义统一的 `Operation` 抽象，至少包含：

- operation schema
- invocation policy
- approval policy
- timeout / retry policy
- result contract
- observability hooks

这样：

- local tool
- MCP tool
- function call
- sub-agent delegation
- human approval

都可以走统一执行平面，而不是现在分散在不同 executor 中靠 if/enum 区分。

### 5.3 真正支持 function calling

建议把 function calling 支持补成两层：

1. platform registry 层
   - 管理 function schema、tool schema、MCP schema

2. model runtime 层
   - 把可用 function 暴露给模型
   - 支持多轮 tool-use loop
   - 把 tool result 作为 observation 返回模型

否则现在的 “tool planner + tool execution + llm summary” 很难覆盖复杂多步任务。

### 5.4 MCP 需要从“能调用”升级到“可运营”

当前 MCP 已具备接通能力，但要进入生产级平台，还需要：

- session pooling
- server health monitoring
- tool schema cache
- per-server timeout policy
- circuit breaker
- capability-level MCP availability gating

否则 MCP 只是一个扩展执行器，不是稳定平台能力。

### 5.5 sub-agent 建议支持 `delegate` 与 `handoff`

现在只有 delegate 语义。建议拆成两类：

- `delegate`
  - 父代理保留控制权
  - 子代理返回结果

- `handoff`
  - 当前轮或后续轮切换主 agent
  - 会话主控权发生转移

这两个模式在产品和 runtime 语义上差别很大，不宜混在一个 `sub_agent_call` 里。

### 5.6 governance 字段应真正参与路由

当前以下字段存在“定义了但弱生效/未生效”的情况：

- `trigger_tags`
- `requires_deps`
- `visible_to_agents`
- `pass_deps_fields`
- `output_contract`

其中：

- `visible_to_agents` 已参与过滤
- `pass_deps_fields`、`output_contract` 已参与执行
- `trigger_tags` 基本未参与路由
- `requires_deps` 基本未参与治理校验

建议让 governance 字段真正参与：

- capability visibility
- route candidate ranking
- execution readiness check
- approval policy
- fallback policy

## 6. 代码实现

### 6.1 去掉全局单例装配

当前 `chat_app_service = ChatApplicationService()` 直接在模块尾部初始化整套系统。这会带来：

- 可测试性差
- registry/planner/runtime 难替换
- 生命周期不清晰
- 启动时初始化副作用不好控

建议改成：

- 应用启动时通过工厂装配
- 在 FastAPI 依赖注入中按需获取 service
- registry/runtime 可显式替换

### 6.2 `session.state` 需要类型化

现在 `ChatSessionContext.state` 是字符串 key 的杂物箱，里面混合了：

- `active_step_ids`
- `llm_system_prompt_prefix`
- `llm_extra_instructions`
- `llm_include_history`
- `_tool_runtime_state`
- `last_error_data_id`

这类状态随着功能增长会迅速失控。建议拆成 typed state：

- `StepRuntimeState`
- `LLMExecutionState`
- `ToolExecutionState`
- `ErrorState`

### 6.3 prompt builder 当前更像字符串模板拼接器

`ChatPromptBuilder` 目前职责包括：

- system overlay 生成
- capability summary 生成
- history 文本化
- context item 文本化

建议拆成：

- `PromptInputAssembler`
- `PromptBudgeter`
- `PromptRenderer`

这样后面加：

- token budgeting
- context compression
- retrieval citation policy
- agent-specific rendering

会容易很多。

### 6.4 `TurnArtifacts.instructions` 目前基本是死字段

`TurnArtifacts` 里有 `instructions`，但主路径几乎没有把能力说明真正写进去。这意味着：

- schema 已准备
- 运行时没有形成稳定使用方式

建议二选一：

- 如果需要，就把它接入 capability / planner / prompt compiler
- 如果不需要，就删掉，减少虚假抽象

### 6.5 capability/action 校验应尽量前置

当前部分问题要到运行时才暴露，例如：

- tool 未注册
- 依赖不满足
- capability 配置无效
- route 后 action 无法执行

建议把校验前置到：

- capability 注册时
- agent 挂载时
- planner 产出 plan 时

运行时只处理动态性错误，而不是静态配置错误。

### 6.6 内置能力不宜硬编码在 service 中

当前 builtin agent / builtin capability 大量写在 service 代码里。短期方便，但长期会有问题：

- 发布流程和配置管理耦合
- 不能版本化演进
- 难以在不同环境下差异化

建议改成：

- manifest 化
- 启动时导入
- 支持版本迁移

## 7. 推荐重构路径

如果允许较大重构，建议按以下顺序推进。

### 第一阶段：止血与清边界

- 修复显式 bug，如 `metadata.get(...)`
- 补齐平台诊断路径测试
- 去掉 module-level 全局单例
- 明确 transport 与 runtime 边界
- 增加 turn 状态机中的 `planned` 态

### 第二阶段：重构执行模型

- 把 turn 执行 ownership 持久化
- 把 `pending/running` 从进程内字典迁移到可恢复模型
- 把 sub-agent 提升为独立 task/turn
- 把 tool/MCP/sub-agent 统一为 operation dispatch

### 第三阶段：重构资源解析模型

- 拆解 `ResourceSelection`
- 把 planner 输出稳定化为 `ExecutionPlan`
- 把 trace 与 projection 分离
- 建立 capability readiness/governance 校验

### 第四阶段：引入真正的 agentic loop

- 模型 runtime 暴露 function/tool schema
- 支持多步 tool-use loop
- 支持 observation -> replanning -> final answer
- 加入 approval/human-in-the-loop operation

## 8. 我会优先做的两件事

如果只允许选两件最有价值的重构事项，我会优先做：

### 8.1 把 turn 改成持久化状态机

这是 runtime 稳定性的基础。没有这个，后续：

- cancel
- retry
- recovery
- queue
- multi-instance

都会越来越脆弱。

### 8.2 把 LLM 执行改成真正的 function/tool-calling loop

这是扩展能力的基础。没有这个，当前系统虽然有 tool/MCP/sub-agent 的外观，但仍然是静态编排优先，离真正 agent runtime 还差一层核心能力。

## 9. 总结

当前 chat 实现已经不是一个简单接口，而是一个正在生长中的平台内核。问题不在于“有没有功能”，而在于：

- 模型、编排、治理、执行、持久化之间的边界还没有真正稳定下来

如果继续沿现状追加功能，复杂度会主要积累在：

- `ResourceSelection` merge
- `ChatSessionContext.state`
- 各类 action executor
- prepare/launch/cancel 的状态一致性

如果接受重构，建议把目标明确为：

- 一个可恢复、可观测、可治理、可扩展的 agent runtime

围绕这个目标重建 turn 状态机、operation 抽象和 function-calling loop，后续 skill/tools/MCP/sub-agent 才会真正站稳。
