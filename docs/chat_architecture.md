# Chat模块开发文档

## 一、架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client (WebSocket)                       │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                      API Layer (FastAPI)                         │
│  api/knowledge_base/v1/chat.py                                   │
│  - WebSocket连接管理                                             │
│  - 命令分发 (turn.start/cancel/resume)                           │
│  - 事件推送                                                      │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│              Application Service Layer                           │
│  service/chat/application/service.py                             │
│  - ChatApplicationService                                       │
│  - 会话管理、权限验证、业务编排                                   │
└─────────────┬────────────────────────────────┬──────────────────┘
              │                                │
┌─────────────▼────────────┐    ┌─────────────▼──────────────────┐
│   Capability Management   │    │   Runtime Engine               │
│  capability/service.py    │    │  runtime/engine.py             │
│  - 能力配置管理           │    │  - Turn任务调度                │
│  - 能力绑定管理           │    │  - 能力Pipeline执行            │
│  - 资源选择解析           │    │  - LLM调用                     │
└─────────────┬────────────┘    │  - 知识检索                    │
              │                 │  - 函数工具执行                │
              │                 └─────────────┬──────────────────┘
              │                               │
              │              ┌────────────────┴──────────────────┐
              │              │  Context & Artifacts              │
              │              │  runtime/context.py               │
              │              │  - TurnArtifacts                 │
              │              │  - ChatSessionContext            │
              │              └────────────────┬──────────────────┘
              │                               │
              └──────────────┬────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                     Data Access Layer                             │
│  store/repository.py                                             │
│  - Conversation/Turn/Step/Data CRUD                              │
│  - EventLog persistence                                          │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                  Database Models (Tortoise ORM)                  │
│  ext/ext_tortoise/models/knowledge_base/                         │
│  - ChatConversation, ChatTurn, ChatStep                          │
│  - ChatData, ChatEventLog, ChatWebSocketSession                  │
└─────────────────────────────────────────────────────────────────┘
```

## 二、核心数据模型

### 2.1 Conversation (会话)
```python
class ConversationSummary:
    id: int                          # 会话ID
    title: str                       # 会话标题
    status: str                      # 状态 (active/archived)
    user_id: int | None             # 所有者ID
    active_turn_id: int | None      # 当前活跃turn
    head_turn_id: int | None        # 最新完成turn
    default_resource_selection: ResourceSelection  # 默认能力配置
```

### 2.2 Turn (对话回合)
```python
class TurnSummary:
    id: int                          # Turn ID
    conversation_id: int            # 所属会话
    seq: int                        # 序号
    status: str                     # 状态 (pending/accepted/running/streaming/completed/failed/canceled)
    trigger: str                    # 触发方式 (user/system)
    input_root_data_id: int | None  # 用户输入数据ID
    output_root_data_id: int | None # AI回复数据ID
    root_step_id: int | None        # 根步骤ID
```

### 2.3 Step (执行步骤)
```python
class StepSummary:
    id: int
    turn_id: int
    kind: str                       # 类型 (system/retrieval/tool/llm)
    status: str                     # (pending/running/completed/failed)
    sequence: int                   # 执行顺序
```

### 2.4 Data (数据记录)
```python
class ChatDataSchema[PayloadT]:
    id: int
    turn_id: int
    step_id: int | None
    kind: str                       # (input/output/reference/intermediate/control)
    payload_type: ChatPayloadTypeEnum
    payload: PayloadT               # 实际数据
    role: str | None                # (user/assistant/system/tool)
```

## 三、Turn执行流程

```
Client                              Server
  │                                   │
  │─────── WebSocket Connect ────────→│
  │←───────── Accept ─────────────────│
  │                                   │
  │─────── turn.start ───────────────→│
  │      {                            │
  │        conversation_id,            │
  │        input: {text},              │
  │        resource_selection         │
  │      }                             │
  │                                   │
  │←───────── turn.accepted ──────────│
  │      {turn_id, conversation}       │
  │                                   │
  │←───────── turn.started ───────────│
  │                                   │
  │←───────── step.created ───────────│  1. 创建根步骤
  │                                   │
  │  ┌─────────────────────────────┐  │
  │  │  Capability Pipeline:        │  │
  │  │                              │  │
  │  │ 1. System Prompt ────────────┼──→│ ← step.created
  │  │    构建系统提示词             │  │   step.started
  │  │                              │  │   data_created (prompt_context)
  │  │ 2. Intent Detection ─────────┼──→│ ← step.completed
  │  │    识别用户意图               │  │
  │  │                              │  │
  │  │ 3. Knowledge Retrieval ──────┼──→│
  │  │    知识库向量检索             │  │
  │  │                              │  │
  │  │ 4. Function Call ────────────┼──→│
  │  │    执行预定义函数             │  │
  │  │    (计算/时间/上下文等)       │  │
  │  │                              │  │
  │  │ 5. Tool Call ────────────────┼──→│  (可选，当前stub模式)
  │  │    外部工具调用               │  │
  │  │                              │  │
  │  │ 6. LLM Response ─────────────┼──→│
  │  │    调用大模型生成回复         │  │   step.started
  │  │                              │  │   message.delta (流式)
  │  │                              │  │   message.completed
  │  │                              │  │ ← step.completed
  │  │                              │  │
  │  └─────────────────────────────┘  │
  │                                   │
  │←───────── turn.completed ─────────│
  │      {                           │
  │        turn: {status, usage}     │
  │      }                           │
  │                                   │
```

## 四、能力系统

### 4.1 能力类型与执行顺序

```python
CAPABILITY_EXECUTION_ORDER = {
    ChatCapabilityKindEnum.system_prompt: 10,      # 1. 系统提示词构建
    ChatCapabilityKindEnum.intent_detection: 20,   # 2. 意图识别
    ChatCapabilityKindEnum.knowledge_retrieval: 30,# 3. 知识库检索
    ChatCapabilityKindEnum.function_call: 40,      # 4. 函数调用
    ChatCapabilityKindEnum.tool_call: 50,          # 5. 工具调用
    ChatCapabilityKindEnum.mcp_call: 60,           # 6. MCP协议调用
    ChatCapabilityKindEnum.llm_response: 90,       # 7. 大模型响应
}
```

### 4.2 能力配置层级

```
Global (系统级)
  └─ CapabilityProfile (能力配置)
      └─ CapabilityBinding (绑定)
          └─ System (所有会话默认)
          └─ Conversation (特定会话)
              └─ Request (请求级内联能力)
```

### 4.3 ResourceSelection 结构

```python
class ResourceSelection:
    use_system_defaults: bool = True       # 使用系统默认能力
    use_conversation_defaults: bool = True # 使用会话默认能力
    capability_profile_ids: list[int]      # 引用的能力配置ID
    capability_binding_ids: list[int]      # 引用的能力绑定ID
    capabilities: list[ResourceCapability] # 内联定义的能力
    metadata: dict                         # 元数据
```

## 五、WebSocket通信协议

### 5.1 客户端命令

| 命令 | 用途 | Payload |
|-----|------|---------|
| `turn.start` | 启动对话回合 | `TurnStartRequest` |
| `turn.cancel` | 取消正在执行的回合 | `TurnCancelRequest` |
| `turn.resume` | 回放回合事件 | `TurnReplayRequest` |
| `ack` | 确认收到事件 | `AckRequest` |
| `ping` | 心跳检测 | `PingRequest` |

### 5.2 服务端事件

| 事件 | 触发时机 | Payload |
|-----|---------|---------|
| `turn.accepted` | Turn已受理 | `TurnEventPayload` |
| `turn.started` | Turn开始执行 | `TurnEventPayload` |
| `turn.completed` | Turn执行完成 | `TurnEventPayload` |
| `turn.failed` | Turn执行失败 | `TurnEventPayload` |
| `turn.canceled` | Turn被取消 | `TurnEventPayload` |
| `step.created` | 步骤已创建 | `StepEventPayload` |
| `step.started` | 步骤开始执行 | `StepEventPayload` |
| `step.completed` | 步骤执行完成 | `StepEventPayload` |
| `step.failed` | 步骤执行失败 | `StepEventPayload` |
| `data.created` | 数据已创建 | `DataEventPayload[T]` |
| `data.completed` | 数据已完成 | `DataEventPayload[T]` |
| `message.delta` | 消息增量(流式) | `MessageDeltaPayload` |
| `message.completed` | 消息完成 | `MessageBundlePayload` |
| `warning` | 警告信息 | `WarningPayload` |
| `error` | 错误信息 | `ErrorPayload` |

## 六、核心组件详解

### 6.1 ChatRuntime (运行时引擎)

**职责**: Turn生命周期管理、能力Pipeline编排

**核心方法**:
```python
class ChatRuntime:
    async def execute_turn(
        self,
        ws_session_id: int | None,
        ws_public_session_id: str | None,
        conversation: ConversationSummary,
        turn_request: TurnStartRequest,
        account_id: int | None,
        is_staff: bool,
        send_event: EventSender,
    ) -> int:
        """执行完整的Turn流程"""

    async def execute_capability(
        self,
        descriptor: CapabilityDescriptor,
        turn,
        root_step,
        session_context: ChatSessionContext,
        cancel_event: asyncio.Event,
        send_event: EventSender,
        emit: Callable,
        seq: int,
        step_sequence: int,
    ) -> int:
        """执行单个能力"""
```

### 6.2 ChatSessionContext (会话上下文)

**职责**: 管理Turn执行过程中的状态和产物

```python
@dataclass
class ChatSessionContext:
    account_id: int | None
    is_staff: bool
    conversation: ConversationSummary
    turn_request: TurnStartRequest
    resolved_selection: ResourceSelection
    resolved_capabilities: list[CapabilityDescriptor]
    artifacts: TurnArtifacts          # 能力执行产物
    prompt_state: SystemPromptState   # 系统提示词状态
    state: dict                       # 临时状态
```

### 6.3 TurnArtifacts (执行产物)

**职责**: 收集各能力执行的结果

```python
@dataclass
class TurnArtifacts:
    instructions: list[str]                    # 额外指令
    context_items: list[CapabilityContextItem] # 上下文项
    prompt_context: PromptContextPayload       # 提示词上下文
    intent_result: IntentRecognitionResult     # 意图识别结果
    executed_functions: list[FunctionExecutionSummary]  # 函数执行摘要
    terminal_output: CapabilityTerminalOutput  # 终态输出
    output_payload: MessageBundlePayload       # 输出消息
    usage: UsagePayload                        # Token使用量
```

### 6.4 CapabilityRegistry (能力注册表)

**职责**: 能力类型注册、配置解析、能力构建

```python
class CapabilityRegistry:
    def register(
        self,
        kind: ChatCapabilityKindEnum,
        step_kind: ChatStepKindEnum,
        default_name: str,
        config_model: type[BaseModel],
    ) -> None:
        """注册能力类型"""

    def build(self, capability: ResourceCapability) -> CapabilityDescriptor:
        """构建能力描述符"""
```

### 6.5 FunctionToolRegistry (函数工具注册表)

**职责**: 内置函数工具注册、匹配、执行

**内置工具**:
- `current_datetime`: 获取当前时间
- `calculate_expression`: 表达式计算
- `session_context`: 会话上下文信息
- `capability_overview`: 能力概览

### 6.6 ChatPromptBuilder (提示词构建器)

**职责**: 根据上下文构建System Prompt和User Prompt

**占位符系统**:
```python
class SystemPromptPlaceholderEnum(StrEnum):
    capability_summary = "能力摘要"
    intent_summary = "意图摘要"
    function_summary = "函数摘要"
    conversation_summary = "会话摘要"
    instructions_summary = "额外约束"
    context_policy = "上下文策略"
```

## 七、扩展开发指南

### 7.1 添加新的能力类型

**步骤**:

1. **定义能力枚举**
```python
# service/chat/domain/schema.py
class ChatCapabilityKindEnum(StrEnum):
    # ...existing...
    my_custom_capability = ("my_custom_capability", "我的自定义能力")
```

2. **定义配置模型**
```python
class MyCustomCapabilityConfig(StrictModel):
    param1: str
    param2: int = Field(default=10)
```

3. **定义ResourceCapability子类**
```python
class MyCustomCapability(BaseResourceCapability):
    kind: Literal[ChatCapabilityKindEnum.my_custom_capability] = ...
    config: MyCustomCapabilityConfig
```

4. **更新Union类型**
```python
ResourceCapability = Annotated[
    IntentDetectionCapability
    | SystemPromptCapability
    # ...
    | MyCustomCapability,  # 添加这里
    Field(discriminator="kind"),
]
```

5. **注册到CapabilityRegistry**
```python
# service/chat/capability/registry.py
def create_default_capability_registry():
    registry = CapabilityRegistry()
    # ...
    registry.register(
        ChatCapabilityKindEnum.my_custom_capability,
        step_kind=ChatStepKindEnum.tool,  # 选择合适的step_kind
        default_name="my_custom_capability",
        config_model=MyCustomCapabilityConfig,
    )
    return registry
```

6. **实现执行逻辑**
```python
# service/chat/runtime/engine.py
class ChatRuntime:
    async def execute_capability(self, descriptor, ...):
        # ...
        if descriptor.kind == ChatCapabilityKindEnum.my_custom_capability:
            return await self.execute_my_custom_capability(...)

    async def execute_my_custom_capability(
        self,
        descriptor,
        *,
        turn,
        root_step,
        session_context,
        emit,
        seq,
        step_sequence,
    ) -> int:
        # 创建step
        step, seq = await self.create_capability_step(...)

        # 执行逻辑
        result = await self.do_something(descriptor.config)

        # 记录结果到artifacts
        session_context.artifacts.add_text_context(
            descriptor,
            text=result,
            title="自定义能力结果"
        )

        # 更新step状态
        await self.repository.update_step(step, status=ChatStepStatusEnum.completed)
        return seq
```

### 7.2 添加新的函数工具

**步骤**:

1. **定义匹配器**
```python
def _match_my_tool(query: str, session: ChatSessionContext) -> float:
    """返回0-1之间的匹配分数"""
    normalized = query.lower()
    return 1.0 if "特定关键词" in normalized else 0.0
```

2. **定义执行器**
```python
async def _execute_my_tool(
    session: ChatSessionContext,
    spec: FunctionToolSpec,
) -> FunctionToolExecutionResult:
    # 执行逻辑
    result = await do_something(session.query)

    return FunctionToolExecutionResult(
        tool_name=spec.tool_name,
        title=spec.title,
        summary="执行摘要",
        text="文本结果",
        data={"key": "value"},  # JSON结果
        prefer_terminal=False,  # True则直接返回
    )
```

3. **注册到FunctionToolRegistry**
```python
# service/chat/runtime/function_tools.py
def create_default_function_tool_registry():
    registry = FunctionToolRegistry()
    # ...
    registry.register(
        "my_tool",
        title="我的工具",
        matcher=_match_my_tool,
        executor=_execute_my_tool,
        default_result_mode=FunctionCallResultModeEnum.context,
    )
    return registry
```

4. **在FunctionCall配置中使用**
```python
{
    "kind": "function_call",
    "config": {
        "tools": [
            {
                "tool_name": "my_tool",
                "title": "我的工具",
                "result_mode": "auto"
            }
        ]
    }
}
```

### 7.3 自定义System Prompt模板

**方式一**: 通过配置变量覆盖
```python
{
    "kind": "system_prompt",
    "config": {
        "variable_overrides": {
            "assistant_identity": "你是一个专业的客服助手",
            "response_policy": "回答要友善、专业、简洁"
        }
    }
}
```

**方式二**: 修改占位符渲染逻辑
```python
# service/chat/runtime/prompting.py
class ChatPromptBuilder:
    def render_placeholder(self, placeholder, context, session, overrides):
        # ...
        if placeholder == SystemPromptPlaceholderEnum.my_custom:
            return self.render_my_custom(context)
```

### 7.4 自定义LLM调用

当前系统使用`pydantic-ai`进行LLM调用，主要实现在:

```python
# service/chat/runtime/engine.py
class ChatRuntime:
    async def generate_response(
        self,
        query: str,
        llm_model_config_id: int | None,
        context: ChatContextEnvelope,
        session_context: ChatSessionContext,
        cancel_event: asyncio.Event,
        send_delta: Callable[[str], Awaitable[None]],
    ) -> tuple[str, UsagePayload]:
        # 构建提示词
        bundle = self.prompt_builder.build(
            query=query,
            context=context,
            session=session_context,
        )

        # 调用LLM (这里使用pydantic-ai)
        # 返回(响应文本, 使用量)
```

如需支持其他LLM提供商，修改此方法即可。

## 八、API接口

### 8.1 WebSocket接口
```
WS /api/v1/chat/ws
```

### 8.2 REST接口
```
# 会话管理
GET  /api/v1/chat/conversation                    # 列表
GET  /api/v1/chat/conversation/{id}               # 详情
GET  /api/v1/chat/conversation/{id}/timeline      # 时间线
PUT  /api/v1/chat/conversation/{id}/resource-selection  # 更新默认能力

# Turn管理
GET  /api/v1/chat/turn/{turn_id}/events          # 回放事件

# 能力配置管理
POST   /api/v1/chat/capability/profile           # 创建配置
PUT    /api/v1/chat/capability/profile/{id}      # 更新配置
GET    /api/v1/chat/capability/profile           # 列表
GET    /api/v1/chat/capability/profile/{id}      # 详情
DELETE /api/v1/chat/capability/profile/{id}      # 删除配置

# 能力绑定管理
POST   /api/v1/chat/capability/binding           # 创建绑定
PUT    /api/v1/chat/capability/binding/{id}      # 更新绑定
GET    /api/v1/chat/capability/binding           # 列表
GET    /api/v1/chat/capability/binding/{id}      # 详情
DELETE /api/v1/chat/capability/binding/{id}      # 删除绑定

# Demo
GET  /api/v1/chat/demo                           # 演示页面
```

## 九、文件清单

```
service/chat/
├── __init__.py
├── domain/schema.py                    # 领域模型定义
├── application/service.py              # 应用服务层
├── runtime/
│   ├── engine.py                       # 运行时引擎(核心)
│   ├── session.py                      # 会话上下文
│   ├── context.py                      # 执行产物管理
│   ├── prompting.py                    # 提示词构建
│   └── function_tools.py               # 函数工具注册
├── capability/
│   ├── registry.py                     # 能力注册表
│   ├── service.py                      # 能力管理服务
│   ├── repository.py                   # 能力数据访问
│   └── schema.py                       # 能力相关Schema
└── store/
    └── repository.py                   # 聊天数据仓库

api/knowledge_base/v1/
├── chat.py                             # WebSocket/REST接口
└── chat_capability.py                  # 能力管理接口
```
