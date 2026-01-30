# Workflow Task 核心逻辑文档

本文档详细说明工作流系统中三个核心任务的核心逻辑和实现。

---

## 目录

1. [Entry Task - 工作流入口任务](#1-entry-task---工作流入口任务)
2. [Activity Task - 活动任务](#2-activity-task---活动任务)
3. [Handoff Task - 活动交接任务](#3-handoff-task---活动交接任务)
4. [完整流程示例](#4-完整流程示例)

---

## 1. Entry Task - 工作流入口任务

### 职责
- 启动新工作流或恢复已存在的工作流
- 将 workflow 状态从 `pending/failed` 转换为 `running`
- 启动第一批就绪的 activity tasks

### Celery Task 定义

```python
@celery_app.task(name="workflow.schedule_workflow_entry", bind=True, queue="workflow_entry")
def _schedule_workflow_celery_entry(
    celery_task: CeleryTask,
    workflow_uid: str | None = None,
    config: dict | None = None,
    config_format: str = "dict",
    initial_inputs: dict | None = None,
) -> str:
```

**队列**: `workflow_entry`
**用途**: 通过外部系统（如 HTTP API）触发工作流，fire-and-forget

---

### 核心逻辑流程

```
Entry Task 被调用
    │
    ├─→ 检查 workflow_uid 或 config 参数
    │   ├─ 有 workflow_uid: 恢复已存在的工作流
    │   └─ 有 config: 创建新工作流
    │
    ├─→ 检查工作流状态
    │   ├─ completed/canceled: 直接返回，无需处理
    │   └─ pending/failed: 更新为 running
    │
    ├─→ 构建任务依赖图 (DAG)
    │
    ├─→ 获取就绪的活动
    │   └─→ 所有前置依赖已完成的 activity
    │
    └─→ 根据 execute_mode 启动活动
        ├─ direct: 直接执行并等待完成
        └─ celery: 派发到 Celery default 队列
```

---

### 核心代码实现

```python
@staticmethod
async def schedule_workflow(
    workflow_uid: uuid.UUID | str | None = None,
    config: dict | None = None,
    config_format: str = "dict",
    initial_inputs: dict | None = None,
    execute_mode: str = "direct",
) -> str:
    # 1. 参数处理
    if isinstance(workflow_uid, str):
        workflow_uid = uuid.UUID(workflow_uid)

    # 2. 创建或获取工作流
    if workflow_uid:
        workflow = await WorkflowManager.get_workflow_by_uid(workflow_uid)
    else:
        if not config:
            raise ValueError("Either workflow_uid or config must be provided")
        workflow = await WorkflowManager.create_workflow(config, config_format, initial_inputs)

    # 3. 检查工作流状态
    workflow.status = WorkflowStatusEnum(workflow.status)

    if workflow.status in [
        WorkflowStatusEnum.completed,
        WorkflowStatusEnum.canceled,
    ]:
        logger.info(f"Workflow {workflow.uid} already {workflow.status.value}")
        return str(workflow.uid)

    # 4. 构建依赖图
    graph = WorkflowManager.build_graph(workflow)

    # 5. 更新工作流状态为 running
    if workflow.status in [
        WorkflowStatusEnum.pending,
        WorkflowStatusEnum.failed,
    ]:
        await WorkflowManager.update_workflow_status(
            workflow.uid,
            WorkflowStatusEnum.running,
            started_at=datetime.now() if not workflow.started_at else None,
        )

    # 6. 获取就绪的活动
    ready_activities = await WorkflowManager.get_ready_activities(workflow.uid, graph)
    logger.info(f"Ready activities: {[a.name for a in ready_activities]}")

    # 7. 根据执行模式启动活动
    if execute_mode == "direct":
        await WorkflowScheduler._launch_activity_tasks_direct(ready_activities)

        # Direct mode: 轮询等待工作流完成
        poll_interval = 0.5
        while True:
            await asyncio.sleep(poll_interval)
            workflow_check = await WorkflowManager.get_workflow_by_uid(workflow.uid)
            current_status = WorkflowStatusEnum(workflow_check.status)

            if current_status in [
                WorkflowStatusEnum.completed,
                WorkflowStatusEnum.failed,
                WorkflowStatusEnum.canceled,
            ]:
                break
        logger.info(f"Workflow {workflow.uid} finished in direct mode")
    else:
        # Celery mode: 派发到 Celery 队列，立即返回
        await WorkflowScheduler._launch_activity_tasks_celery(ready_activities)

    return str(workflow.uid)
```

---

### 关键逻辑点

| 逻辑点 | 说明 |
|---------|------|
| **工作流恢复** | 支持通过 `workflow_uid` 恢复已存在的工作流（从失败状态重新执行） |
| **状态转换** | 只从 `pending/failed` 转换到 `running`，避免重复处理已完成的工作流 |
| **依赖解析** | 使用 DAG 构建任务依赖关系，只启动前置任务已完成的活动 |
| **Direct Mode** | 阻塞等待工作流完成，适合测试和简单场景 |
| **Celery Mode** | Fire-and-forget，立即返回，适合生产环境 |

---

## 2. Activity Task - 活动任务

### 职责
- 执行用户定义的业务逻辑
- 管理 activity 状态生命周期
- 完成后触发 handoff 任务

### 基类定义

```python
@activity_task
class MyTask(ActivityTaskTemplate):
    async def execute(self) -> dict[str, Any]:
        # 用户业务逻辑
        return {"result": "success"}
```

**队列**: `default` (所有 activity tasks 共享)
**执行者**: 任意 Celery worker (可配置并发数)

---

### 核心逻辑流程

```
Activity Task 被调用
    │
    ├─→ 1. 加载 activity 和 workflow 数据
    │   └─→ _load_activity()
    │
    ├─→ 2. 设置状态为 running
    │   └─→ _set_running()
    │
    ├─→ 3. 执行用户业务逻辑
    │   └─→ execute() [用户实现的方法]
    │   ├─ 成功
    │   │   └─→ _set_completed()
    │   └─ 失败
    │       └─→ _handle_exception()
    │
    └─→ 4. 触发 handoff
        ├─ celery mode: 派发 handoff task 到 workflow_handoff 队列
        └─ direct mode: 直接调用 handoff 逻辑
```

---

### 核心代码实现

#### 生命周期入口

```python
async def _execute_lifecycle(self) -> dict[str, Any]:
    """Full execution lifecycle

    Automatically handles:
    1. Load activity and workflow
    2. Set status to running
    3. Execute user logic
    4. Save output and set status to completed
    5. Trigger activity handoff
    """
    await self._load_activity()

    with logger.contextualize(trace_id=str(self.activity.workflow_uid), activity_uid=str(self.activity.uid)):
        try:
            await self._set_running()
            output = await self.execute()
            await self._set_completed(output)
            return output
        except Exception as e:
            await self._handle_exception(e)
            raise
```

#### 设置运行状态

```python
async def _set_running(self) -> None:
    """Set activity status to running"""
    await WorkflowManager.update_activity_status(
        self.activity_uid,
        ActivityStatusEnum.running,
        started_at=datetime.now(),
    )
```

#### 设置完成状态

```python
async def _set_completed(self, output: dict[str, Any]) -> None:
    """Set activity status to completed and trigger handoff"""
    await WorkflowManager.update_activity_status(
        self.activity_uid,
        ActivityStatusEnum.completed,
        output=output,
        completed_at=datetime.now(),
    )

    # 根据 execute_mode 触发 handoff
    if self.execute_mode == "celery":
        # Celery mode: 派发 handoff task
        _schedule_activity_handoff_celery_entry.apply_async(args=[str(self.activity_uid)])
    else:
        # Direct mode: 直接调用 handoff 逻辑
        await WorkflowScheduler.schedule_activity_handoff(self.activity_uid, execute_mode=self.execute_mode)
```

#### 异常处理

```python
async def _handle_exception(self, exception: Exception) -> None:
    """Handle exception and set appropriate status"""
    error_message = str(exception)
    stack_trace = traceback.format_exc()

    logger.error(f"Activity failed: {error_message}\n{stack_trace}")

    activity = await WorkflowManager.get_activity_by_uid(self.activity_uid)

    max_retries = activity.execute_params.get("max_retries", 3)
    current_retry = activity.retry_count

    if current_retry < max_retries:
        # 重试中
        await WorkflowManager.update_activity_status(
            self.activity_uid,
            ActivityStatusEnum.retrying,
            error_message=error_message,
            stack_trace=stack_trace,
            increment_retry=True,
        )
    else:
        # 失败
        await WorkflowManager.update_activity_status(
            self.activity_uid,
            ActivityStatusEnum.failed,
            error_message=error_message,
            stack_trace=stack_trace,
            completed_at=datetime.now(),
        )

    # 仍然需要触发 handoff（处理失败状态）
    if self.execute_mode == "celery":
        _schedule_activity_handoff_celery_entry.apply_async(args=[str(self.activity_uid)])
    else:
        await WorkflowScheduler.schedule_activity_handoff(self.activity_uid, execute_mode=self.execute_mode)
```

---

### 关键逻辑点

| 逻辑点 | 说明 |
|---------|------|
| **状态生命周期** | `pending` → `running` → `completed/failed/retrying` |
| **重试机制** | 支持自动重试，默认最多 3 次，可在 `execute_params` 中配置 |
| **错误处理** | 捕获用户异常，记录堆栈，更新 activity 状态 |
| **输出传递** | 完成后的 output 会自动传递给下游 activity 的 input |
| **Handoff 触发** | 无论成功或失败，都会触发 handoff 来处理下游逻辑 |

---

## 3. Handoff Task - 活动交接任务

### 职责
- 检查活动状态（成功/失败）
- 传播活动输出到下游活动
- 查找并启动就绪的下游活动
- 检查工作流是否全部完成，更新工作流状态

### Celery Task 定义

```python
@celery_app.task(name="workflow.activity_handoff_entry", bind=True, queue="workflow_handoff")
def _schedule_activity_handoff_celery_entry(celery_task: CeleryTask, activity_uid: str) -> str:
```

**队列**: `workflow_handoff`
**并发限制**: `concurrency=1` (串行执行，防止并发导致的竞态条件)

---

### 核心逻辑流程

```
Handoff Task 被调用
    │
    ├─→ 1. 加载 activity 数据
    │
    ├─→ 2. 检查工作流状态
    │   ├─ 已完成/失败/取消: 跳过处理
    │   └─ running: 继续
    │
    ├─→ 3. 检查 activity 状态
    │   ├─ 失败: 标记工作流失败，返回
    │   └─ 成功: 继续
    │
    ├─→ 4. 传播输出到下游
    │   └─→ 将当前 activity 的 output 更新到下游 activity 的 input
    │
    ├─→ 5. 获取就绪的下游活动
    │   └─→ 通过 DAG 查找所有依赖已满足的 activity
    │
    ├─→ 6. 启动下游活动
    │   ├─ direct mode: 直接执行
    │   └─ celery mode: 派发到 Celery default 队列
    │
    └─→ 7. 检查工作流是否完成
        └─→ 如果没有更多就绪活动，检查所有 activity 是否都为终态
            ├─ 是: 更新 workflow 状态为 completed/failed
            └─ 否: 继续（等待其他 activity 完成）
```

---

### 核心代码实现

```python
@staticmethod
async def schedule_activity_handoff(
    activity_uid: uuid.UUID | str,
    execute_mode: str = "direct",
) -> str:
    """Schedule activity handoff to downstream activities"""
    if isinstance(activity_uid, str):
        activity_uid = uuid.UUID(activity_uid)

    # 1. 加载 activity 数据
    activity = await WorkflowManager.get_activity_by_uid(activity_uid)

    with logger.contextualize(trace_id=str(activity.workflow_uid), activity_uid=str(activity.uid)):
        logger.info(f"Activity handoff: {activity.name}, status={activity.status}, mode={execute_mode}")

        # 2. 检查工作流状态
        workflow = await WorkflowManager.get_workflow_by_uid(activity.workflow_uid)

        if workflow.status in [
            WorkflowStatusEnum.completed.value,
            WorkflowStatusEnum.failed.value,
            WorkflowStatusEnum.canceled.value,
        ]:
            logger.info(f"Workflow {workflow.uid} is {workflow.status}, skipping")
            return str(activity.uid)

        # 3. 检查 activity 状态
        if WorkflowManager.is_failed_status(ActivityStatusEnum(activity.status)):
            logger.warning(f"Activity {activity.name} failed, marking workflow failed")
            await WorkflowManager.mark_workflow_failed(
                activity.workflow_uid,
                f"Activity {activity.name} failed: {activity.error_message}",
            )
            return str(activity.uid)

        # 4. 传播输出到下游
        graph = WorkflowManager.build_graph(workflow)
        await WorkflowManager.propagate_output_to_downstream(activity, graph)

        # 5. 获取就绪的下游活动
        ready_activities = await WorkflowManager.get_ready_activities(activity.workflow_uid, graph)
        logger.info(f"Ready downstream: {[a.name for a in ready_activities]}")

        # 6. 启动下游活动
        if execute_mode == "direct":
            await WorkflowScheduler._launch_activity_tasks_direct(ready_activities)
        else:
            await WorkflowScheduler._launch_activity_tasks_celery(ready_activities)

        # 7. 检查工作流是否完成
        if not ready_activities:
            workflow_uid_str = str(activity.workflow_uid)
            lock = WorkflowScheduler._completion_locks.get(workflow_uid_str)
            if lock is None:
                lock = asyncio.Lock()
                WorkflowScheduler._completion_locks[workflow_uid_str] = lock

            async with lock:
                is_completed = await WorkflowManager.is_workflow_completed(activity.workflow_uid)
                if is_completed:
                    logger.info(f"Workflow {workflow.uid} completed in {execute_mode} mode")

        return str(activity.uid)
```

---

### 工作流完成检查

```python
@staticmethod
async def is_workflow_completed(workflow_uid: uuid.UUID) -> bool:
    """Check if workflow is completed and update status"""
    with logger.contextualize(trace_id=str(workflow_uid)):
        activities = await WorkflowManager.get_activities_by_workflow(workflow_uid)

        # 检查是否有失败的 activity
        has_failed = any(
            WorkflowManager.is_failed_status(ActivityStatusEnum(act.status))
            for act in activities
        )

        # 检查是否所有 activity 都已完成
        all_finished = all(
            WorkflowManager.is_terminal_status(ActivityStatusEnum(act.status))
            for act in activities
        )

        if not all_finished:
            return False

        # 检查工作流是否已经在终态
        workflow = await WorkflowManager.get_workflow_by_uid(workflow_uid)
        current_status = WorkflowStatusEnum(workflow.status)

        if current_status in [
            WorkflowStatusEnum.completed,
            WorkflowStatusEnum.failed,
            WorkflowStatusEnum.canceled,
        ]:
            return True

        # 更新工作流状态
        if has_failed:
            await WorkflowManager.update_workflow_status(
                workflow_uid,
                WorkflowStatusEnum.failed,
                completed_at=datetime.now(),
            )
            logger.info(f"Workflow {workflow_uid} marked as failed")
        else:
            await WorkflowManager.update_workflow_status(
                workflow_uid,
                WorkflowStatusEnum.completed,
                completed_at=datetime.now(),
            )

        return True
```

---

### 关键逻辑点

| 逻辑点 | 说明 |
|---------|------|
| **输出传播** | 将上游 activity 的 output 合并到下游 activity 的 input |
| **失败处理** | 任何 activity 失败都会标记整个工作流为 failed，并取消所有 pending 的 activity |
| **竞态条件防护** | 使用 `concurrency=1` 和 `completion_locks` 确保串行执行 |
| **完成检查** | 只有当所有 activity 都为终态（completed/failed/canceled）时才更新 workflow 状态 |
| **终态判断** | `completed`, `failed`, `canceled` 为终态，无法再转换 |

---

## 4. 完整流程示例

### Direct Mode (直接执行模式)

```
Entry Task (直接调用)
    │
    ├─→ 启动: FetchFileTask (concurrent)
    │
FetchFileTask 完成
    │
    ├─→ Handoff (inline)
    │   ├─→ 传播输出
    │   ├─→ 启动: LoadFileTask
    │   └─→ 检查完成 (未完成)
    │
LoadFileTask 完成
    │
    ├─→ Handoff (inline)
    │   ├─→ 传播输出
    │   ├─→ 启动: ReplaceContentTask
    │   └─→ 检查完成 (未完成)
    │
ReplaceContentTask 完成
    │
    ├─→ Handoff (inline)
    │   ├─→ 传播输出
    │   ├─→ 启动: SummaryTask
    │   └─→ 检查完成 (未完成)
    │
SummaryTask 完成
    │
    └─→ Handoff (inline)
        ├─→ 传播输出
        ├─→ 启动: (无下游)
        └─→ 检查完成 ✓
            └─→ Workflow 状态: completed
```

---

### Celery Mode (分布式执行模式)

```
Entry Task (workflow_entry queue)
    │
    └─→ 派发到 default: FetchFileTask
    │
┌─────────────────────────────────────────────────────────┐
│ default workers (concurrency=4)                 │
├─────────────────────────────────────────────────────┤
│                                                 │
│  FetchFileTask 完成                               │
│      │                                          │
│      └─→ 派发到 handoff: handoff task    │
│                                                 │
├─────────────────────────────────────────────────────┤
│                                                 │
│ handoff worker (concurrency=1, 串行)           │
├─────────────────────────────────────────────────────┤
│                                                 │
│  Handoff 执行:                                   │
│      ├─→ 传播输出                                 │
│      ├─→ 派发到 default: LoadFileTask         │
│      └─→ 检查完成 (未完成)                       │
│                                                 │
├─────────────────────────────────────────────────────┤
│                                                 │
│ default workers                                   │
├─────────────────────────────────────────────────────┤
│                                                 │
│  LoadFileTask 完成                               │
│      │                                          │
│      └─→ 派发到 handoff: handoff task    │
│                                                 │
│ handoff worker                                   │
│  Handoff 执行:                                   │
│      ├─→ 传播输出                                 │
│      ├─→ 派发到 default: ReplaceContentTask │
│      └─→ 检查完成 (未完成)                       │
│                                                 │
│  ... (继续相同流程)                              │
│                                                 │
│  SummaryTask 完成                                │
│      │                                          │
│      └─→ 派发到 handoff: handoff task    │
│                                                 │
│ handoff worker                                   │
│  Handoff 执行:                                   │
│      ├─→ 传播输出                                 │
│      ├─→ 派发到 default: (无)                 │
│      └─→ 检查完成 ✓                           │
│          └─→ Workflow 状态: completed                │
└─────────────────────────────────────────────────────────┘
```

---

## Worker 配置

```bash
# 1. Handoff worker (串行执行)
uv run celery -A ext.ext_celery.worker worker -Q workflow_handoff -c 1 -n handoff

# 2. Workflow entry worker (可选)
uv run celery -A ext.ext_celery.worker worker -Q workflow_entry -c 4 -n entry

# 3. Activity tasks workers (高并发)
uv run celery -A ext.ext_celery.worker worker -c 4 -n default
```

---

## 总结

| Task | 队列 | 并发 | 职责 |
|------|--------|-------|--------|
| **Entry Task** | `workflow_entry` | 4 | 启动工作流，初始化第一批活动 |
| **Activity Task** | `default` | 4 | 执行用户业务逻辑 |
| **Handoff Task** | `workflow_handoff` | 1 | 传播输出，启动下游活动，检查工作流完成 |

### 关键设计原则

1. **Entry**: Fire-and-forget，支持异步触发
2. **Activity**: 业务逻辑封装，自动状态管理
3. **Handoff**: 串行执行，防止竞态条件，确保工作流状态一致性
4. **两种模式**: Direct（测试）和 Celery（生产）无缝切换
