"""
调度任务

实现三种调度任务：
1. 一类调度任务（重头执行入口）
2. 二类调度任务（断点继续执行入口）
3. 三类调度任务（activity handoff）

支持两种调用方式：
1. 直接调用（同步方式）
2. Celery apply_async 调用（异步方式）
"""
import uuid
import asyncio
from loguru import logger
from datetime import datetime
from typing import Any, Dict

from celery import Task as CeleryTask
from ext.ext_celery.app import celery_app
from ext.ext_tortoise.enums import ActivityStatusEnum, WorkflowStatusEnum
from ext.ext_tortoise.models.knowledge_base import Activity, Workflow


def run_async(coro):
    """
    运行异步协程，自动检测环境

    在 Celery worker 环境中，复用 worker 的事件循环（不创建新循环）
    在非 worker 环境（测试、脚本）中，创建新的事件循环运行
    """
    # 尝试获取当前运行的事件循环
    loop = asyncio.get_event_loop()
    # 如果已有循环在运行，使用 run_coroutine_threadsafe 并等待结果
    if loop.is_running():
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        # 等待协程完成并返回结果
        return future.result()
    else:
        # 循环存在但未运行，直接运行
        return loop.run_until_complete(coro)


async def _schedule_workflow_start_async(
    config: Dict[str, Any],
    config_format: str = "dict",
    initial_inputs: Dict[str, Any] = None,
    use_async: bool = True,
) -> str:
    """一类调度任务的异步实现

    Args:
        config: 工作流配置
        config_format: 配置格式
        initial_inputs: 初始输入
        use_async: 是否使用 Celery apply_async，False 表示直接调用（用于本地调试）
    """
    from .manager import WorkflowManager

    # 1. 创建工作流和所有活动记录
    workflow = await WorkflowManager.create_workflow(
        config, config_format, initial_inputs
    )
    logger.info(f"Created workflow: {workflow.uid}, use_async={use_async}")

    # 2. 构建 DAG 图
    graph = WorkflowManager.build_graph(workflow)
    logger.info(f"Built DAG graph for workflow: {workflow.uid}")

    # 3. 更新工作流状态为运行中
    await WorkflowManager.update_workflow_status(
        str(workflow.uid),
        WorkflowStatusEnum.running,
        started_at=datetime.now(),
    )

    # 4. 获取第一批准备执行的活动（根节点）
    ready_activities = await WorkflowManager.get_ready_activities(
        str(workflow.uid), graph
    )
    logger.info(f"Ready activities: {[act.name for act in ready_activities]}")

    # 5. 启动第一批活动的任务
    for activity in ready_activities:
        task_name = activity.execute_params.get("task_name")
        if not task_name:
            logger.warning(f"Activity {activity.name} has no task_name, skipping")
            continue

        # 获取任务函数
        task = celery_app.tasks.get(task_name)
        if not task:
            logger.error(f"Task not found: {task_name}")
            continue

        if use_async:
            # 使用 apply_async 启动任务（异步）
            celery_task_result = task.apply_async(args=[str(activity.uid)])
            await WorkflowManager.update_activity_status(
                str(activity.uid),
                ActivityStatusEnum.pending,
                celery_task_id=celery_task_result.id,
            )
            logger.info(
                f"Started task {task_name} for activity {activity.name} (async), "
                f"task_id: {celery_task_result.id}"
            )
        else:
            # 直接调用任务（同步，用于本地调试）
            # 直接调用不会返回 celery_task_id
            logger.info(f"Starting task {task_name} for activity {activity.name} (direct call)")
            await task.async_call(str(activity.uid), use_async)
            logger.info(
                f"Completed task {task_name} for activity {activity.name} (direct call)"
            )

    return str(workflow.uid)


async def schedule_workflow_start(
    config: Dict[str, Any],
    config_format: str = "dict",
    initial_inputs: Dict[str, Any] = None,
    use_async: bool = True,
) -> str:
    """
    一类调度任务（重头执行入口）- 异步版本

    接收 config 和初始输入，初始化创建 workflow/完整 activity 记录，
    然后按 dag 启动第一批 activity 的 celery 任务

    Args:
        config: 工作流配置（DAG 结构）
        config_format: 配置格式（yaml/json/dict）
        initial_inputs: 初始输入数据
        use_async: 是否使用 Celery apply_async，False 表示直接调用（用于本地调试）

    Returns:
        Workflow UID
    """
    return await _schedule_workflow_start_async(config, config_format, initial_inputs, use_async)


def schedule_workflow_start_sync(
    config: Dict[str, Any],
    config_format: str = "dict",
    initial_inputs: Dict[str, Any] = None,
    use_async: bool = True,
) -> str:
    """
    一类调度任务（重头执行入口）- 同步版本

    接收 config 和初始输入，初始化创建 workflow/完整 activity 记录，
    然后按 dag 启动第一批 activity 的 celery 任务

    Args:
        config: 工作流配置（DAG 结构）
        config_format: 配置格式（yaml/json/dict）
        initial_inputs: 初始输入数据
        use_async: 是否使用 Celery apply_async，False 表示直接调用（用于本地调试）

    Returns:
        Workflow UID
    """
    return run_async(
        _schedule_workflow_start_async(config, config_format, initial_inputs, use_async)
    )


async def _schedule_workflow_resume_async(workflow_uid: uuid.UUID, use_async: bool = True) -> str:
    """二类调度任务的异步实现

    Args:
        workflow_uid: 工作流 UID
        use_async: 是否使用 Celery apply_async，False 表示直接调用（用于本地调试）
    """
    from .manager import WorkflowManager

    # 1. 获取工作流
    workflow = await WorkflowManager.get_workflow_by_uid(workflow_uid)
    if not workflow:
        raise ValueError(f"Workflow not found: {workflow_uid}")

    logger.info(f"Resuming workflow: {workflow_uid}, current status: {workflow.status}, use_async={use_async}")

    # 2. 检查工作流状态
    if workflow.status in [
        WorkflowStatusEnum.completed.value,
        WorkflowStatusEnum.canceled.value,
    ]:
        logger.info(f"Workflow {workflow_uid} is already {workflow.status}, skipping")
        return workflow_uid

    # 3. 构建 DAG 图
    graph = WorkflowManager.build_graph(workflow)

    # 4. 更新工作流状态为运行中（如果之前是失败状态）
    if workflow.status in [
        WorkflowStatusEnum.pending.value,
        WorkflowStatusEnum.failed.value,
    ]:
        await WorkflowManager.update_workflow_status(
            workflow_uid,
            WorkflowStatusEnum.running,
            started_at=datetime.now() if not workflow.started_at else None,
        )

    # 5. 获取准备执行的活动
    ready_activities = await WorkflowManager.get_ready_activities(workflow_uid, graph)
    logger.info(f"Ready activities to resume: {[act.name for act in ready_activities]}")

    # 6. 启动准备执行的活动
    for activity in ready_activities:
        task_name = activity.execute_params.get("task_name")
        if not task_name:
            logger.warning(f"Activity {activity.name} has no task_name, skipping")
            continue

        # 获取任务函数
        task = celery_app.tasks.get(task_name)
        if not task:
            logger.error(f"Task not found: {task_name}")
            continue

        if use_async:
            # 使用 apply_async 启动任务
            celery_task_result = task.apply_async(args=[str(activity.uid)])
            await WorkflowManager.update_activity_status(
                str(activity.uid),
                ActivityStatusEnum.pending,
                celery_task_id=celery_task_result.id,
            )
            logger.info(
                f"Started task {task_name} for activity {activity.name} (async), "
                f"task_id: {celery_task_result.id}"
            )
        else:
            # 直接调用任务（同步，用于本地调试）
            logger.info(f"Starting task {task_name} for activity {activity.name} (direct call)")
            await task.async_call(str(activity.uid), use_async)
            logger.info(
                f"Completed task {task_name} for activity {activity.name} (direct call)"
            )

    return workflow_uid


async def schedule_workflow_resume(workflow_uid: uuid.UUID, use_async: bool = True) -> str:
    """
    二类调度任务（断点继续执行入口）- 异步版本

    接收 workflow uid，继续执行状态为终态且不为已完成的 activity

    Args:
        workflow_uid: 工作流 UID
        use_async: 是否使用 Celery apply_async，False 表示直接调用（用于本地调试）

    Returns:
        Workflow UID
    """
    return await _schedule_workflow_resume_async(workflow_uid, use_async)


def schedule_workflow_resume_sync(workflow_uid: uuid.UUID, use_async: bool = True) -> str:
    """
    二类调度任务（断点继续执行入口）- 同步版本

    接收 workflow uid，继续执行状态为终态且不为已完成的 activity

    Args:
        workflow_uid: 工作流 UID
        use_async: 是否使用 Celery apply_async，False 表示直接调用（用于本地调试）

    Returns:
        Workflow UID
    """
    return run_async(_schedule_workflow_resume_async(workflow_uid, use_async))


async def _schedule_activity_handoff_async(activity_uid: str, use_async: bool = True) -> str:
    """三类调度任务的异步实现

    Args:
        activity_uid: Activity UID
        use_async: 是否使用 Celery apply_async，False 表示直接调用（用于本地调试）
    """
    from .manager import WorkflowManager

    # 1. 加载 Activity 记录
    activity = await Activity.filter(uid=activity_uid).first()
    if not activity:
        raise ValueError(f"Activity not found: {activity_uid}")

    logger.info(
        f"Activity handoff: {activity.name} (status: {activity.status}), "
        f"workflow: {activity.workflow_uid}, use_async={use_async}"
    )

    # 2. 检查工作流是否已经失败或取消
    workflow = await WorkflowManager.get_workflow_by_uid(activity.workflow_uid)
    if not workflow:
        raise ValueError(f"Workflow not found: {activity.workflow_uid}")

    if workflow.status in [
        WorkflowStatusEnum.completed.value,
        WorkflowStatusEnum.failed.value,
        WorkflowStatusEnum.canceled.value,
    ]:
        logger.info(
            f"Workflow {activity.workflow_uid} is already {workflow.status}, "
            "skipping handoff"
        )
        return activity_uid

    # 3. 检查 Activity 是否失败，如果失败则标记工作流失败
    if activity.status in [
        ActivityStatusEnum.failed.value,
        ActivityStatusEnum.canceled.value,
    ]:
        logger.warning(f"Activity {activity.name} failed, marking workflow as failed")
        await WorkflowManager.mark_workflow_failed(
            activity.workflow_uid,
            f"Activity {activity.name} failed: {activity.error_message}",
        )
        return activity_uid

    # 4. 将输出传播到下游活动的输入
    graph = WorkflowManager.build_graph(workflow)
    await WorkflowManager.propagate_output_to_downstream(activity, graph)

    # 5. 获取准备执行的下游活动
    ready_activities = await WorkflowManager.get_ready_activities(
        activity.workflow_uid, graph
    )
    logger.info(f"Ready downstream activities: {[act.name for act in ready_activities]}")

    # 6. 启动准备执行的下游活动
    for ready_activity in ready_activities:
        task_name = ready_activity.execute_params.get("task_name")
        if not task_name:
            logger.warning(f"Activity {ready_activity.name} has no task_name, skipping")
            continue

        # 获取任务函数
        task = celery_app.tasks.get(task_name)
        if not task:
            logger.error(f"Task not found: {task_name}")
            continue

        if use_async:
            # 使用 apply_async 启动任务
            celery_task_result = task.apply_async(args=[str(ready_activity.uid)])
            await WorkflowManager.update_activity_status(
                str(ready_activity.uid),
                ActivityStatusEnum.pending,
                celery_task_id=celery_task_result.id,
            )
            logger.info(
                f"Started task {task_name} for downstream activity {ready_activity.name} (async), "
                f"task_id: {celery_task_result.id}"
            )
        else:
            # 直接调用任务（同步，用于本地调试）
            logger.info(
                f"Starting task {task_name} for downstream activity {ready_activity.name} (direct call)"
            )
            await task.async_call(str(ready_activity.uid), use_async)
            logger.info(
                f"Completed task {task_name} for downstream activity {ready_activity.name} (direct call)"
            )

    # 7. 检查工作流是否完成
    is_completed = await WorkflowManager.check_workflow_completion(
        activity.workflow_uid
    )
    if is_completed:
        logger.info(f"Workflow {activity.workflow_uid} completed")

    return activity_uid


async def schedule_activity_handoff(activity_uid: str, use_async: bool = True) -> str:
    """
    三类调度任务（activity handoff）- 异步版本

    Activity 任务流转管理，当某个 activity 任务执行完之后会启动该调度任务。
    查询 dag 配置和该任务的下游任务能否执行，当下游任务依赖多个前置任务时，
    必须保证所有的前置任务（包括同级并行的）全部已完成。
    如果存在任务为失败的，无法继续后续执行，则将 workflow 状态置为失败。
    该 celery 任务只能单个执行，不能并发。

    Args:
        activity_uid: 完成的 Activity UID
        use_async: 是否使用 Celery apply_async，False 表示直接调用（用于本地调试）

    Returns:
        Activity UID
    """
    return await _schedule_activity_handoff_async(activity_uid, use_async)


def schedule_activity_handoff_sync(activity_uid: str, use_async: bool = True) -> str:
    """
    三类调度任务（activity handoff）- 同步版本

    Activity 任务流转管理，当某个 activity 任务执行完之后会启动该调度任务。
    查询 dag 配置和该任务的下游任务能否执行，当下游任务依赖多个前置任务时，
    必须保证所有的前置任务（包括同级并行的）全部已完成。
    如果存在任务为失败的，无法继续后续执行，则将 workflow 状态置为失败。
    该 celery 任务只能单个执行，不能并发。

    Args:
        activity_uid: 完成的 Activity UID
        use_async: 是否使用 Celery apply_async，False 表示直接调用（用于本地调试）

    Returns:
        Activity UID
    """
    return run_async(_schedule_activity_handoff_async(activity_uid, use_async))


# Celery Task 装饰器包装
# 这些任务用于 Celery worker 执行，内部会调用上述同步函数


@celery_app.task(
    name="workflow.schedule_start",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def schedule_workflow_start_celery(
    celery_task: CeleryTask,
    config: Dict[str, Any],
    config_format: str = "dict",
    initial_inputs: Dict[str, Any] = None,
    use_async: bool = True,
    queyue="workflow_entry"
) -> str:
    """Celery 版本的一类调度任务
    uv run celery -A ext.ext_celery.worker worker -l info -Q workflow_entry -c 1 -n activity_entry_worker@%%h
    """
    return schedule_workflow_start_sync(config, config_format, initial_inputs, use_async)


@celery_app.task(
    name="workflow.schedule_resume",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    queyue="workflow_entry"
)
def schedule_workflow_resume_celery(
    celery_task: CeleryTask,
    workflow_uid: uuid.UUID,
    use_async: bool = True,
) -> str:
    """Celery 版本的二类调度任务
    uv run celery -A ext.ext_celery.worker worker -l info -Q workflow_entry -c 1 -n activity_entry_worker@%%h
    """
    return schedule_workflow_resume_sync(workflow_uid, use_async)


@celery_app.task(
    name="workflow.activity_handoff",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    queue="workflow_activity_handoff",
)
def _schedule_activity_handoff_celery(
    celery_task: CeleryTask,
    activity_uid: str,
    use_async: bool = True,
) -> str:
    """Celery 版本的三类调度任务

    注意：此任务必须使用单独的 worker 执行，并发数为 1，确保顺序执行
    启动命令：
        uv run celery -A ext.ext_celery.worker worker -l info -Q workflow_activity_handoff -c 1 -n activity_handoff_worker@%%h
    """
    return schedule_activity_handoff_sync(activity_uid, use_async)


# 导出同步版本的任务函数（用于直接调用）
__all__ = [
    "schedule_workflow_start",
    "schedule_workflow_resume",
    "schedule_activity_handoff",
]
