import asyncio
import uuid
from datetime import datetime
from typing import Literal
from weakref import WeakValueDictionary

from celery import Task as CeleryTask
from loguru import logger

from ext.ext_celery.app import celery_app
from ext.ext_tortoise.enums import (
    ActivityStatusEnum,
    WorkflowStatusEnum,
)
from ext.ext_tortoise.models.knowledge_base import Activity, Workflow
from ext.workflow.exceptions import (
    WorkflowNotFoundError,
    ActivityNotFoundError,
    TaskNotFoundError,
    WorkflowAlreadyCompletedError,
)
from ext.workflow.manager import WorkflowManager
from util.graph import GraphUtil


class WorkflowScheduler:
    """Core workflow scheduling logic"""

    _completion_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()

    @staticmethod
    async def schedule_workflow(
        workflow_uid: uuid.UUID | str | None = None,
        config: dict | None = None,
        config_format: str = "dict",
        initial_inputs: dict | None = None,
        execute_mode: str = "direct",
    ) -> str:
        """Schedule workflow execution (start new or resume existing)

        Args:
            workflow_uid: Existing workflow UID (for resume mode)
            config: Workflow config (for new workflow)
            config_format: Config format (yaml/json/dict)
            initial_inputs: Initial inputs
            execute_mode: "celery" (fire-and-forget) or "direct" (wait complete)

        Returns:
            Workflow UID

        Raises:
            ValueError: If neither workflow_uid nor config provided
            WorkflowNotFoundError: If workflow not found
        """
        if isinstance(workflow_uid, str):
            workflow_uid = uuid.UUID(workflow_uid)

        trace_id = workflow_uid

        if workflow_uid:
            workflow = await WorkflowManager.get_workflow_by_uid(workflow_uid)
        else:
            if not config:
                raise ValueError("Either workflow_uid or config must be provided")
            workflow = await WorkflowManager.create_workflow(config, config_format, initial_inputs)
            trace_id = workflow.uid

        with logger.contextualize(trace_id=str(trace_id)):
            logger.info(f"Scheduling workflow {workflow.uid}, mode={execute_mode}")

            workflow.status = WorkflowStatusEnum(workflow.status)

            if workflow.status in [
                WorkflowStatusEnum.completed,
                WorkflowStatusEnum.canceled,
            ]:
                logger.info(f"Workflow {workflow.uid} already {workflow.status.value}")
                return str(workflow.uid)

            graph = WorkflowManager.build_graph(workflow)

            if workflow.status in [
                WorkflowStatusEnum.pending,
                WorkflowStatusEnum.failed,
            ]:
                await WorkflowManager.update_workflow_status(
                    workflow.uid,
                    WorkflowStatusEnum.running,
                    started_at=datetime.now() if not workflow.started_at else None,
                )

            ready_activities = await WorkflowManager.get_ready_activities(workflow.uid, graph)
            logger.info(f"Ready activities: {[a.name for a in ready_activities]}")

            if execute_mode == "direct":
                await WorkflowScheduler._launch_activity_tasks_direct(ready_activities)

                # In direct mode, wait for workflow completion (no timeout)
                poll_interval = 0.5  # seconds

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
                await WorkflowScheduler._launch_activity_tasks_celery(ready_activities)

            return str(workflow.uid)

    @staticmethod
    async def schedule_activity_handoff(
        activity_uid: uuid.UUID | str,
        execute_mode: str = "direct",
    ) -> str:
        """Schedule activity handoff to downstream activities

        IMPORTANT: In Celery mode, this task uses queue="workflow_handoff" with concurrency=1
        to ensure serial execution and prevent race conditions.

        Args:
            activity_uid: Completed activity UID
            execute_mode: "celery" or "direct"

        Returns:
            Activity UID as string
        """
        if isinstance(activity_uid, str):
            activity_uid = uuid.UUID(activity_uid)

        activity = await WorkflowManager.get_activity_by_uid(activity_uid)

        with logger.contextualize(trace_id=str(activity.workflow_uid), activity_uid=str(activity.uid)):
            logger.info(f"Activity handoff: {activity.name}, status={activity.status}, mode={execute_mode}")

            workflow = await WorkflowManager.get_workflow_by_uid(activity.workflow_uid)

            if workflow.status in [
                WorkflowStatusEnum.completed.value,
                WorkflowStatusEnum.failed.value,
                WorkflowStatusEnum.canceled.value,
            ]:
                logger.info(f"Workflow {workflow.uid} is {workflow.status}, skipping")
                return str(activity.uid)

            if WorkflowManager.is_failed_status(ActivityStatusEnum(activity.status)):
                logger.warning(f"Activity {activity.name} failed, marking workflow failed")
                await WorkflowManager.mark_workflow_failed(
                    activity.workflow_uid,
                    f"Activity {activity.name} failed: {activity.error_message}",
                )
                return str(activity.uid)

            graph = WorkflowManager.build_graph(workflow)
            await WorkflowManager.propagate_output_to_downstream(activity, graph)

            ready_activities = await WorkflowManager.get_ready_activities(activity.workflow_uid, graph)
            logger.info(f"Ready downstream: {[a.name for a in ready_activities]}")

            if execute_mode == "direct":
                await WorkflowScheduler._launch_activity_tasks_direct(ready_activities)

            else:
                await WorkflowScheduler._launch_activity_tasks_celery(ready_activities)

            # Only check completion if no more ready activities
            if not ready_activities:
                workflow_uid_str = str(activity.workflow_uid)
                lock = WorkflowScheduler._completion_locks.get(workflow_uid_str)
                if lock is None:
                    lock = asyncio.Lock()
                    WorkflowScheduler._completion_locks[workflow_uid_str] = lock

                async with lock:
                    is_completed = await WorkflowManager.is_workflow_completed(activity.workflow_uid)
                    if is_completed:
                        logger.info(f"Workflow {workflow.uid} completed in direct mode")
            return str(activity.uid)


    @staticmethod
    async def _launch_activity_tasks_direct(activities: list[Activity]) -> None:
        """Launch activity tasks directly (concurrent execution)

        Args:
            activities: List of activities to execute
        """
        if not activities:
            return

        tasks = []
        for activity in activities:
            task_name = activity.execute_params.get("task_name")
            if not task_name:
                logger.warning(f"Activity {activity.name} has no task_name, skipping")
                continue

            task = celery_app.tasks.get(task_name)
            if not task:
                logger.error(f"Task not found: {task_name}")
                continue

            # Use task's async_call method for direct execution
            if hasattr(task, "async_call"):
                tasks.append(task.async_call(str(activity.uid)))
            else:
                logger.error(f"Task {task_name} does not support direct execution")

        if tasks:
            logger.info(f"Launching {len(tasks)} activities concurrently in direct mode")
            await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    async def _launch_activity_tasks_celery(activities: list[Activity]) -> None:
        """Launch activity tasks via Celery apply_async (fire-and-forget)

        Args:
            activities: List of activities to execute
        """
        if not activities:
            return

        for activity in activities:
            task_name = activity.execute_params.get("task_name")
            if not task_name:
                logger.warning(f"Activity {activity.name} has no task_name, skipping")
                continue

            task = celery_app.tasks.get(task_name)
            if not task:
                logger.error(f"Task not found: {task_name}")
                continue

            celery_task_result = task.apply_async(args=[str(activity.uid), "celery"], countdown=0)
            await WorkflowManager.update_activity_status(
                activity.uid,
                ActivityStatusEnum.pending,
                celery_task_id=celery_task_result.id,
            )
            logger.info(f"Launched task {task_name} for {activity.name}, task_id={celery_task_result.id}")


# Celery task wrappers for external calls
@celery_app.task(name="workflow.schedule_workflow_entry", bind=True, queue="workflow_entry")
def schedule_workflow_celery_entry(
    celery_task: CeleryTask,
    workflow_uid: str | None = None,
    config: dict | None = None,
    config_format: str = "dict",
    initial_inputs: dict | None = None,
) -> str:
    """Celery entry point for workflow scheduling"""
    import asyncio

    async def _schedule():
        return await WorkflowScheduler.schedule_workflow(
            workflow_uid=workflow_uid,
            config=config,
            config_format=config_format,
            initial_inputs=initial_inputs,
            execute_mode="celery",
        )

    # Detect and reuse existing event loop (Celery worker environment)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Use run_coroutine_threadsafe with existing loop
            future = asyncio.run_coroutine_threadsafe(_schedule(), loop)
            return str(future.result())
        else:
            # Loop exists but not running
            return str(loop.run_until_complete(_schedule()))
    except RuntimeError:
        # No event loop exists, create new one (test environment)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return str(loop.run_until_complete(_schedule()))


@celery_app.task(name="workflow.activity_handoff_entry", bind=True, queue="workflow_handoff")
def schedule_activity_handoff_celery_entry(celery_task: CeleryTask, activity_uid: str) -> str:
    """Celery entry point for activity handoff

    IMPORTANT: This task MUST be executed with concurrency=1 to ensure
    only one handoff executes at a time. This prevents race conditions
    when multiple activities complete concurrently.

    Worker startup command:
        uv run celery -A ext.ext_celery.worker worker -Q workflow_handoff -c 1
    """
    import asyncio

    async def _handoff():
        return await WorkflowScheduler.schedule_activity_handoff(activity_uid=activity_uid, execute_mode="celery")

    # Detect and reuse existing event loop (Celery worker environment)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Use run_coroutine_threadsafe with existing loop
            future = asyncio.run_coroutine_threadsafe(_handoff(), loop)
            return str(future.result())
        else:
            # Loop exists but not running
            return str(loop.run_until_complete(_handoff()))
    except RuntimeError:
        # No event loop exists, create new one (test environment)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return str(loop.run_until_complete(_handoff()))
