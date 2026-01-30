import uuid
import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from loguru import logger

from ext.ext_celery.app import celery_app
from ext.ext_tortoise.enums import ActivityStatusEnum
from ext.workflow.exceptions import ActivityNotFoundError
from ext.workflow.manager import WorkflowManager
from ext.workflow.scheduler import WorkflowScheduler, schedule_activity_handoff_celery_entry
from ext.ext_tortoise.models.knowledge_base import Activity, Workflow
from util.graph import GraphUtil


class ActivityTaskTemplate(ABC):
    """Base class for activity tasks

    Users must implement the execute() method.
    All lifecycle management (state transitions, handoffs) is handled automatically.
    """

    def __init__(self, activity_uid: uuid.UUID | str, execute_mode: str = "direct"):
        """Initialize task

        Args:
            activity_uid: Activity UID (string or UUID)
            execute_mode: Execution mode (direct/celery)
        """
        if isinstance(activity_uid, str):
            activity_uid = uuid.UUID(activity_uid)

        self.activity_uid = activity_uid
        self.execute_mode = execute_mode
        self._activity: Activity | None = None
        self._workflow: Workflow | None = None
        self._graph: GraphUtil | None = None

    @abstractmethod
    async def execute(self) -> dict[str, Any]:
        """Execute business logic (must be implemented by user)

        Returns:
            Output data dictionary
        """

    async def _execute_lifecycle(self) -> dict[str, Any]:
        """Full execution lifecycle

        Automatically handles:
        1. Load activity and workflow
        2. Set status to running
        3. Execute user logic
        4. Save output and set status to completed
        5. Trigger activity handoff

        Returns:
            Output data
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

    async def _load_activity(self) -> None:
        """Load activity and workflow records"""
        self._activity = await WorkflowManager.get_activity_by_uid(self.activity_uid)
        self._workflow = await WorkflowManager.get_workflow_by_uid(self._activity.workflow_uid)
        self._graph = WorkflowManager.build_graph(self._workflow)

    async def _set_running(self) -> None:
        """Set activity status to running"""
        await WorkflowManager.update_activity_status(
            self.activity_uid,
            ActivityStatusEnum.running,
            started_at=datetime.now(),
        )

    async def _set_completed(self, output: dict[str, Any]) -> None:
        """Set activity status to completed and trigger handoff

        Args:
            output: Output data
        """
        await WorkflowManager.update_activity_status(
            self.activity_uid,
            ActivityStatusEnum.completed,
            output=output,
            completed_at=datetime.now(),
        )

        if self.execute_mode == "celery":
            schedule_activity_handoff_celery_entry.apply_async(args=[str(self.activity_uid)]) # type: ignore
        else:
            await WorkflowScheduler.schedule_activity_handoff(self.activity_uid, execute_mode=self.execute_mode)

    async def _handle_exception(self, exception: Exception) -> None:
        """Handle exception and set appropriate status

        Args:
            exception: Exception object
        """
        error_message = str(exception)
        stack_trace = traceback.format_exc()

        logger.error(f"Activity failed: {error_message}\n{stack_trace}")

        activity = await WorkflowManager.get_activity_by_uid(self.activity_uid)

        max_retries = activity.execute_params.get("max_retries", 3)
        current_retry = activity.retry_count

        if current_retry < max_retries:
            await WorkflowManager.update_activity_status(
                self.activity_uid,
                ActivityStatusEnum.retrying,
                error_message=error_message,
                stack_trace=stack_trace,
                increment_retry=True,
            )
        else:
            await WorkflowManager.update_activity_status(
                self.activity_uid,
                ActivityStatusEnum.failed,
                error_message=error_message,
                stack_trace=stack_trace,
                completed_at=datetime.now(),
            )

        if self.execute_mode == "celery":
            schedule_activity_handoff_celery_entry.apply_async(args=[str(self.activity_uid)]) # type: ignore
        else:
            await WorkflowScheduler.schedule_activity_handoff(self.activity_uid, execute_mode=self.execute_mode)

    @property
    def input(self) -> dict[str, Any]:
        """Get activity input"""
        if not self._activity:
            raise ActivityNotFoundError("Activity not loaded")
        return self._activity.input or {}

    @property
    def activity(self) -> Activity:
        """Get activity object"""
        if not self._activity:
            raise ActivityNotFoundError("Activity not loaded")
        return self._activity

    @property
    def workflow(self) -> Workflow:
        """Get workflow object"""
        if not self._workflow:
            raise ActivityNotFoundError("Workflow not loaded")
        return self._workflow

    @property
    def graph(self) -> GraphUtil:
        """Get DAG graph"""
        if not self._graph:
            raise ActivityNotFoundError("Graph not loaded")
        return self._graph

    @property
    def activity_name(self) -> str:
        """Get activity name"""
        if not self._activity:
            raise ActivityNotFoundError("Activity not loaded")
        return self._activity.name

    async def get_upstream_outputs(self) -> dict[str, Any]:
        """Get outputs from all upstream activities

        Returns:
            Dict mapping upstream activity names to their outputs
        """
        if not self._activity or not self._graph:
            raise ActivityNotFoundError("Activity or graph not loaded")

        node_info = self._graph.get_node_info(self.activity_name)

        outputs: dict[str, Any] = {}
        for parent_name in node_info.parents:
            parent_activity = await Activity.filter(
                workflow_uid=self._activity.workflow_uid,
                name=parent_name,
                status=ActivityStatusEnum.completed.value,
            ).first()

            if parent_activity:
                outputs[parent_name] = parent_activity.output or {}

        return outputs

    async def get_workflow_context(self) -> dict[str, Any]:
        """Get workflow context

        Returns:
            Workflow context dict
        """
        if not self._workflow:
            raise ActivityNotFoundError("Workflow not loaded")

        return {
            "workflow_uid": str(self._workflow.uid),
            "config_format": self._workflow.config_format.value,
            "status": self._workflow.status.value,
            "started_at": self._workflow.started_at,
        }


def activity_task(task_class: type[ActivityTaskTemplate]):
    """Decorator to create Celery task from ActivityTaskTemplate

    Usage:
        @activity_task
        class MyTask(ActivityTaskTemplate):
            async def execute(self) -> dict[str, Any]:
                return {"result": "success"}

    The resulting task can be called in two ways:
    1. Via Celery: task.apply_async(args=[activity_uid_str, "celery"])
    2. Direct: await task.async_call(activity_uid_str, "direct")

    Args:
        task_class: ActivityTaskTemplate subclass

    Returns:
        Task callable
    """
    task_name = f"workflow_activity.{task_class.__name__}"

    @celery_app.task(
        name=task_name,
        bind=True,
        max_retries=3,
    )
    def _celery_task_wrapper(celery_task, activity_uid: str, execute_mode: str = "direct") -> dict[str, Any]:
        """Celery task wrapper (sync entry point)"""
        import asyncio

        async def _execute():
            instance = task_class(activity_uid, execute_mode=execute_mode)
            return await instance._execute_lifecycle()

        # Detect and reuse existing event loop (Celery worker environment)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Use run_coroutine_threadsafe with existing loop
                future = asyncio.run_coroutine_threadsafe(_execute(), loop)
                return future.result()
            else:
                # Loop exists but not running
                return loop.run_until_complete(_execute())
        except RuntimeError:
            # No event loop exists, create new one (test environment)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(_execute())

    async def _async_wrapper(activity_uid: str, execute_mode: str = "direct") -> dict[str, Any]:
        """Async wrapper for direct calling"""
        instance = task_class(activity_uid, execute_mode=execute_mode)
        return await instance._execute_lifecycle()

    _celery_task_wrapper.async_call = _async_wrapper # type: ignore

    return _celery_task_wrapper
