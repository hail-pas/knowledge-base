"""
from ext.workflow import schedule_workflow, ActivityTaskTemplate, activity_task

# Define a custom task
@activity_task
class MyTask(ActivityTaskTemplate):
    async def execute(self) -> dict[str, Any]:
        return {"result": "success"}

# Schedule workflow
workflow_uid = await schedule_workflow(
    config=workflow_config,
    execute_mode="direct",  # or "celery"
)
"""

from typing import Literal

# Core scheduling functions
from ext.workflow.scheduler import (
    WorkflowScheduler,
)

# Task template
from ext.workflow.template import ActivityTaskTemplate, activity_task

# Manager
from ext.workflow.manager import WorkflowManager


from ext.workflow.scheduler import schedule_activity_handoff_celery_entry, schedule_workflow_celery_entry


async def schedule_workflow(
    workflow_uid: str | None = None,
    config: dict | None = None,
    config_format: str = "dict",
    initial_inputs: dict | None = None,
    execute_mode: Literal["celery", "direct"] = "direct",
) -> str:
    """Schedule workflow execution

    Start new workflow or resume existing one.

    Args:
        workflow_uid: Existing workflow UID (resume mode)
        config: Workflow DAG config (new workflow mode)
        config_format: Config format (yaml/json/dict)
        initial_inputs: Initial inputs for activities
        execute_mode: Execution mode
            - celery: Fire-and-forget via Celery apply_async
            - direct: Execute concurrently and wait for completion

    Returns:
        Workflow UID as string

    Examples:
        # Start new workflow
        workflow_uid = await schedule_workflow(
            config=my_config,
            execute_mode="direct"
        )

        # Resume existing workflow
        workflow_uid = await schedule_workflow(
            workflow_uid="existing-uid",
            execute_mode="celery"
        )
    """
    import uuid

    result_uid = await WorkflowScheduler.schedule_workflow(
        workflow_uid=workflow_uid,
        config=config,
        config_format=config_format,
        initial_inputs=initial_inputs,
        execute_mode=execute_mode,
    )
    return str(result_uid)


async def schedule_activity_handoff(
    activity_uid: str,
    execute_mode: Literal["celery", "direct"] = "direct",
) -> str:
    """Schedule activity handoff to downstream activities

    Triggered after an activity completes.

    Args:
        activity_uid: Completed activity UID
        execute_mode: Execution mode (celery/direct)

    Returns:
        Activity UID
    """
    result_uid = await WorkflowScheduler.schedule_activity_handoff(activity_uid=activity_uid, execute_mode=execute_mode)
    return str(result_uid)


__all__ = [
    # Core functions
    "schedule_workflow",
    "schedule_activity_handoff",
    # Core classes
    "WorkflowScheduler",
    "WorkflowManager",
    "ActivityTaskTemplate",
    "activity_task",
    # Celery entry points
    "schedule_workflow_celery_entry",
    "schedule_activity_handoff_celery_entry",
]
