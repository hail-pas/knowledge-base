import uuid
from datetime import datetime
from typing import Any

from loguru import logger

from ext.ext_tortoise.enums import (
    ActivityStatusEnum,
    WorkflowStatusEnum,
)
from ext.ext_tortoise.models.knowledge_base import Activity, Workflow
from ext.workflow.exceptions import (
    WorkflowNotFoundError,
    ActivityNotFoundError,
)
from util.graph import GraphUtil


class WorkflowManager:
    """Workflow and activity state management"""

    @staticmethod
    async def create_workflow(
        config: dict[str, Any],
        config_format: str = "dict",
        initial_inputs: dict[str, Any] | None = None,
    ) -> Workflow:
        """Create workflow and all activity records

        Args:
            config: Workflow DAG configuration
            config_format: Configuration format (yaml/json/dict)
            initial_inputs: Initial input data for activities

        Returns:
            Created Workflow instance
        """
        workflow_uid = uuid.uuid4()

        with logger.contextualize(trace_id=str(workflow_uid)):
            workflow = await Workflow.create(
                uid=workflow_uid,
                config=config,
                config_format=config_format,  # type: ignore
                status=WorkflowStatusEnum.pending.value,
            )

            await WorkflowManager.create_activities(
                workflow_uid=workflow_uid,
                config=config,
                config_format=config_format,  # type: ignore
                initial_inputs=initial_inputs,
            )

            logger.info(f"Created workflow {workflow_uid} with {len(config)} activities")
            return workflow

    @staticmethod
    async def create_activities(
        workflow_uid: uuid.UUID,
        config: dict[str, Any],
        config_format: str = "dict",
        initial_inputs: dict[str, Any] | None = None,
    ) -> list[Activity]:
        """Create activity records for existing workflow

        Args:
            workflow_uid: Workflow UID
            config: Workflow DAG configuration
            config_format: Configuration format
            initial_inputs: Initial input data

        Returns:
            Created Activity list
        """
        with logger.contextualize(trace_id=str(workflow_uid)):
            graph = GraphUtil(config=config, config_format=config_format)  # type: ignore

            activities = []
            for node_name, node_config in graph.config.items():
                node_info = graph.get_node_info(node_name)

                activity_input = {}
                if initial_inputs:
                    activity_input.update(initial_inputs.get(node_name, {}))
                activity_input.update(node_config.input)

                activity = await Activity.create(
                    workflow_uid=workflow_uid,
                    uid=uuid.uuid4(),
                    name=node_name,
                    input=activity_input,
                    output={},
                    retry_count=0,
                    execute_params=node_config.execute_params,
                    status=ActivityStatusEnum.pending.value,
                )
                activities.append(activity)

            logger.debug(f"Created {len(activities)} activities for workflow {workflow_uid}")
            return activities

    @staticmethod
    async def get_workflow_by_uid(workflow_uid: uuid.UUID) -> Workflow:
        """Get workflow by UID

        Args:
            workflow_uid: Workflow UID

        Returns:
            Workflow instance

        Raises:
            WorkflowNotFoundError: If workflow not found
        """
        with logger.contextualize(trace_id=str(workflow_uid)):
            workflow = await Workflow.filter(uid=workflow_uid).first()
            if not workflow:
                raise WorkflowNotFoundError(f"Workflow not found: {workflow_uid}")
            return workflow

    @staticmethod
    async def get_activities_by_workflow(workflow_uid: uuid.UUID) -> list[Activity]:
        """Get all activities for workflow

        Auto-creates activities if they don't exist.

        Args:
            workflow_uid: Workflow UID

        Returns:
            Activity list
        """
        with logger.contextualize(trace_id=str(workflow_uid)):
            activities = await Activity.filter(workflow_uid=workflow_uid)

            if not activities:
                workflow = await WorkflowManager.get_workflow_by_uid(workflow_uid)
                logger.info(f"No activities found for {workflow_uid}, creating from config")

                await WorkflowManager.create_activities(
                    workflow_uid=workflow_uid,
                    config=workflow.config,
                    config_format=workflow.config_format.value
                    if isinstance(workflow.config_format, WorkflowStatusEnum)
                    else workflow.config_format,  # type: ignore
                    initial_inputs=None,
                )

                activities = await Activity.filter(workflow_uid=workflow_uid)
                logger.info(f"Created {len(activities)} activities for {workflow_uid}")

            return activities

    @staticmethod
    async def get_activity_by_uid(activity_uid: uuid.UUID) -> Activity:
        """Get activity by UID

        Args:
            activity_uid: Activity UID

        Returns:
            Activity instance

        Raises:
            ActivityNotFoundError: If activity not found
        """
        activity = await Activity.filter(uid=activity_uid).first()
        if not activity:
            with logger.contextualize(activity_uid=str(activity_uid)):
                raise ActivityNotFoundError(f"Activity not found: {activity_uid}")

        with logger.contextualize(trace_id=str(activity.workflow_uid), activity_uid=str(activity_uid)):
            return activity

    @staticmethod
    async def update_workflow_status(
        workflow_uid: uuid.UUID,
        status: WorkflowStatusEnum,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        canceled_at: datetime | None = None,
    ) -> bool:
        """Update workflow status

        Args:
            workflow_uid: Workflow UID
            status: New status
            started_at: Start time
            completed_at: Completion time
            canceled_at: Cancel time

        Returns:
            True if updated successfully
        """
        with logger.contextualize(trace_id=str(workflow_uid)):
            update_data: dict[str, Any] = {"status": status.value}

            if started_at:
                update_data["started_at"] = started_at
            if completed_at:
                update_data["completed_at"] = completed_at
            if canceled_at:
                update_data["canceled_at"] = canceled_at

            result = await Workflow.filter(uid=workflow_uid).update(**update_data)
            logger.debug(f"Updated workflow {workflow_uid} status to {status.value}")
            return result > 0

    @staticmethod
    async def update_activity_status(
        activity_uid: uuid.UUID,
        status: ActivityStatusEnum,
        output: dict[str, Any] | None = None,
        error_message: str | None = None,
        stack_trace: str | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        canceled_at: datetime | None = None,
        celery_task_id: str | None = None,
        increment_retry: bool = False,
    ) -> bool:
        """Update activity status

        Args:
            activity_uid: Activity UID
            status: New status
            output: Output data
            error_message: Error message
            stack_trace: Stack trace
            started_at: Start time
            completed_at: Completion time
            canceled_at: Cancel time
            celery_task_id: Celery task ID
            increment_retry: Increment retry count

        Returns:
            True if updated successfully
        """
        with logger.contextualize(activity_uid=str(activity_uid)):
            update_data: dict[str, Any] = {"status": status.value}

            if output is not None:
                update_data["output"] = output
            if error_message is not None:
                update_data["error_message"] = error_message
            if stack_trace is not None:
                update_data["stack_trace"] = stack_trace
            if started_at:
                update_data["started_at"] = started_at
            if completed_at:
                update_data["completed_at"] = completed_at
            if canceled_at:
                update_data["canceled_at"] = canceled_at
            if celery_task_id:
                update_data["celery_task_id"] = celery_task_id
            if increment_retry:
                from tortoise.expressions import F

                update_data["retry_count"] = F("retry_count") + 1

            result = await Activity.filter(uid=activity_uid).update(**update_data)
            logger.debug(f"Updated activity {activity_uid} status to {status.value}")
            return result > 0

    @staticmethod
    async def get_ready_activities(
        workflow_uid: uuid.UUID,
        graph: GraphUtil,
    ) -> list[Activity]:
        """Get activities ready for execution

        Args:
            workflow_uid: Workflow UID
            graph: DAG graph

        Returns:
            Ready activities list
        """
        with logger.contextualize(trace_id=str(workflow_uid)):
            all_activities = await WorkflowManager.get_activities_by_workflow(workflow_uid)

            completed_activities: set[str] = {
                act.name for act in all_activities if act.status == ActivityStatusEnum.completed.value
            }

            pending_activities = [
                act
                for act in all_activities
                if act.status
                in [
                    ActivityStatusEnum.pending.value,
                    ActivityStatusEnum.failed.value,
                    ActivityStatusEnum.canceled.value,
                ]
            ]

            ready_activities: list[Activity] = []
            for activity in pending_activities:
                if graph.is_node_ready(activity.name, completed_activities):
                    ready_activities.append(activity)

            logger.debug(f"Found {len(ready_activities)} ready activities for {workflow_uid}")
            return ready_activities

    @staticmethod
    async def is_workflow_completed(workflow_uid: uuid.UUID) -> bool:
        """Check if workflow is completed and update status

        Args:
            workflow_uid: Workflow UID

        Returns:
            True if workflow is completed
        """
        with logger.contextualize(trace_id=str(workflow_uid)):
            activities = await WorkflowManager.get_activities_by_workflow(workflow_uid)

            has_failed = any(WorkflowManager.is_failed_status(ActivityStatusEnum(act.status)) for act in activities)

            all_finished = all(WorkflowManager.is_terminal_status(ActivityStatusEnum(act.status)) for act in activities)

            if not all_finished:
                return False

            # Check if workflow is already in terminal state
            workflow = await WorkflowManager.get_workflow_by_uid(workflow_uid)
            current_status = WorkflowStatusEnum(workflow.status)

            if current_status in [
                WorkflowStatusEnum.completed,
                WorkflowStatusEnum.failed,
                WorkflowStatusEnum.canceled,
            ]:
                return True

            # Update workflow status
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

    @staticmethod
    async def mark_workflow_failed(workflow_uid: uuid.UUID, reason: str) -> None:
        """Mark workflow as failed and cancel pending activities

        Args:
            workflow_uid: Workflow UID
            reason: Failure reason
        """
        with logger.contextualize(trace_id=str(workflow_uid)):
            await WorkflowManager.update_workflow_status(
                workflow_uid,
                WorkflowStatusEnum.failed,
                completed_at=datetime.now(),
            )

            await Activity.filter(
                workflow_uid=workflow_uid,
                status=ActivityStatusEnum.pending.value,
            ).update(
                status=ActivityStatusEnum.canceled.value,
                canceled_at=datetime.now(),
                error_message=f"Workflow failed: {reason}",
            )

            logger.warning(f"Workflow {workflow_uid} failed: {reason}")

    @staticmethod
    def build_graph(workflow: Workflow) -> GraphUtil:
        """Build DAG graph from workflow config

        Args:
            workflow: Workflow instance

        Returns:
            GraphUtil instance
        """
        config_format = (
            workflow.config_format.value
            if isinstance(workflow.config_format, WorkflowStatusEnum)
            else workflow.config_format
        )
        return GraphUtil(config=workflow.config, config_format=config_format)  # type: ignore

    @staticmethod
    async def propagate_output_to_downstream(
        activity: Activity,
        graph: GraphUtil,
    ) -> None:
        """Propagate activity output to downstream activities

        Args:
            activity: Completed activity
            graph: DAG graph
        """
        with logger.contextualize(trace_id=str(activity.workflow_uid), activity_uid=str(activity.uid)):
            node_info = graph.get_node_info(activity.name)
            output = activity.output

            downstream_activities = await Activity.filter(
                workflow_uid=activity.workflow_uid,
                name__in=node_info.children,
                status=ActivityStatusEnum.pending.value,
            )

            for downstream in downstream_activities:
                downstream.input.update(output)
                await downstream.save()

            logger.debug(
                f"Propagated output from {activity.name} to {len(downstream_activities)} downstream activities"
            )

    @staticmethod
    def is_terminal_status(status: ActivityStatusEnum) -> bool:
        """Check if status is terminal (cannot transition further)

        Args:
            status: Activity status

        Returns:
            True if terminal
        """
        return status in [
            ActivityStatusEnum.completed,
            ActivityStatusEnum.failed,
            ActivityStatusEnum.canceled,
        ]

    @staticmethod
    def is_failed_status(status: ActivityStatusEnum) -> bool:
        """Check if status indicates failure

        Args:
            status: Activity status

        Returns:
            True if failed
        """
        return status in [
            ActivityStatusEnum.failed,
            ActivityStatusEnum.canceled,
        ]
