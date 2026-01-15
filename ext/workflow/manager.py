"""
工作流管理器

处理工作流的核心逻辑，包括工作流创建、状态管理、任务调度等
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from loguru import logger

from ext.ext_tortoise.enums import (
    ActivityStatusEnum,
    WorkflowConfigFormatEnum,
    WorkflowStatusEnum,
)
from ext.ext_tortoise.models.knowledge_base import Activity, Workflow
from util.graph import GraphUtil


class WorkflowManager:
    """工作流管理器"""

    @staticmethod
    async def create_workflow(
        config: Dict[str, Any],
        config_format: str = "dict",
        initial_inputs: Optional[Dict[str, Any]] = None,
    ) -> Workflow:
        """创建工作流和所有活动记录

        Args:
            config: 工作流配置（DAG 结构）
            config_format: 配置格式（yaml/json/dict）
            initial_inputs: 初始输入数据

        Returns:
            创建的 Workflow 实例
        """
        workflow_uid = uuid.uuid4()

        # 创建工作流记录
        workflow = await Workflow.create(
            uid=workflow_uid,
            config=config,
            config_format=config_format,
            status=WorkflowStatusEnum.pending.value,
        )

        # 解析 DAG 配置
        graph = GraphUtil(config=config, config_format=config_format)

        # 创建所有活动记录

        await WorkflowManager.create_activities_for_workflow(
            workflow_uid,
            config,
            config_format,
            initial_inputs
        )

        return workflow

    @staticmethod
    async def create_activities_for_workflow(
        workflow_uid: uuid.UUID,
        config: Dict[str, Any],
        config_format: str = "dict",
        initial_inputs: Optional[Dict[str, Any]] = None,
    ) -> List[Activity]:
        """为已存在的工作流创建所有活动记录

        Args:
            workflow_uid: 已存在的 workflow UID
            config: 工作流配置（DAG 结构）
            config_format: 配置格式（yaml/json/dict）
            initial_inputs: 初始输入数据

        Returns:
            创建的 Activity 列表
        """
        # 解析 DAG 配置
        graph = GraphUtil(config=config, config_format=config_format)

        # 创建所有活动记录
        activities = []
        for node_name, node_config in graph.config.items():
            node_info = graph.get_node_info(node_name)

            # 合并初始输入
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

        return activities

    @staticmethod
    async def get_workflow_by_uid(workflow_uid: str) -> Optional[Workflow]:
        """根据 UID 获取工作流

        Args:
            workflow_uid: 工作流 UID

        Returns:
            Workflow 实例，如果不存在则返回 None
        """
        return await Workflow.filter(uid=workflow_uid).first()

    @staticmethod
    async def get_activities_by_workflow(workflow_uid: uuid.UUID) -> List[Activity]:
        """获取工作流的所有活动

        如果活动不存在，会自动根据 workflow 的配置创建所有活动

        Args:
            workflow_uid: 工作流 UID

        Returns:
            活动列表
        """
        # 获取所有活动
        activities = await Activity.filter(workflow_uid=workflow_uid)

        # 如果活动不存在，则自动创建
        if not activities:
            # 获取 workflow 信息
            workflow = await WorkflowManager.get_workflow_by_uid(workflow_uid)
            if workflow:
                logger.info(f"No activities found for workflow {workflow_uid}, creating activities...")
                config = workflow.config
                config_format = workflow.config_format or "dict"

                # 创建所有活动
                await WorkflowManager.create_activities_for_workflow(
                    workflow_uid=workflow_uid,
                    config=config,
                    config_format=config_format,
                    initial_inputs=None,
                )

                # 重新获取活动列表
                activities = await Activity.filter(workflow_uid=workflow_uid)
                logger.info(f"Created {len(activities)} activities for workflow {workflow_uid}")

        return activities

    @staticmethod
    async def update_workflow_status(
        workflow_uid: str,
        status: WorkflowStatusEnum,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        canceled_at: Optional[datetime] = None,
    ) -> bool:
        """更新工作流状态

        Args:
            workflow_uid: 工作流 UID
            status: 新状态
            started_at: 开始时间
            completed_at: 完成时间
            canceled_at: 取消时间

        Returns:
            是否更新成功
        """
        update_data: Dict[str, Any] = {"status": status.value}

        if started_at:
            update_data["started_at"] = started_at
        if completed_at:
            update_data["completed_at"] = completed_at
        if canceled_at:
            update_data["canceled_at"] = canceled_at

        result = await Workflow.filter(uid=workflow_uid).update(**update_data)
        return result > 0

    @staticmethod
    async def update_activity_status(
        activity_uid: str,
        status: ActivityStatusEnum,
        output: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        stack_trace: Optional[str] = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        canceled_at: Optional[datetime] = None,
        celery_task_id: Optional[str] = None,
        increment_retry: bool = False,
    ) -> bool:
        """更新活动状态

        Args:
            activity_uid: 活动 UID
            status: 新状态
            output: 输出数据
            error_message: 错误信息
            stack_trace: 堆栈跟踪
            started_at: 开始时间
            completed_at: 完成时间
            canceled_at: 取消时间
            celery_task_id: Celery 任务 ID
            increment_retry: 是否增加重试次数

        Returns:
            是否更新成功
        """
        update_data: Dict[str, Any] = {"status": status.value}

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
            # 使用 F 表达式增加重试次数
            from tortoise.expressions import F
            update_data["retry_count"] = F("retry_count") + 1

        result = await Activity.filter(uid=activity_uid).update(**update_data)
        return result > 0

    @staticmethod
    async def get_ready_activities(
        workflow_uid: str, graph: GraphUtil
    ) -> List[Activity]:
        """获取准备执行的活动

        Args:
            workflow_uid: 工作流 UID
            graph: DAG 图实例

        Returns:
            准备执行的活动列表
        """
        # 获取所有活动
        all_activities = await WorkflowManager.get_activities_by_workflow(workflow_uid)

        # 构建已完成的活动名称集合
        completed_activities: Set[str] = {
            act.name
            for act in all_activities
            if act.status == ActivityStatusEnum.completed.value
        }

        # 获取待执行的活动
        pending_activities = [
            act
            for act in all_activities
            if act.status in [ActivityStatusEnum.pending, ActivityStatusEnum.failed, ActivityStatusEnum.canceled]
        ]

        # 检查哪些待执行的活动可以执行
        ready_activities: List[Activity] = []
        for activity in pending_activities:
            if graph.is_node_ready(activity.name, completed_activities):
                ready_activities.append(activity)
        
        

        return ready_activities

    @staticmethod
    async def check_workflow_completion(workflow_uid: str) -> bool:
        """检查工作流是否完成

        Args:
            workflow_uid: 工作流 UID

        Returns:
            是否完成
        """
        activities = await WorkflowManager.get_activities_by_workflow(workflow_uid)

        # 检查是否有失败的活动
        has_failed = any(
            act.status in [ActivityStatusEnum.failed.value, ActivityStatusEnum.canceled.value]
            for act in activities
        )

        # 检查是否所有活动都已完成（成功或失败）
        all_finished = all(
            act.status in [
                ActivityStatusEnum.completed.value,
                ActivityStatusEnum.failed.value,
                ActivityStatusEnum.canceled.value,
            ]
            for act in activities
        )

        if not all_finished:
            return False

        # 更新工作流状态
        if has_failed:
            await WorkflowManager.update_workflow_status(
                workflow_uid,
                WorkflowStatusEnum.failed,
                completed_at=datetime.now(),
            )
        else:
            await WorkflowManager.update_workflow_status(
                workflow_uid,
                WorkflowStatusEnum.completed,
                completed_at=datetime.now(),
            )

        return True

    @staticmethod
    async def mark_workflow_failed(workflow_uid: str, reason: str) -> None:
        """标记工作流为失败状态

        Args:
            workflow_uid: 工作流 UID
            reason: 失败原因
        """
        # 更新工作流状态
        await WorkflowManager.update_workflow_status(
            workflow_uid,
            WorkflowStatusEnum.failed,
            completed_at=datetime.now(),
        )

        # 取消所有待执行的活动
        await Activity.filter(
            workflow_uid=workflow_uid,
            status=ActivityStatusEnum.pending.value,
        ).update(
            status=ActivityStatusEnum.canceled.value,
            canceled_at=datetime.now(),
            error_message=f"Workflow failed: {reason}",
        )

    @staticmethod
    def build_graph(workflow: Workflow) -> GraphUtil:
        """根据工作流配置构建 DAG 图

        Args:
            workflow: Workflow 实例

        Returns:
            GraphUtil 实例
        """
        config_format = (
            workflow.config_format.value
            if isinstance(workflow.config_format, WorkflowConfigFormatEnum)
            else workflow.config_format
        )
        return GraphUtil(config=workflow.config, config_format=config_format)

    @staticmethod
    async def propagate_output_to_downstream(
        activity: Activity, graph: GraphUtil
    ) -> None:
        """将活动输出传播到下游活动的输入

        Args:
            activity: 完成的活动
            graph: DAG 图实例
        """
        node_info = graph.get_node_info(activity.name)
        output = activity.output

        # 获取所有下游活动
        downstream_activities = await Activity.filter(
            workflow_uid=activity.workflow_uid,
            name__in=node_info.children,
            status=ActivityStatusEnum.pending.value,
        )

        # 更新下游活动的输入
        for downstream in downstream_activities:
            # 合并输出到下游活动的输入
            downstream.input.update(output)
            await downstream.save()
