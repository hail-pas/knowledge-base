"""
通用任务模板

提供 Activity 任务的模板，用户只需实现业务逻辑，无需关心执行前后的处理

支持两种调用方式：
1. 直接调用（同步方式）
2. Celery apply_async 调用（异步方式）
"""
import asyncio
import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional

from celery import Task as CeleryTask
from celery.exceptions import Retry

from ext.ext_celery.app import get_celery_app
from ext.workflow.tasks import schedule_activity_handoff_sync
from ext.ext_tortoise.enums import ActivityStatusEnum
from ext.ext_tortoise.models.knowledge_base import Activity, Workflow
from util.graph import GraphUtil
from loguru import logger


def run_async(coro):
    """
    运行异步协程，自动检测环境

    如果已有事件循环在运行，在其中运行协程
    否则创建新的事件循环运行
    """
    try:
        # 尝试获取当前运行的事件循环
        loop = asyncio.get_running_loop()
        # 如果已有循环在运行，创建任务并等待
        if loop.is_running():
            # 创建一个新任务
            task = asyncio.create_task(coro)
            return task
        else:
            # 循环存在但未运行，直接运行
            return loop.run_until_complete(coro)
    except RuntimeError:
        # 没有运行的事件循环，创建一个新的
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


class ActivityTaskTemplate(ABC):
    """Activity 任务模板

    用户需要继承此类并实现 execute 方法

    核心思路：
    1. 任务执行时肯定存在一条 activity 记录
    2. 能够从该 activity 记录获取所需的全部输入，输入可以为空
    3. 需要更新 activity 的状态为运行中
    4. 执行自定义业务逻辑，可以获取 workflow 相关上下文和输入数据
    5. 如果自定义业务逻辑正常执行完毕，保存自定义业务逻辑的返回 dict 输出保存到 activity 表，更新状态
    6. 然后启动一个三类调度任务
    7. 如果业务逻辑执行异常，则保存相关异常信息到 activity 表，并更新状态（如果重试次数小于配置的次数则继续重试，状态为重试中，否则设置状态为失败）
    8. 无论成功还是失败都需要启动一个三类调度任务
    """

    def __init__(self, celery_task: Optional[CeleryTask], activity_uid: str):
        """初始化任务模板

        Args:
            celery_task: Celery 任务实例（可能为 None，直接调用时）
            activity_uid: Activity UID
        """
        self.celery_task = celery_task
        self.activity_uid = activity_uid
        self.activity: Optional[Activity] = None
        self.workflow: Optional[Workflow] = None
        self.graph: Optional[GraphUtil] = None

    async def _call_async(self) -> Dict[str, Any]:
        """异步执行任务的完整流程"""
        try:
            # 1. 加载 Activity 记录
            await self._load_activity()

            # 2. 更新状态为运行中
            await self._update_status_to_running()

            # 3. 执行用户自定义的业务逻辑
            output = await self.execute()

            # 4. 保存输出并更新状态为完成
            await self._update_status_to_completed(output)

            return output

        except Retry:
            # Celery 重试异常，不更新状态，直接抛出
            raise
        except Exception as e:
            # 5. 处理异常，更新状态
            await self._handle_exception(e)
            raise

    async def _load_activity(self) -> None:
        """加载 Activity 和 Workflow 记录"""
        logger.info(f"ActivityTemplate: {self.__dict__}")
        self.activity = await Activity.filter(uid=self.activity_uid).first()
        if not self.activity:
            raise ValueError(f"Activity not found: {self.activity_uid}")

        # 加载 Workflow
        self.workflow = await Workflow.filter(uid=self.activity.workflow_uid).first()
        if not self.workflow:
            raise ValueError(f"Workflow not found: {self.activity.workflow_uid}")

        # 构建 DAG 图
        self.graph = GraphUtil(
            config=self.workflow.config,
            config_format=self.workflow.config_format.value,
        )

    async def _update_status_to_running(self) -> None:
        """更新状态为运行中"""
        if not self.activity:
            raise ValueError("Activity not loaded")

        await Activity.filter(uid=self.activity_uid).update(
            status=ActivityStatusEnum.running.value,
            started_at=datetime.now(),
        )

    async def _update_status_to_completed(self, output: Dict[str, Any]) -> None:
        """更新状态为完成

        Args:
            output: 输出数据
        """
        if not self.activity:
            raise ValueError("Activity not loaded")

        await Activity.filter(uid=self.activity_uid).update(
            status=ActivityStatusEnum.completed.value,
            output=output,
            completed_at=datetime.now(),
        )

        # 启动三类调度任务
        await self._trigger_activity_handoff()

    async def _handle_exception(self, exception: Exception) -> None:
        """处理异常

        Args:
            exception: 异常对象
        """
        if not self.activity:
            raise ValueError("Activity not loaded")

        error_message = str(exception)
        stack_trace = traceback.format_exc()

        # 获取重试配置
        max_retries = self.activity.execute_params.get("max_retries", 3)
        current_retry = self.activity.retry_count

        if current_retry < max_retries:
            # 重试
            await Activity.filter(uid=self.activity_uid).update(
                status=ActivityStatusEnum.retrying.value,
                error_message=error_message,
                stack_trace=stack_trace,
                retry_count=current_retry + 1,
            )
        else:
            # 失败
            await Activity.filter(uid=self.activity_uid).update(
                status=ActivityStatusEnum.failed.value,
                error_message=error_message,
                stack_trace=stack_trace,
                completed_at=datetime.now(),
            )

        # 无论成功还是失败都启动三类调度任务
        await self._trigger_activity_handoff()

    async def _trigger_activity_handoff(self) -> None:
        """触发三类调度任务（activity handoff）"""
        if not self.activity:
            raise ValueError("Activity not loaded")

        # 使用 celery_app 获取 Celery 任务并使用 apply_async 启动
        celery_app = get_celery_app()
        handoff_task = celery_app.tasks.get("workflow.activity_handoff")
        if handoff_task:
            handoff_task.apply_async(args=[str(self.activity.uid)], countdown=0)

    @abstractmethod
    async def execute(self) -> Dict[str, Any]:
        """执行自定义业务逻辑（用户需要实现）

        Returns:
            执行结果（字典格式）
        """
        pass

    # 便利方法：获取上下文数据

    @property
    def input(self) -> Dict[str, Any]:
        """获取 Activity 输入"""
        if not self.activity:
            raise ValueError("Activity not loaded")
        return self.activity.input or {}

    @property
    def workflow_config(self) -> Dict[str, Any]:
        """获取 Workflow 配置"""
        if not self.workflow:
            raise ValueError("Workflow not loaded")
        return self.workflow.config or {}

    @property
    def activity_name(self) -> str:
        """获取 Activity 名称"""
        if not self.activity:
            raise ValueError("Activity not loaded")
        return self.activity.name

    @property
    def workflow_uid(self) -> str:
        """获取 Workflow UID"""
        if not self.activity:
            raise ValueError("Activity not loaded")
        return str(self.activity.workflow_uid)

    async def get_upstream_outputs(self) -> Dict[str, Any]:
        """获取所有上游 Activity 的输出

        Returns:
            上游 Activity 输出字典，key 为 Activity 名称
        """
        if not self.activity or not self.graph:
            raise ValueError("Activity or graph not loaded")

        node_info = self.graph.get_node_info(self.activity_name)

        outputs: Dict[str, Any] = {}
        for parent_name in node_info.parents:
            parent_activity = await Activity.filter(
                workflow_uid=self.activity.workflow_uid,
                name=parent_name,
                status=ActivityStatusEnum.completed.value,
            ).first()

            if parent_activity:
                outputs[parent_name] = parent_activity.output or {}

        return outputs

    async def get_workflow_context(self) -> Dict[str, Any]:
        """获取 Workflow 上下文

        Returns:
            Workflow 上下文字典
        """
        if not self.workflow:
            raise ValueError("Workflow not loaded")

        return {
            "workflow_uid": str(self.workflow.uid),
            "config_format": self.workflow.config_format.value,
            "status": self.workflow.status.value,
            "started_at": self.workflow.started_at,
        }


def activity_task(template_class: type):
    """装饰器：将 ActivityTaskTemplate 子类转换为 Celery 任务

    使用方法:
        @activity_task
        class MyTask(ActivityTaskTemplate):
            async def execute(self) -> Dict[str, Any]:
                # 实现业务逻辑
                return {"result": "success"}

    Args:
        template_class: ActivityTaskTemplate 子类

    Returns:
        Celery 任务函数
    """
    celery_app = get_celery_app()
    task_name = f"workflow_activity.{template_class.__name__}"

    # 定义 Celery 任务包装器
    @celery_app.task(
        name=task_name,
        bind=True,
        max_retries=3,
    )
    def _celery_task_wrapper(celery_task: CeleryTask, activity_uid: str) -> Dict[str, Any]:
        """Celery 任务包装器（同步）"""
        # 创建任务模板实例
        task_instance = template_class(celery_task, activity_uid)

        # 使用 run_async 执行异步任务
        # 在 Celery worker 环境中，不会有已有的事件循环
        return run_async(task_instance._call_async())

    # 定义可以直接调用的同步函数
    def _sync_wrapper(activity_uid: str) -> Dict[str, Any]:
        """同步包装器，用于直接调用"""
        # 创建任务模板实例（celery_task=None）
        task_instance = template_class(None, activity_uid)

        # 使用 run_async 执行异步任务
        return run_async(task_instance._call_async())

    # 给任务对象添加一些属性，使其看起来像一个 Celery 任务
    _sync_wrapper.name = task_name
    _sync_wrapper.apply_async = _celery_task_wrapper.apply_async

    # 返回同步包装器，支持直接调用和 apply_async
    return _sync_wrapper


__all__ = [
    "ActivityTaskTemplate",
    "activity_task",
]
