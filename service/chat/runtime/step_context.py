"""
Step Context - 步骤上下文管理器

管理单个步骤的生命周期，包括：
- 自动发送 on_step_start 和 on_step_complete/on_step_failed 事件
- 自动捕获异常
- 自动创建 Artifact
- 支持流式更新
"""

import asyncio
from typing import Optional, Dict, Any, List
from uuid import uuid4
from datetime import datetime
import traceback

from service.chat.enums import StepTypeEnum, ArtifactTypeEnum, StepStatusEnum, ErrorCodeEnum


class StepContext:
    """
    步骤上下文管理器

    管理单个步骤的生命周期，自动处理事件发送和产物创建。

    使用方式：
        async with trace.step_context(
            step_type=StepTypeEnum.retrieval,
            step_name="知识库检索",
            input={"query": "什么是机器学习？"},
        ) as step:
            # 业务逻辑
            results = await search("什么是机器学习？")

            # 创建产物
            step.create_artifact(
                artifact_type=ArtifactTypeEnum.retrieval_results,
                artifact_data={"chunks": results},
            )

            # 设置输出
            step.set_output({"results_count": len(results)})
    """

    def __init__(
        self,
        trace_manager,
        step_type: StepTypeEnum,
        step_name: str,
        input: Dict[str, Any],
        parent_step_id: Optional[str] = None,
    ):
        """
        初始化 StepContext

        Args:
            trace_manager: TraceManager 实例
            step_type: 步骤类型（枚举）
            step_name: 步骤名称
            input: 步骤输入
            parent_step_id: 父步骤ID（用于嵌套）
        """
        self.trace_manager = trace_manager
        self.step_id = f"step_{uuid4()}"
        self.step_type = step_type
        self.step_name = step_name
        self.input = input
        self.parent_step_id = parent_step_id
        self.status = StepStatusEnum.pending

        # 产物ID列表
        self.artifact_ids: List[str] = []

        # 输出（由用户设置）
        self.output: Optional[Dict[str, Any]] = None

        # 时间戳
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.latency_ms: Optional[int] = None

    async def __aenter__(self):
        """进入步骤：发送 on_step_start 事件"""
        self.start_time = datetime.utcnow()
        self.status = StepStatusEnum.running

        # 更新 Trace 统计
        self.trace_manager.total_steps += 1

        await self.trace_manager._emit_event(
            {
                "event_id": f"evt_{uuid4()}",
                "event_type": "on_step_start",
                "timestamp": self.start_time.isoformat(),
                "trace_id": self.trace_manager.trace_id,
                "step_id": self.step_id,
                "parent_step_id": self.parent_step_id,
                "data": {
                    "step_type": self.step_type.value,
                    "step_name": self.step_name,
                    "input": self.input,
                },
            }
        )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """退出步骤：发送 on_step_complete/on_step_failed 事件"""
        self.end_time = datetime.utcnow()
        self.latency_ms = int((self.end_time - self.start_time).total_seconds() * 1000)

        if exc_type:
            # 步骤失败
            self.status = StepStatusEnum.failed
            self.trace_manager.failed_steps += 1

            error_code = ErrorCodeEnum.INTERNAL_ERROR
            error_message = str(exc_val)
            stack_trace = traceback.format_exc()

            await self.trace_manager._emit_event(
                {
                    "event_id": f"evt_{uuid4()}",
                    "event_type": "on_step_failed",
                    "timestamp": self.end_time.isoformat(),
                    "trace_id": self.trace_manager.trace_id,
                    "step_id": self.step_id,
                    "parent_step_id": self.parent_step_id,
                    "data": {
                        "error_code": error_code.value,
                        "error_message": error_message,
                        "stack_trace": stack_trace,
                    },
                }
            )

            # 添加到 trace.steps
            self.trace_manager.steps.append(
                {
                    "step_id": self.step_id,
                    "parent_step_id": self.parent_step_id,
                    "step_type": self.step_type.value,
                    "step_name": self.step_name,
                    "status": self.status.value,
                    "input": self.input,
                    "output": {},
                    "artifact_ids": self.artifact_ids,
                    "start_time": self.start_time.isoformat(),
                    "end_time": self.end_time.isoformat(),
                    "latency_ms": self.latency_ms,
                    "error_code": error_code.value,
                    "error_message": error_message,
                }
            )
        else:
            # 步骤成功
            self.status = StepStatusEnum.completed
            self.trace_manager.completed_steps += 1

            # 如果没有设置输出，使用空字典
            if self.output is None:
                self.output = {}

            await self.trace_manager._emit_event(
                {
                    "event_id": f"evt_{uuid4()}",
                    "event_type": "on_step_complete",
                    "timestamp": self.end_time.isoformat(),
                    "trace_id": self.trace_manager.trace_id,
                    "step_id": self.step_id,
                    "parent_step_id": self.parent_step_id,
                    "data": {
                        "output": self.output,
                        "latency_ms": self.latency_ms,
                        "artifact_ids": self.artifact_ids,
                    },
                }
            )

            # 添加到 trace.steps
            self.trace_manager.steps.append(
                {
                    "step_id": self.step_id,
                    "parent_step_id": self.parent_step_id,
                    "step_type": self.step_type.value,
                    "step_name": self.step_name,
                    "status": self.status.value,
                    "input": self.input,
                    "output": self.output,
                    "artifact_ids": self.artifact_ids,
                    "start_time": self.start_time.isoformat(),
                    "end_time": self.end_time.isoformat(),
                    "latency_ms": self.latency_ms,
                }
            )

        # 标记需要持久化
        self.trace_manager._mark_pending_persist()

    def create_artifact(
        self,
        artifact_type: ArtifactTypeEnum,
        artifact_data: Dict[str, Any],
    ) -> str:
        """
        创建产物

        Args:
            artifact_type: 产物类型（枚举）
            artifact_data: 产物数据

        Returns:
            str: 产物ID
        """
        artifact_id = f"artifact_{uuid4()}"

        # 发送 on_artifact_created 事件（异步）
        asyncio.create_task(
            self.trace_manager._emit_event(
                {
                    "event_id": f"evt_{uuid4()}",
                    "event_type": "on_artifact_created",
                    "timestamp": datetime.utcnow().isoformat(),
                    "trace_id": self.trace_manager.trace_id,
                    "step_id": self.step_id,
                    "data": {
                        "artifact_id": artifact_id,
                        "artifact_type": artifact_type.value,
                        "artifact_data": artifact_data,
                    },
                }
            )
        )

        # 添加到 trace.artifacts
        self.trace_manager.artifacts.append(
            {
                "artifact_id": artifact_id,
                "step_id": self.step_id,
                "artifact_type": artifact_type.value,
                "created_at": datetime.utcnow().isoformat(),
                "data": artifact_data,
            }
        )

        # 记录 artifact_id
        self.artifact_ids.append(artifact_id)

        return artifact_id

    def set_output(self, output: Dict[str, Any]):
        """
        设置步骤输出

        Args:
            output: 输出数据
        """
        self.output = output

    async def stream_update(self, update_type: str, update_data: Dict[str, Any]):
        """
        流式更新步骤（用于 LLM 流式输出）

        Args:
            update_type: 更新类型（如 "token_delta", "token_batch"）
            update_data: 更新数据
        """
        from service.chat.enums import StepUpdateTypeEnum

        await self.trace_manager._emit_event(
            {
                "event_id": f"evt_{uuid4()}",
                "event_type": "on_step_update",
                "timestamp": datetime.utcnow().isoformat(),
                "trace_id": self.trace_manager.trace_id,
                "step_id": self.step_id,
                "parent_step_id": self.parent_step_id,
                "data": {
                    "update_type": update_type,
                    "update_data": update_data,
                },
            }
        )

    async def update_progress(self, progress_percentage: float, current_status: Optional[str] = None):
        """
        更新步骤进度

        Args:
            progress_percentage: 进度百分比（0-100）
            current_status: 当前状态描述
        """
        await self.trace_manager._emit_event(
            {
                "event_id": f"evt_{uuid4()}",
                "event_type": "on_step_progress",
                "timestamp": datetime.utcnow().isoformat(),
                "trace_id": self.trace_manager.trace_id,
                "step_id": self.step_id,
                "parent_step_id": self.parent_step_id,
                "data": {
                    "progress_percentage": progress_percentage,
                    "current_status": current_status,
                },
            }
        )
