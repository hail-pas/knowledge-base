"""
Trace Manager - 核心管理器

管理整个聊天请求的生命周期，包括：
- 创建和管理 Trace
- 自动发送 WebSocket 事件
- 管理 Step 生命周期
- 自动持久化到数据库
"""

import asyncio
from typing import Optional, List, Dict, Any
from uuid import uuid4
from datetime import datetime
import traceback

from fastapi import WebSocket

from service.chat.schemas import (
    TraceStartEvent,
    TraceProgressEvent,
    TraceCompleteEvent,
    TraceErrorEvent,
    TraceCancelledEvent,
)
from service.chat.schemas.trace import Trace, TraceSummary
from service.chat.enums import TraceStatusEnum
from service.chat.runtime.step_context import StepContext


class TraceManager:
    """
    Trace 管理器（核心）

    管理整个聊天请求的生命周期，自动处理事件发送和数据库持久化。

    使用方式：
        async with TraceManager(
            user_id="user_123",
            session_id="session_456",
            chat_mode="rag",
            websocket=websocket,
        ) as trace:
            # 创建步骤
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
        user_id: str,
        session_id: str,
        chat_mode: str = "normal",
        llm_model: Optional[str] = None,
        websocket: Optional[WebSocket] = None,
        enable_event_stream: bool = False,
        persist_policy: str = "batch",
        batch_size: int = 3,
        batch_interval: int = 2,
        message_id: Optional[str] = None,
    ):
        """
        初始化 TraceManager

        Args:
            user_id: 用户ID
            session_id: 会话ID
            chat_mode: 聊天模式（normal/rag）
            llm_model: LLM模型名称
            websocket: WebSocket连接（可选，用于实时推送）
            enable_event_stream: 是否启用事件流快照（默认False）
            persist_policy: 持久化策略（"immediate_async" | "batch"）
            batch_size: 批量持久化的步骤数（默认3）
            batch_interval: 批量持久化的时间间隔（秒，默认2）
            message_id: 关联的消息ID（可选，稍后可以设置）
        """
        self.user_id = user_id
        self.session_id = session_id
        self.chat_mode = chat_mode
        self.llm_model = llm_model
        self.websocket = websocket

        # Trace 基础信息
        self.trace_id = f"trace_{uuid4()}"
        self.message_id = message_id
        self.status = TraceStatusEnum.pending

        # 数据存储
        self.steps: List[Dict[str, Any]] = []
        self.artifacts: List[Dict[str, Any]] = []
        self.event_stream: List[Dict[str, Any]] = []

        # 配置
        self.enable_event_stream = enable_event_stream
        self.persist_policy = persist_policy
        self.batch_size = batch_size
        self.batch_interval = batch_interval

        # 批量持久化状态
        self._persist_timer: Optional[asyncio.Task] = None
        self._pending_persist = False

        # 统计
        self.total_steps = 0
        self.completed_steps = 0
        self.failed_steps = 0
        self.cancelled_steps = 0

        # 时间戳
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    async def __aenter__(self):
        """进入上下文：发送 on_trace_start 事件"""
        self.start_time = datetime.utcnow()
        self.status = TraceStatusEnum.running

        await self._emit_event(
            TraceStartEvent(
                event_id=f"evt_{uuid4()}",
                event_type="on_trace_start",
                timestamp=self.start_time,
                trace_id=self.trace_id,
                data={
                    "trace_metadata": {
                        "user_id": self.user_id,
                        "session_id": self.session_id,
                        "chat_mode": self.chat_mode,
                        "llm_model": self.llm_model,
                    }
                },
            )
        )

        # 启动批量持久化定时器
        if self.persist_policy == "batch":
            self._start_persist_timer()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """退出上下文：发送完成事件并持久化"""
        self.end_time = datetime.utcnow()

        # 取消定时器
        if self._persist_timer:
            self._persist_timer.cancel()

        # 发送完成事件
        if exc_type:
            self.status = TraceStatusEnum.failed
            await self._emit_event(
                TraceErrorEvent(
                    event_id=f"evt_{uuid4()}",
                    event_type="on_trace_error",
                    timestamp=self.end_time,
                    trace_id=self.trace_id,
                    data={
                        "error_code": "INTERNAL_ERROR",
                        "error_message": str(exc_val),
                        "stack_trace": traceback.format_exc(),
                    },
                )
            )
        else:
            self.status = TraceStatusEnum.completed
            total_latency_ms = int((self.end_time - self.start_time).total_seconds() * 1000)

            await self._emit_event(
                TraceCompleteEvent(
                    event_id=f"evt_{uuid4()}",
                    event_type="on_trace_complete",
                    timestamp=self.end_time,
                    trace_id=self.trace_id,
                    data={
                        "summary": {
                            "total_steps": self.total_steps,
                            "successful_steps": self.completed_steps,
                            "failed_steps": self.failed_steps,
                            "cancelled_steps": self.cancelled_steps,
                            "total_artifacts": len(self.artifacts),
                        },
                        "total_latency_ms": total_latency_ms,
                    },
                )
            )

        # 最后一次持久化
        await self._persist_to_db()

    def step_context(
        self,
        step_type: str,
        step_name: str,
        input: Dict[str, Any],
        parent_step_id: Optional[str] = None,
    ):
        """
        创建步骤上下文管理器

        Args:
            step_type: 步骤类型（枚举值）
            step_name: 步骤名称
            input: 步骤输入
            parent_step_id: 父步骤ID（用于嵌套）

        Returns:
            StepContext: 步骤上下文管理器
        """
        from service.chat.enums import StepTypeEnum

        return StepContext(
            trace_manager=self,
            step_type=StepTypeEnum(step_type),
            step_name=step_name,
            input=input,
            parent_step_id=parent_step_id,
        )

    async def _emit_event(self, event: Dict[str, Any]):
        """
        发送事件到 WebSocket

        Args:
            event: 事件字典（Pydantic model_dump()）
        """
        # 发送到 WebSocket
        if self.websocket:
            try:
                await self.websocket.send_json(event)
            except Exception as e:
                print(f"⚠️  WebSocket 发送失败: {e}")

        # 存储事件流快照
        if self.enable_event_stream:
            self.event_stream.append(event)

    async def _persist_to_db(self):
        """
        持久化到数据库

        根据持久化策略选择同步或异步
        """
        if not self.message_id:
            return

        if self.persist_policy == "immediate_async":
            # 异步持久化（不阻塞）
            asyncio.create_task(self._do_persist())
        else:
            # 同步持久化
            await self._do_persist()

    async def _do_persist(self):
        """
        实际执行持久化

        更新或创建 ChatMessageArtifact 记录
        """
        from ext.ext_tortoise.models.knowledge_base import ChatMessageArtifact

        # 准备数据
        data = {
            "message_id": self.message_id,
            "steps": self.steps,
            "artifacts": self.artifacts,
            "event_stream": self.event_stream if self.enable_event_stream else [],
            "total_steps": self.total_steps,
            "pending_steps": 0,  # 暂时固定为0
            "running_steps": 0,  # 暂时固定为0
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "cancelled_steps": self.cancelled_steps,
            "total_artifacts": len(self.artifacts),
            "first_step_at": self.steps[0]["start_time"] if self.steps else None,
            "last_step_at": self.steps[-1]["end_time"] if self.steps else None,
        }

        try:
            # 更新或创建记录
            await ChatMessageArtifact.update_or_create(
                uid=self.trace_id,
                defaults=data,
            )
        except Exception as e:
            print(f"⚠️  数据库持久化失败: {e}")

    def _start_persist_timer(self):
        """启动批量持久化定时器"""

        async def persist_timer():
            while True:
                await asyncio.sleep(self.batch_interval)
                if self._pending_persist:
                    await self._persist_to_db()
                    self._pending_persist = False

        self._persist_timer = asyncio.create_task(persist_timer())

    def _mark_pending_persist(self):
        """
        标记需要持久化

        根据批量大小决定是否立即持久化
        """
        if self.persist_policy == "batch":
            self._pending_persist = True

            # 检查是否达到批量大小
            if len(self.steps) % self.batch_size == 0 and len(self.steps) > 0:
                asyncio.create_task(self._persist_to_db())
                self._pending_persist = False

    def set_message_id(self, message_id: str):
        """
        设置关联的消息ID

        Args:
            message_id: 消息ID
        """
        self.message_id = message_id
