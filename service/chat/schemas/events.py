"""
WebSocket Event Schema Definitions

定义所有实时推送的事件类型（Pydantic BaseModel）
用于 WebSocket 实时通信
"""

from typing import Any, Dict, Literal, Optional
from datetime import datetime

from pydantic import Field, BaseModel

from service.chat.enums import (
    StepTypeEnum,
    ErrorCodeEnum,
    EventTypeEnum,
    ArtifactTypeEnum,
    StepUpdateTypeEnum,
)

# =============================================================================
# 通用事件基础模型
# =============================================================================


class BaseEvent(BaseModel):
    """事件基础模型"""

    event_id: str = Field(..., description="事件唯一ID，格式: evt_ + UUID")
    event_type: EventTypeEnum = Field(..., description="事件类型")
    timestamp: datetime = Field(..., description="事件时间戳（ISO 8601）")
    trace_id: str = Field(..., description="关联的追踪ID，格式: trace_ + UUID")
    step_id: str | None = Field(None, description="关联的步骤ID，格式: step_ + UUID")
    parent_step_id: str | None = Field(None, description="父步骤ID（用于嵌套步骤）")


# =============================================================================
# Trace 级别事件
# =============================================================================


class TraceMetadata(BaseModel):
    """追踪元数据"""

    user_id: str | None = None
    session_id: str | None = None
    chat_mode: str | None = None
    llm_model: str | None = None


class TraceStartEventData(BaseModel):
    """请求开始事件数据"""

    trace_metadata: TraceMetadata


class TraceStartEvent(BaseEvent):
    """请求开始事件"""

    event_type: Literal[EventTypeEnum.on_trace_start] = EventTypeEnum.on_trace_start
    data: TraceStartEventData


class TraceProgressEventData(BaseModel):
    """进度更新事件数据"""

    progress_percentage: float = Field(..., ge=0, le=100, description="进度百分比")
    message: str | None = None


class TraceProgressEvent(BaseEvent):
    """进度更新事件"""

    event_type: Literal[EventTypeEnum.on_trace_progress] = EventTypeEnum.on_trace_progress
    data: TraceProgressEventData


class TraceSummary(BaseModel):
    """追踪摘要"""

    total_steps: int
    successful_steps: int
    failed_steps: int
    total_artifacts: int


class TraceCompleteEventData(BaseModel):
    """请求完成事件数据"""

    summary: TraceSummary
    total_latency_ms: int


class TraceCompleteEvent(BaseEvent):
    """请求完成事件"""

    event_type: Literal[EventTypeEnum.on_trace_complete] = EventTypeEnum.on_trace_complete
    data: TraceCompleteEventData


class TraceErrorEventData(BaseModel):
    """请求失败事件数据"""

    error_code: ErrorCodeEnum
    error_message: str
    stack_trace: str | None = None


class TraceErrorEvent(BaseEvent):
    """请求失败事件"""

    event_type: Literal[EventTypeEnum.on_trace_error] = EventTypeEnum.on_trace_error
    data: TraceErrorEventData


class TraceCancelledEventData(BaseModel):
    """请求取消事件数据"""

    reason: str | None = None


class TraceCancelledEvent(BaseEvent):
    """请求取消事件"""

    event_type: Literal[EventTypeEnum.on_trace_cancelled] = EventTypeEnum.on_trace_cancelled
    data: TraceCancelledEventData


# =============================================================================
# Step 级别事件
# =============================================================================


class StepStartEventData(BaseModel):
    """步骤开始事件数据"""

    step_type: StepTypeEnum
    step_name: str
    input: dict[str, Any]


class StepStartEvent(BaseEvent):
    """步骤开始事件"""

    event_type: Literal[EventTypeEnum.on_step_start] = EventTypeEnum.on_step_start
    data: StepStartEventData


class TokenDeltaUpdate(BaseModel):
    """Token增量更新数据"""

    token: str
    index: int


class TokenBatchUpdate(BaseModel):
    """Token批量更新数据"""

    tokens: list[str]
    indices: list[int]


class ProgressUpdate(BaseModel):
    """进度更新数据"""

    progress_percentage: float = Field(..., ge=0, le=100)
    current_status: str | None = None


class StepUpdateEventData(BaseModel):
    """步骤更新事件数据"""

    update_type: StepUpdateTypeEnum
    update_data: dict[str, Any]


class StepUpdateEvent(BaseEvent):
    """步骤更新事件"""

    event_type: Literal[EventTypeEnum.on_step_update] = EventTypeEnum.on_step_update
    data: StepUpdateEventData


class StepProgressEventData(BaseModel):
    """步骤进度事件数据"""

    progress_percentage: float = Field(..., ge=0, le=100)
    current_status: str | None = None


class StepProgressEvent(BaseEvent):
    """步骤进度事件"""

    event_type: Literal[EventTypeEnum.on_step_progress] = EventTypeEnum.on_step_progress
    data: StepProgressEventData


class StepCompleteEventData(BaseModel):
    """步骤完成事件数据"""

    output: dict[str, Any]
    latency_ms: int
    artifact_ids: list[str] = Field(default_factory=list)


class StepCompleteEvent(BaseEvent):
    """步骤完成事件"""

    event_type: Literal[EventTypeEnum.on_step_complete] = EventTypeEnum.on_step_complete
    data: StepCompleteEventData


class StepFailedEventData(BaseModel):
    """步骤失败事件数据"""

    error_code: ErrorCodeEnum
    error_message: str
    stack_trace: str | None = None


class StepFailedEvent(BaseEvent):
    """步骤失败事件"""

    event_type: Literal[EventTypeEnum.on_step_failed] = EventTypeEnum.on_step_failed
    data: StepFailedEventData


class StepCancelledEventData(BaseModel):
    """步骤取消事件数据"""

    reason: str | None = None


class StepCancelledEvent(BaseEvent):
    """步骤取消事件"""

    event_type: Literal[EventTypeEnum.on_step_cancelled] = EventTypeEnum.on_step_cancelled
    data: StepCancelledEventData


# =============================================================================
# Artifact 级别事件
# =============================================================================


class ArtifactCreatedEventData(BaseModel):
    """产物创建事件数据"""

    artifact_id: str = Field(..., description="产物ID，格式: artifact_ + UUID")
    artifact_type: ArtifactTypeEnum
    artifact_data: dict[str, Any]


class ArtifactCreatedEvent(BaseEvent):
    """产物创建事件"""

    event_type: Literal[EventTypeEnum.on_artifact_created] = EventTypeEnum.on_artifact_created
    data: ArtifactCreatedEventData


class ArtifactUpdatedEventData(BaseModel):
    """产物更新事件数据"""

    artifact_id: str
    updated_fields: dict[str, Any]


class ArtifactUpdatedEvent(BaseEvent):
    """产物更新事件"""

    event_type: Literal[EventTypeEnum.on_artifact_updated] = EventTypeEnum.on_artifact_updated
    data: ArtifactUpdatedEventData


# =============================================================================
# 联合类型
# =============================================================================


Event = (
    TraceStartEvent
    | TraceProgressEvent
    | TraceCompleteEvent
    | TraceErrorEvent
    | TraceCancelledEvent
    | StepStartEvent
    | StepUpdateEvent
    | StepProgressEvent
    | StepCompleteEvent
    | StepFailedEvent
    | StepCancelledEvent
    | ArtifactCreatedEvent
    | ArtifactUpdatedEvent
)


# =============================================================================
# WebSocket 消息包装器
# =============================================================================


class WebSocketMessage(BaseModel):
    """WebSocket 消息包装器"""

    event_id: str
    event_type: EventTypeEnum
    timestamp: datetime
    trace_id: str
    step_id: str | None = None
    parent_step_id: str | None = None
    data: dict[str, Any]

    class Config:
        json_schema_extra = {
            "example": {
                "event_id": "evt_abc123def456",
                "event_type": "on_step_start",
                "timestamp": "2024-01-01T00:00:00.000Z",
                "trace_id": "trace_xyz789",
                "step_id": "step_123",
                "parent_step_id": None,
                "data": {"step_type": "retrieval", "step_name": "知识库检索", "input": {}},
            },
        }
