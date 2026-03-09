"""
Trace Schema Definitions

定义请求追踪相关的数据结构（Pydantic BaseModel）
Trace 代表一次完整的聊天请求
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

from service.chat.enums import TraceStatusEnum


# =============================================================================
# Trace 模型
# =============================================================================


class Trace(BaseModel):
    """请求追踪模型"""

    trace_id: str = Field(..., description="追踪ID，格式: trace_ + UUID")
    user_id: Optional[str] = Field(None, description="用户ID")
    session_id: Optional[str] = Field(None, description="会话ID")

    status: TraceStatusEnum = Field(default=TraceStatusEnum.pending, description="追踪状态")

    chat_mode: Optional[str] = Field(None, description="聊天模式: normal/rag")
    llm_model: Optional[str] = Field(None, description="使用的LLM模型")

    input_data: Dict[str, Any] = Field(default_factory=dict, description="输入数据")
    output_data: Dict[str, Any] = Field(default_factory=dict, description="输出数据")

    step_ids: List[str] = Field(default_factory=list, description="包含的步骤ID列表")
    artifact_ids: List[str] = Field(default_factory=list, description="包含的产物ID列表")

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_latency_ms: Optional[int] = None

    error_code: Optional[str] = None
    error_message: Optional[str] = None

    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


# =============================================================================
# Trace 统计摘要
# =============================================================================


class TraceSummary(BaseModel):
    """追踪摘要"""

    trace_id: str
    status: TraceStatusEnum
    total_steps: int
    successful_steps: int
    failed_steps: int
    cancelled_steps: int
    total_artifacts: int
    total_latency_ms: Optional[int] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
