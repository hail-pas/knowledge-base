"""
Step Schema Definitions

定义所有步骤相关的数据结构（Pydantic BaseModel）
步骤是逻辑处理单元，支持嵌套
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from pydantic import Field, BaseModel

from service.chat.enums import StepTypeEnum, StepStatusEnum

# =============================================================================
# 基础步骤模型
# =============================================================================


class BaseStep(BaseModel):
    """步骤基础模型"""

    step_id: str = Field(..., description="步骤唯一ID，格式: step_ + UUID")
    parent_step_id: str | None = Field(None, description="父步骤ID（用于嵌套）")
    step_type: StepTypeEnum = Field(..., description="步骤类型")
    step_name: str = Field(..., description="步骤名称")
    status: StepStatusEnum = Field(default=StepStatusEnum.pending, description="步骤状态")
    input: dict[str, Any] = Field(default_factory=dict, description="步骤输入")
    output: dict[str, Any] = Field(default_factory=dict, description="步骤输出")
    artifact_ids: list[str] = Field(default_factory=list, description="产生的产物ID列表")

    start_time: datetime | None = None
    end_time: datetime | None = None
    latency_ms: int | None = None

    error_code: str | None = None
    error_message: str | None = None
    stack_trace: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict, description="步骤元数据")


# =============================================================================
# 具体步骤模型
# =============================================================================


class UserInputProcessingStep(BaseStep):
    """用户输入处理步骤"""

    step_type: StepTypeEnum = StepTypeEnum.user_input_processing
    step_name: str = "用户输入处理"

    class InputData(BaseModel):
        text: str | None = None
        images: list[str] = Field(default_factory=list)
        files: list[str] = Field(default_factory=list)

    class OutputData(BaseModel):
        parsed_text: str | None = None
        language: str | None = None

    input: InputData
    output: OutputData


class IntentRecognitionStep(BaseStep):
    """意图识别步骤"""

    step_type: StepTypeEnum = StepTypeEnum.intent_recognition
    step_name: str = "意图识别"

    class InputData(BaseModel):
        text: str

    class OutputData(BaseModel):
        intent: str
        confidence: float
        entities: list[dict[str, Any]] = Field(default_factory=list)

    input: InputData
    output: OutputData


class HistoryCompressionStep(BaseStep):
    """历史压缩步骤"""

    step_type: StepTypeEnum = StepTypeEnum.history_compression
    step_name: str = "历史压缩"

    class InputData(BaseModel):
        messages: list[dict[str, Any]]
        max_tokens: int

    class OutputData(BaseModel):
        compressed_messages: list[dict[str, Any]]
        original_count: int
        compressed_count: int
        saved_tokens: int

    input: InputData
    output: OutputData


class RetrievalStep(BaseStep):
    """检索步骤"""

    step_type: StepTypeEnum = StepTypeEnum.retrieval
    step_name: str = "知识库检索"

    class InputData(BaseModel):
        query: str
        collection_ids: list[int]
        top_k: int
        filters: dict[str, Any] = Field(default_factory=dict)

    class OutputData(BaseModel):
        results_count: int
        query: str
        retrieval_method: str

    input: InputData
    output: OutputData


class ToolCallStep(BaseStep):
    """工具调用步骤"""

    step_type: StepTypeEnum = StepTypeEnum.tool_call
    step_name: str = "工具调用"

    class InputData(BaseModel):
        tool_name: str
        tool_args: dict[str, Any]

    class OutputData(BaseModel):
        tool_name: str
        success: bool
        result: dict[str, Any] | None = None
        error: str | None = None

    input: InputData
    output: OutputData


class LLMCallStep(BaseStep):
    """LLM调用步骤"""

    step_type: StepTypeEnum = StepTypeEnum.llm_call
    step_name: str = "LLM调用"

    class InputData(BaseModel):
        model: str
        messages: list[dict[str, Any]]
        temperature: float | None = None
        max_tokens: int | None = None
        tools: list[dict[str, Any]] | None = None

    class OutputData(BaseModel):
        text: str | None = None
        finish_reason: str | None = None
        tool_calls: list[dict[str, Any]] | None = None

    input: InputData
    output: OutputData


class ResponseGenerationStep(BaseStep):
    """响应生成步骤"""

    step_type: StepTypeEnum = StepTypeEnum.response_generation
    step_name: str = "响应生成"

    class InputData(BaseModel):
        llm_output: dict[str, Any]
        artifacts: dict[str, Any]

    class OutputData(BaseModel):
        final_response: str
        display_format: str = "markdown"

    input: InputData
    output: OutputData


class CustomStep(BaseStep):
    """自定义步骤"""

    step_type: StepTypeEnum = StepTypeEnum.custom
    step_name: str = Field(..., description="自定义步骤名称")
    input: dict[str, Any]
    output: dict[str, Any]
