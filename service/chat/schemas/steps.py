"""
Step Schema Definitions

定义所有步骤相关的数据结构（Pydantic BaseModel）
步骤是逻辑处理单元，支持嵌套
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime

from service.chat.enums import StepTypeEnum, StepStatusEnum


# =============================================================================
# 基础步骤模型
# =============================================================================


class BaseStep(BaseModel):
    """步骤基础模型"""

    step_id: str = Field(..., description="步骤唯一ID，格式: step_ + UUID")
    parent_step_id: Optional[str] = Field(None, description="父步骤ID（用于嵌套）")
    step_type: StepTypeEnum = Field(..., description="步骤类型")
    step_name: str = Field(..., description="步骤名称")
    status: StepStatusEnum = Field(default=StepStatusEnum.pending, description="步骤状态")
    input: Dict[str, Any] = Field(default_factory=dict, description="步骤输入")
    output: Dict[str, Any] = Field(default_factory=dict, description="步骤输出")
    artifact_ids: List[str] = Field(default_factory=list, description="产生的产物ID列表")

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    latency_ms: Optional[int] = None

    error_code: Optional[str] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None

    metadata: Dict[str, Any] = Field(default_factory=dict, description="步骤元数据")


# =============================================================================
# 具体步骤模型
# =============================================================================


class UserInputProcessingStep(BaseStep):
    """用户输入处理步骤"""

    step_type: StepTypeEnum = StepTypeEnum.user_input_processing
    step_name: str = "用户输入处理"

    class InputData(BaseModel):
        text: Optional[str] = None
        images: List[str] = Field(default_factory=list)
        files: List[str] = Field(default_factory=list)

    class OutputData(BaseModel):
        parsed_text: Optional[str] = None
        language: Optional[str] = None

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
        entities: List[Dict[str, Any]] = Field(default_factory=list)

    input: InputData
    output: OutputData


class HistoryCompressionStep(BaseStep):
    """历史压缩步骤"""

    step_type: StepTypeEnum = StepTypeEnum.history_compression
    step_name: str = "历史压缩"

    class InputData(BaseModel):
        messages: List[Dict[str, Any]]
        max_tokens: int

    class OutputData(BaseModel):
        compressed_messages: List[Dict[str, Any]]
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
        collection_ids: List[int]
        top_k: int
        filters: Dict[str, Any] = Field(default_factory=dict)

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
        tool_args: Dict[str, Any]

    class OutputData(BaseModel):
        tool_name: str
        success: bool
        result: Optional[Dict[str, Any]] = None
        error: Optional[str] = None

    input: InputData
    output: OutputData


class LLMCallStep(BaseStep):
    """LLM调用步骤"""

    step_type: StepTypeEnum = StepTypeEnum.llm_call
    step_name: str = "LLM调用"

    class InputData(BaseModel):
        model: str
        messages: List[Dict[str, Any]]
        temperature: Optional[float] = None
        max_tokens: Optional[int] = None
        tools: Optional[List[Dict[str, Any]]] = None

    class OutputData(BaseModel):
        text: Optional[str] = None
        finish_reason: Optional[str] = None
        tool_calls: Optional[List[Dict[str, Any]]] = None

    input: InputData
    output: OutputData


class ResponseGenerationStep(BaseStep):
    """响应生成步骤"""

    step_type: StepTypeEnum = StepTypeEnum.response_generation
    step_name: str = "响应生成"

    class InputData(BaseModel):
        llm_output: Dict[str, Any]
        artifacts: Dict[str, Any]

    class OutputData(BaseModel):
        final_response: str
        display_format: str = "markdown"

    input: InputData
    output: OutputData


class CustomStep(BaseStep):
    """自定义步骤"""

    step_type: StepTypeEnum = StepTypeEnum.custom
    step_name: str = Field(..., description="自定义步骤名称")
    input: Dict[str, Any]
    output: Dict[str, Any]
