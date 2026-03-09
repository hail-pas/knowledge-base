"""
Artifact Schema Definitions

定义所有产物类型（Pydantic BaseModel）
产物是步骤执行产生的数据结果
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

from service.chat.enums import ArtifactTypeEnum


# =============================================================================
# 基础产物模型
# =============================================================================


class BaseArtifact(BaseModel):
    """产物基础模型"""

    artifact_id: str = Field(..., description="产物唯一ID，格式: artifact_ + UUID")
    step_id: str = Field(..., description="所属步骤ID")
    artifact_type: ArtifactTypeEnum = Field(..., description="产物类型")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


# =============================================================================
# 文本类产物
# =============================================================================


class TextArtifact(BaseArtifact):
    """文本产物"""

    artifact_type: ArtifactTypeEnum = ArtifactTypeEnum.text

    text: str = Field(..., description="文本内容")
    language: Optional[str] = Field(None, description="语言代码")


class JSONArtifact(BaseArtifact):
    """JSON数据产物"""

    artifact_type: ArtifactTypeEnum = ArtifactTypeEnum.json

    data: Dict[str, Any] = Field(..., description="JSON数据")


# =============================================================================
# 文件类产物
# =============================================================================


class ImageArtifact(BaseArtifact):
    """图片产物"""

    artifact_type: ArtifactTypeEnum = ArtifactTypeEnum.image

    oss_keys: List[str] = Field(..., description="OSS key列表")
    total_size_bytes: Optional[int] = Field(None, description="总大小（字节）")
    origin_names: List[str] = Field(..., description="原始文件名列表")
    mime_types: List[str] = Field(default_factory=list, description="MIME类型列表")


class FileArtifact(BaseArtifact):
    """文件产物"""

    artifact_type: ArtifactTypeEnum = ArtifactTypeEnum.file

    oss_keys: List[str] = Field(..., description="OSS key列表")
    total_size_bytes: Optional[int] = Field(None, description="总大小（字节）")
    origin_names: List[str] = Field(..., description="原始文件名列表")
    file_types: List[str] = Field(default_factory=list, description="文件类型列表")


# =============================================================================
# 检索相关产物
# =============================================================================


class RetrievalResultChunk(BaseModel):
    """检索结果块"""

    chunk_id: int
    document_id: int
    collection_id: int
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResultsArtifact(BaseArtifact):
    """检索结果产物"""

    artifact_type: ArtifactTypeEnum = ArtifactTypeEnum.retrieval_results

    chunks: List[RetrievalResultChunk] = Field(..., description="检索结果块列表")
    total_count: int = Field(..., description="结果总数")
    query: str = Field(..., description="检索查询")
    retrieval_method: str = Field(..., description="检索方法")


# =============================================================================
# 意图识别产物
# =============================================================================


class IntentArtifact(BaseArtifact):
    """意图产物"""

    artifact_type: ArtifactTypeEnum = ArtifactTypeEnum.intent

    intent: str = Field(..., description="意图")
    confidence: float = Field(..., description="置信度", ge=0, le=1)
    entities: List[Dict[str, Any]] = Field(default_factory=list, description="实体列表")


# =============================================================================
# 工具调用产物
# =============================================================================


class ToolCallArtifact(BaseArtifact):
    """工具调用产物"""

    artifact_type: ArtifactTypeEnum = ArtifactTypeEnum.tool_call

    tool_name: str = Field(..., description="工具名称")
    tool_args: Dict[str, Any] = Field(..., description="工具参数")


class ToolResultArtifact(BaseArtifact):
    """工具结果产物"""

    artifact_type: ArtifactTypeEnum = ArtifactTypeEnum.tool_result

    tool_name: str = Field(..., description="工具名称")
    success: bool = Field(..., description="是否成功")
    result: Optional[Dict[str, Any]] = Field(None, description="结果数据")
    error: Optional[str] = Field(None, description="错误信息")


# =============================================================================
# LLM相关产物
# =============================================================================


class LLMOutputArtifact(BaseArtifact):
    """LLM输出产物"""

    artifact_type: ArtifactTypeEnum = ArtifactTypeEnum.llm_output

    text: str = Field(..., description="生成的文本")
    finish_reason: Optional[str] = Field(None, description="结束原因")
    model: str = Field(..., description="使用的模型")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="工具调用列表")


class UsageStatsArtifact(BaseArtifact):
    """使用统计产物"""

    artifact_type: ArtifactTypeEnum = ArtifactTypeEnum.usage_stats

    input_tokens: int = Field(..., description="输入token数")
    output_tokens: int = Field(..., description="输出token数")
    total_tokens: int = Field(..., description="总token数")
    estimated_cost: Optional[float] = Field(None, description="预估成本（USD）")
    model: str = Field(..., description="使用的模型")


# =============================================================================
# 错误产物
# =============================================================================


class ErrorArtifact(BaseArtifact):
    """错误产物"""

    artifact_type: ArtifactTypeEnum = ArtifactTypeEnum.error

    error_code: str = Field(..., description="错误码")
    error_message: str = Field(..., description="错误信息")
    stack_trace: Optional[str] = Field(None, description="错误堆栈")
    retryable: bool = Field(default=False, description="是否可重试")


# =============================================================================
# 联合类型
# =============================================================================


Artifact = (
    TextArtifact
    | JSONArtifact
    | ImageArtifact
    | FileArtifact
    | RetrievalResultsArtifact
    | IntentArtifact
    | ToolCallArtifact
    | ToolResultArtifact
    | LLMOutputArtifact
    | UsageStatsArtifact
    | ErrorArtifact
)
