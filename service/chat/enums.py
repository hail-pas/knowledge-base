"""
Chat Event Streaming Enums

基于实时事件流协议的枚举定义，支持 WebSocket 实时推送
协议文档: docs/event_streaming_protocol.md
"""

from core.types import StrEnum


# =============================================================================
# 事件类型枚举 (Event Types)
# =============================================================================


class EventTypeEnum(StrEnum):
    """WebSocket 事件类型枚举"""

    on_trace_start = ("on_trace_start", "请求开始")
    on_trace_progress = ("on_trace_progress", "进度更新")
    on_trace_complete = ("on_trace_complete", "请求完成")
    on_trace_error = ("on_trace_error", "请求失败")
    on_trace_cancelled = ("on_trace_cancelled", "请求取消")

    on_step_start = ("on_step_start", "步骤开始")
    on_step_update = ("on_step_update", "步骤更新")
    on_step_progress = ("on_step_progress", "步骤进度")
    on_step_complete = ("on_step_complete", "步骤完成")
    on_step_failed = ("on_step_failed", "步骤失败")
    on_step_cancelled = ("on_step_cancelled", "步骤取消")

    on_artifact_created = ("on_artifact_created", "产物创建")
    on_artifact_updated = ("on_artifact_updated", "产物更新")


# =============================================================================
# 追踪状态枚举 (Trace Status)
# =============================================================================


class TraceStatusEnum(StrEnum):
    """追踪状态枚举"""

    pending = ("pending", "等待处理")
    running = ("running", "处理中")
    completed = ("completed", "已完成")
    failed = ("failed", "失败")
    cancelled = ("cancelled", "已取消")


# =============================================================================
# 步骤状态枚举 (Step Status)
# =============================================================================


class StepStatusEnum(StrEnum):
    """步骤状态枚举"""

    pending = ("pending", "等待执行")
    running = ("running", "执行中")
    completed = ("completed", "执行成功")
    failed = ("failed", "执行失败")
    cancelled = ("cancelled", "已取消")


# =============================================================================
# 步骤类型枚举 (Step Types)
# =============================================================================


class StepTypeEnum(StrEnum):
    """步骤类型枚举"""

    user_input_processing = ("user_input_processing", "用户输入处理")
    intent_recognition = ("intent_recognition", "意图识别")
    history_compression = ("history_compression", "历史压缩")
    retrieval = ("retrieval", "检索")
    tool_call = ("tool_call", "工具调用")
    llm_call = ("llm_call", "LLM调用")
    response_generation = ("response_generation", "响应生成")
    custom = ("custom", "自定义步骤")


class RetrievalSubTypeEnum(StrEnum):
    """检索步骤子类型枚举"""

    dense_retrieval = ("dense_retrieval", "稠密检索")
    sparse_retrieval = ("sparse_retrieval", "稀疏检索")
    hybrid_retrieval = ("hybrid_retrieval", "混合检索")
    rerank = ("rerank", "重排序")


class LLMSubTypeEnum(StrEnum):
    """LLM调用步骤子类型枚举"""

    llm_thinking = ("llm_thinking", "LLM思考")
    llm_generation = ("llm_generation", "LLM生成")
    llm_summarization = ("llm_summarization", "LLM总结")
    llm_extraction = ("llm_extraction", "LLM提取")


# =============================================================================
# 产物类型枚举 (Artifact Types)
# =============================================================================


class ArtifactTypeEnum(StrEnum):
    """产物类型枚举"""

    text = ("text", "文本")
    json = ("json", "JSON数据")
    image = ("image", "图片")
    file = ("file", "文件")
    retrieval_results = ("retrieval_results", "检索结果")
    intent = ("intent", "意图")
    tool_call = ("tool_call", "工具调用")
    tool_result = ("tool_result", "工具结果")
    llm_output = ("llm_output", "LLM输出")
    usage_stats = ("usage_stats", "使用统计")
    error = ("error", "错误")


# =============================================================================
# 步骤更新类型枚举 (Step Update Types)
# =============================================================================


class StepUpdateTypeEnum(StrEnum):
    """步骤更新类型枚举"""

    token_delta = ("token_delta", "Token增量")
    token_batch = ("token_batch", "Token批量")
    progress_update = ("progress_update", "进度更新")
    status_change = ("status_change", "状态变更")
    data_update = ("data_update", "数据更新")


# =============================================================================
# 错误码枚举 (Error Codes)
# =============================================================================


class ErrorCodeEnum(StrEnum):
    """错误码枚举"""

    LLM_API_ERROR = ("LLM_API_ERROR", "LLM API调用失败")
    RETRIEVAL_FAILED = ("RETRIEVAL_FAILED", "检索服务失败")
    TIMEOUT = ("TIMEOUT", "请求超时")
    INVALID_INPUT = ("INVALID_INPUT", "输入参数无效")
    RATE_LIMIT = ("RATE_LIMIT", "触发速率限制")
    AUTH_FAILED = ("AUTH_FAILED", "认证失败")
    INTERNAL_ERROR = ("INTERNAL_ERROR", "内部错误")
