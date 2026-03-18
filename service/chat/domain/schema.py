from __future__ import annotations

from typing import Any, Generic, Literal, TypeVar, Annotated
from datetime import datetime

from pydantic import Field, BaseModel, ConfigDict

from core.types import StrEnum
from service.document.schema import (
    DocumentList,
    DocumentChunkList,
    DocumentGeneratedFaqList,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


PayloadT = TypeVar("PayloadT", bound=BaseModel)
DetailT = TypeVar("DetailT", bound=BaseModel)


class ChatRoleEnum(StrEnum):
    user = ("user", "用户")
    assistant = ("assistant", "助手")
    system = ("system", "系统")
    tool = ("tool", "工具")


class ContentBlockTypeEnum(StrEnum):
    text = ("text", "文本")
    file = ("file", "文件")
    json = ("json", "JSON")
    retrieval_hit = ("retrieval_hit", "检索命中")
    error = ("error", "错误")


class EventNameEnum(StrEnum):
    ack = ("ack", "确认")
    warning = ("warning", "警告")
    error = ("error", "错误")
    ping = ("ping", "心跳")
    turn_accepted = ("turn.accepted", "turn已受理")
    turn_started = ("turn.started", "turn已开始")
    turn_completed = ("turn.completed", "turn已完成")
    turn_failed = ("turn.failed", "turn失败")
    turn_canceled = ("turn.canceled", "turn已取消")
    step_created = ("step.created", "step已创建")
    step_started = ("step.started", "step已开始")
    step_completed = ("step.completed", "step已完成")
    step_failed = ("step.failed", "step失败")
    data_created = ("data.created", "数据已创建")
    data_completed = ("data.completed", "数据已完成")
    message_delta = ("message.delta", "消息增量")
    message_completed = ("message.completed", "消息完成")


class ChatWarningCodeEnum(StrEnum):
    warning = ("warning", "通用警告")
    knowledge_retrieval_skipped = ("knowledge_retrieval_skipped", "知识库检索跳过")
    function_call_skipped = ("function_call_skipped", "函数调用跳过")
    tool_call_skipped = ("tool_call_skipped", "工具调用跳过")


class ChatErrorCodeEnum(StrEnum):
    chat_error = ("chat_error", "通用聊天错误")
    auth_failed = ("auth_failed", "鉴权失败")
    socket_init_failed = ("socket_init_failed", "WebSocket 初始化失败")
    socket_session_error = ("socket_session_error", "WebSocket 会话初始化失败")
    unknown_command = ("unknown_command", "未知命令")
    command_error = ("command_error", "命令执行失败")


class ChatCapabilityKindEnum(StrEnum):
    intent_detection = ("intent_detection", "意图识别")
    system_prompt = ("system_prompt", "系统提示词")
    knowledge_retrieval = ("knowledge_retrieval", "知识库检索")
    function_call = ("function_call", "函数调用")
    tool_call = ("tool_call", "工具调用")
    mcp_call = ("mcp_call", "MCP调用")
    llm_response = ("llm_response", "模型响应")


CAPABILITY_EXECUTION_ORDER: dict[ChatCapabilityKindEnum, int] = {
    ChatCapabilityKindEnum.system_prompt: 10,
    ChatCapabilityKindEnum.intent_detection: 20,
    ChatCapabilityKindEnum.knowledge_retrieval: 30,
    ChatCapabilityKindEnum.function_call: 40,
    ChatCapabilityKindEnum.tool_call: 50,
    ChatCapabilityKindEnum.mcp_call: 60,
    ChatCapabilityKindEnum.llm_response: 90,
}


def capability_execution_order(kind: ChatCapabilityKindEnum) -> int:
    return CAPABILITY_EXECUTION_ORDER.get(kind, 999)


RESOURCE_SELECTION_TOP_LEVEL_KEYS = frozenset(
    {
        "use_system_defaults",
        "use_conversation_defaults",
        "capability_profile_ids",
        "capability_binding_ids",
        "capabilities",
        "metadata",
    },
)


class ClientCommandEnum(StrEnum):
    turn_start = ("turn.start", "启动turn")
    turn_cancel = ("turn.cancel", "取消turn")
    turn_resume = ("turn.resume", "回放turn")
    ack = ("ack", "客户端确认")
    ping = ("ping", "心跳")


class ChatPayloadTypeEnum(StrEnum):
    message_bundle = ("message_bundle", "消息包")
    intent_result = ("intent_result", "意图结果")
    function_result = ("function_result", "函数结果")
    prompt_context = ("prompt_context", "提示词上下文")
    retrieval_hit_list = ("retrieval_hit_list", "检索命中列表")
    tool_result = ("tool_result", "工具结果")
    mcp_result = ("mcp_result", "MCP结果")


class ResponseModeEnum(StrEnum):
    text = ("text", "文本输出")
    json = ("json", "JSON输出")


class ToolExecutionPolicyEnum(StrEnum):
    stub = ("stub", "占位执行")
    optional = ("optional", "可选执行")
    required = ("required", "必须执行")


class TextBlock(StrictModel):
    type: Literal["text"] = "text"
    text: str = Field(min_length=1)


class FileBlock(StrictModel):
    type: Literal["file"] = "file"
    file_id: str | None = None
    name: str
    mime_type: str | None = None
    url: str | None = None


class JsonBlock(StrictModel):
    type: Literal["json"] = "json"
    data: dict[str, Any]


class RetrievalBlock(StrictModel):
    type: Literal["retrieval_hit"] = "retrieval_hit"
    source_id: str
    collection_id: int
    document_id: int | None = None
    score: float
    snippet: str
    document: DocumentList | None = None
    chunk: DocumentChunkList | None = None
    faq: DocumentGeneratedFaqList | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ErrorBlock(StrictModel):
    type: Literal["error"] = "error"
    code: ChatErrorCodeEnum = ChatErrorCodeEnum.chat_error
    message: str


ContentBlock = Annotated[
    TextBlock | FileBlock | JsonBlock | RetrievalBlock | ErrorBlock,
    Field(discriminator="type"),
]


class MessageBundlePayload(StrictModel):
    type: Literal["message_bundle"] = "message_bundle"
    role: ChatRoleEnum
    blocks: list[ContentBlock] = Field(min_length=1)

    @property
    def text(self) -> str:
        return "\n".join(block.text for block in self.blocks if isinstance(block, TextBlock)).strip()


class MessageDeltaPayload(StrictModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class StepMetricPayload(StrictModel):
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    retrieval_hit_count: int | None = None


class DataRefSchema(StrictModel):
    ref_type: str = Field(min_length=1, max_length=64)
    ref_id: str = Field(min_length=1, max_length=128)
    label: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UsagePayload(StrictModel):
    requests: int = Field(default=0, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class KnowledgeRetrievalConfig(StrictModel):
    collection_ids: list[int] = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class LLMResponseConfig(StrictModel):
    llm_model_config_id: int | None = None
    response_mode: ResponseModeEnum = ResponseModeEnum.text


class ToolCallConfig(StrictModel):
    policy: ToolExecutionPolicyEnum = ToolExecutionPolicyEnum.stub
    tool_names: list[str] = Field(default_factory=list)


class MCPCallConfig(StrictModel):
    server_name: str = Field(min_length=1, max_length=128)
    tool_names: list[str] = Field(default_factory=list)
    timeout_ms: int = Field(default=30000, ge=1000, le=300000)


class IntentRule(StrictModel):
    intent: str = Field(min_length=1, max_length=64)
    description: str = Field(default="", max_length=500)
    keywords: list[str] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)
    terminal_response: str | None = Field(default=None, max_length=4000)


class IntentDetectionConfig(StrictModel):
    intents: list[IntentRule] = Field(min_length=1)
    default_intent: str = Field(default="general", min_length=1, max_length=64)
    add_to_context: bool = True
    add_instruction: bool = True
    set_terminal_response: bool = False


class SystemPromptTemplateKeyEnum(StrEnum):
    default = ("default", "默认模板")


class SystemPromptPlaceholderEnum(StrEnum):
    capability_summary = ("capability_summary", "能力摘要")
    intent_summary = ("intent_summary", "意图摘要")
    function_summary = ("function_summary", "函数摘要")
    conversation_summary = ("conversation_summary", "会话摘要")
    instructions_summary = ("instructions_summary", "额外约束")
    context_policy = ("context_policy", "上下文策略")


DEFAULT_SYSTEM_PROMPT_PLACEHOLDERS: tuple[SystemPromptPlaceholderEnum, ...] = (
    SystemPromptPlaceholderEnum.capability_summary,
    SystemPromptPlaceholderEnum.intent_summary,
    SystemPromptPlaceholderEnum.function_summary,
    SystemPromptPlaceholderEnum.conversation_summary,
    SystemPromptPlaceholderEnum.instructions_summary,
    SystemPromptPlaceholderEnum.context_policy,
)


class SystemPromptConfig(StrictModel):
    template_key: SystemPromptTemplateKeyEnum = SystemPromptTemplateKeyEnum.default
    include_placeholders: list[SystemPromptPlaceholderEnum] = Field(default_factory=list)
    highlight_capabilities: list[ChatCapabilityKindEnum] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)
    variable_overrides: dict[str, str] = Field(default_factory=dict)


class FunctionCallResultModeEnum(StrEnum):
    auto = ("auto", "自动")
    context = ("context", "作为上下文")
    terminal = ("terminal", "直接返回")


class FunctionToolSpec(StrictModel):
    tool_name: str = Field(min_length=1, max_length=64)
    title: str | None = Field(default=None, max_length=200)
    result_mode: FunctionCallResultModeEnum = FunctionCallResultModeEnum.auto


class FunctionCallConfig(StrictModel):
    tools: list[FunctionToolSpec] = Field(min_length=1)
    fail_on_no_match: bool = False
    stop_after_terminal: bool = True


class BaseResourceCapability(StrictModel):
    capability_id: str | None = Field(default=None, min_length=1, max_length=128)
    profile_id: int | None = Field(default=None, ge=1)
    binding_id: int | None = Field(default=None, ge=1)
    name: str | None = Field(default=None, min_length=1, max_length=100)
    source: str | None = Field(default=None, min_length=1, max_length=64)
    enabled: bool = True
    priority: int = Field(default=100, ge=0, le=1000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeRetrievalCapability(BaseResourceCapability):
    kind: Literal[ChatCapabilityKindEnum.knowledge_retrieval] = ChatCapabilityKindEnum.knowledge_retrieval
    config: KnowledgeRetrievalConfig


class IntentDetectionCapability(BaseResourceCapability):
    kind: Literal[ChatCapabilityKindEnum.intent_detection] = ChatCapabilityKindEnum.intent_detection
    config: IntentDetectionConfig


class SystemPromptCapability(BaseResourceCapability):
    kind: Literal[ChatCapabilityKindEnum.system_prompt] = ChatCapabilityKindEnum.system_prompt
    config: SystemPromptConfig


class FunctionCallCapability(BaseResourceCapability):
    kind: Literal[ChatCapabilityKindEnum.function_call] = ChatCapabilityKindEnum.function_call
    config: FunctionCallConfig


class ToolCallCapability(BaseResourceCapability):
    kind: Literal[ChatCapabilityKindEnum.tool_call] = ChatCapabilityKindEnum.tool_call
    config: ToolCallConfig = Field(default_factory=ToolCallConfig)


class MCPCallCapability(BaseResourceCapability):
    kind: Literal[ChatCapabilityKindEnum.mcp_call] = ChatCapabilityKindEnum.mcp_call
    config: MCPCallConfig


class LLMResponseCapability(BaseResourceCapability):
    kind: Literal[ChatCapabilityKindEnum.llm_response] = ChatCapabilityKindEnum.llm_response
    config: LLMResponseConfig = Field(default_factory=LLMResponseConfig)


ResourceCapability = Annotated[
    IntentDetectionCapability
    | SystemPromptCapability
    | KnowledgeRetrievalCapability
    | FunctionCallCapability
    | ToolCallCapability
    | MCPCallCapability
    | LLMResponseCapability,
    Field(discriminator="kind"),
]


class ResourceSelection(StrictModel):
    use_system_defaults: bool = True
    use_conversation_defaults: bool = True
    capability_profile_ids: list[int] = Field(default_factory=list)
    capability_binding_ids: list[int] = Field(default_factory=list)
    capabilities: list[ResourceCapability] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def capabilities_by_kind(self) -> dict[ChatCapabilityKindEnum, list[ResourceCapability]]:
        grouped: dict[ChatCapabilityKindEnum, list[ResourceCapability]] = {}
        for item in self.capabilities:
            if not item.enabled:
                continue
            grouped.setdefault(item.kind, []).append(item)
        return grouped

    def normalized_capabilities(self) -> list[ResourceCapability]:
        normalized: list[ResourceCapability] = []
        seen_ids: set[str] = set()
        has_llm = False
        for item in self.capabilities:
            if not item.enabled:
                continue
            if item.capability_id and item.capability_id in seen_ids:
                continue
            if item.capability_id:
                seen_ids.add(item.capability_id)
            normalized.append(item)
            if item.kind == ChatCapabilityKindEnum.llm_response:
                has_llm = True

        if not has_llm:
            normalized.append(
                LLMResponseCapability(
                    capability_id="builtin:llm_response",
                    name="llm_response",
                    source="builtin",
                    priority=100,
                ),
            )

        sorted_items = sorted(
            normalized,
            key=lambda item: (
                item.priority,
                capability_execution_order(item.kind),
                item.capability_id or item.name or "",
            ),
        )
        final_items: list[ResourceCapability] = []
        llm_seen = False
        for item in sorted_items:
            if item.kind == ChatCapabilityKindEnum.llm_response:
                if llm_seen:
                    continue
                llm_seen = True
            final_items.append(item)
        return final_items


def parse_resource_selection(value: Any) -> ResourceSelection:
    if isinstance(value, ResourceSelection):
        return value
    payload = value or {}
    if not isinstance(payload, dict):
        return ResourceSelection()
    normalized_payload = {key: payload[key] for key in RESOURCE_SELECTION_TOP_LEVEL_KEYS if key in payload}
    if normalized_payload:
        return ResourceSelection.model_validate(normalized_payload)
    return ResourceSelection()


class ConversationSummary(StrictModel):
    id: int
    title: str
    status: str
    user_id: int | None = None
    active_turn_id: int | None = None
    head_turn_id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    default_resource_selection: ResourceSelection = Field(default_factory=ResourceSelection)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TurnSummary(StrictModel):
    id: int
    conversation_id: int
    seq: int
    status: str
    trigger: str
    input_root_data_id: int | None = None
    output_root_data_id: int | None = None
    root_step_id: int | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    error_message: str | None = None
    usage: UsagePayload = Field(default_factory=UsagePayload)
    resource_selection: ResourceSelection = Field(default_factory=ResourceSelection)


class StepSummary(StrictModel):
    id: int
    turn_id: int
    parent_step_id: int | None = None
    root_step_id: int | None = None
    kind: str
    name: str
    status: str
    sequence: int = 0
    attempt: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    metrics: StepMetricPayload = Field(default_factory=StepMetricPayload)
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatDataSchema(StrictModel, Generic[PayloadT]):
    id: int
    turn_id: int
    step_id: int | None = None
    kind: str
    payload_type: ChatPayloadTypeEnum
    role: str | None = None
    is_final: bool = False
    is_visible: bool = True
    payload: PayloadT
    refs: list[DataRefSchema] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatStepSchema(StrictModel, Generic[DetailT]):
    step: StepSummary
    detail: DetailT | None = None


class ChatEvent(StrictModel, Generic[PayloadT]):
    id: str
    session_id: str | None = None
    conversation_id: int | None = None
    turn_id: int | None = None
    seq: int = Field(ge=0)
    event: str
    ts: datetime
    payload: PayloadT


class AckPayload(StrictModel):
    command: ClientCommandEnum
    accepted: bool = True
    request_id: str | None = None


class AckRequest(StrictModel):
    seq: int = Field(ge=0)


class WarningPayload(StrictModel):
    message: str
    code: ChatWarningCodeEnum = ChatWarningCodeEnum.warning


class ErrorPayload(StrictModel):
    message: str
    code: ChatErrorCodeEnum = ChatErrorCodeEnum.chat_error
    retryable: bool = False


class TurnEventPayload(StrictModel):
    turn: TurnSummary


class StepEventPayload(StrictModel):
    step: StepSummary


class DataEventPayload(StrictModel, Generic[PayloadT]):
    data: ChatDataSchema[PayloadT]


class RetrievalListPayload(StrictModel):
    items: list[RetrievalBlock] = Field(default_factory=list)


class IntentResultPayload(StrictModel):
    type: Literal["intent_result"] = "intent_result"
    result: IntentRecognitionResult


class FunctionResultPayload(StrictModel):
    type: Literal["function_result"] = "function_result"
    tool_name: str = Field(min_length=1, max_length=64)
    title: str | None = Field(default=None, max_length=200)
    result_mode: FunctionCallResultModeEnum
    matched: bool = True
    summary: str | None = None
    content_text: str | None = None
    content_data: dict[str, Any] | None = None
    terminal: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromptContextPayload(StrictModel):
    type: Literal["prompt_context"] = "prompt_context"
    template_key: SystemPromptTemplateKeyEnum
    placeholders: list[SystemPromptPlaceholderEnum] = Field(default_factory=list)
    highlighted_capabilities: list[ChatCapabilityKindEnum] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)
    variable_overrides: dict[str, str] = Field(default_factory=dict)
    applied_capability_ids: list[str] = Field(default_factory=list)


class ConversationListItem(StrictModel):
    conversation: ConversationSummary
    latest_user_text: str | None = None
    latest_assistant_text: str | None = None


class ConversationTurnDetail(StrictModel):
    turn: TurnSummary
    input: MessageBundlePayload | None = None
    output: MessageBundlePayload | None = None
    events: list[dict[str, Any]] = Field(default_factory=list)


class ConversationTimeline(StrictModel):
    conversation: ConversationSummary
    turns: list[ConversationTurnDetail] = Field(default_factory=list)


class TurnStartRequest(StrictModel):
    conversation_id: int | None = None
    conversation_title: str | None = Field(default=None, min_length=1, max_length=255)
    request_id: str | None = None
    input: MessageBundlePayload
    resource_selection: ResourceSelection = Field(default_factory=ResourceSelection)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TurnStartAccepted(StrictModel):
    turn_id: int
    conversation: ConversationSummary


class TurnCancelRequest(StrictModel):
    turn_id: int
    reason: str | None = None


class TurnReplayRequest(StrictModel):
    turn_id: int
    last_seq: int = 0


class PingRequest(StrictModel):
    nonce: str | None = None


ClientPayload = TurnStartRequest | TurnCancelRequest | TurnReplayRequest | AckRequest | PingRequest


class ClientCommand(StrictModel, Generic[PayloadT]):
    command: ClientCommandEnum
    payload: PayloadT


class ChatHistoryItem(StrictModel):
    user_text: str
    assistant_text: str


class CapabilityResultDispositionEnum(StrEnum):
    context = ("context", "作为上下文")
    terminal = ("terminal", "作为终态输出")


class ChatContextItemTypeEnum(StrEnum):
    text = ("text", "文本")
    json = ("json", "JSON")
    retrieval = ("retrieval", "检索结果")


class CapabilityContextItem(StrictModel):
    capability_id: str = Field(min_length=1, max_length=128)
    capability_kind: ChatCapabilityKindEnum
    capability_name: str = Field(min_length=1, max_length=100)
    source: str = Field(min_length=1, max_length=64)
    disposition: CapabilityResultDispositionEnum = CapabilityResultDispositionEnum.context
    item_type: ChatContextItemTypeEnum
    title: str | None = Field(default=None, max_length=200)
    priority: int = Field(default=100, ge=0, le=1000)
    text: str | None = None
    data: dict[str, Any] | None = None
    retrievals: list[RetrievalBlock] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CapabilityTerminalOutput(StrictModel):
    capability_id: str = Field(min_length=1, max_length=128)
    capability_kind: ChatCapabilityKindEnum
    capability_name: str = Field(min_length=1, max_length=100)
    source: str = Field(min_length=1, max_length=64)
    payload: MessageBundlePayload
    metadata: dict[str, Any] = Field(default_factory=dict)


class IntentRecognitionResult(StrictModel):
    intent: str = Field(min_length=1, max_length=64)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    matched_keywords: list[str] = Field(default_factory=list)
    description: str = Field(default="", max_length=500)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FunctionExecutionSummary(StrictModel):
    tool_name: str = Field(min_length=1, max_length=64)
    title: str | None = Field(default=None, max_length=200)
    result_mode: FunctionCallResultModeEnum
    matched: bool = True
    summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatContextEnvelope(StrictModel):
    history: list[ChatHistoryItem] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)
    context_items: list[CapabilityContextItem] = Field(default_factory=list)
    prompt_context: PromptContextPayload | None = None
    intent_result: IntentRecognitionResult | None = None
    executed_functions: list[FunctionExecutionSummary] = Field(default_factory=list)
    terminal_output: CapabilityTerminalOutput | None = None

    def ordered_context_items(self) -> list[CapabilityContextItem]:
        return sorted(
            self.context_items,
            key=lambda item: (item.priority, item.capability_kind.value, item.capability_id),
        )

    @property
    def references(self) -> list[RetrievalBlock]:
        items: list[RetrievalBlock] = []
        for context_item in self.ordered_context_items():
            if context_item.item_type == ChatContextItemTypeEnum.retrieval:
                items.extend(context_item.retrievals)
        return items


class DemoPageState(StrictModel):
    websocket_path: str
    demo_title: str = "Chat Demo"
