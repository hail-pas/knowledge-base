from __future__ import annotations

from typing import Any, Generic, Literal, TypeVar, Annotated
from datetime import datetime

from pydantic import Field, BaseModel, ConfigDict, model_validator

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
    knowledge_retrieval_no_hit = ("knowledge_retrieval_no_hit", "知识库检索无命中")
    knowledge_retrieval_unavailable = ("knowledge_retrieval_unavailable", "知识库检索未实际执行")
    function_call_skipped = ("function_call_skipped", "函数调用跳过")
    tool_call_skipped = ("tool_call_skipped", "工具调用跳过")


class ChatErrorCodeEnum(StrEnum):
    chat_error = ("chat_error", "通用聊天错误")
    auth_failed = ("auth_failed", "鉴权失败")
    socket_init_failed = ("socket_init_failed", "WebSocket 初始化失败")
    socket_session_error = ("socket_session_error", "WebSocket 会话初始化失败")
    unknown_command = ("unknown_command", "未知命令")
    command_error = ("command_error", "命令执行失败")


class ChatActionKindEnum(StrEnum):
    intent_detection = ("intent_detection", "意图识别")
    system_prompt = ("system_prompt", "系统提示词")
    knowledge_retrieval = ("knowledge_retrieval", "知识库检索")
    function_call = ("function_call", "函数调用")
    tool_call = ("tool_call", "工具调用")
    mcp_call = ("mcp_call", "MCP调用")
    sub_agent_call = ("sub_agent_call", "子代理调用")
    llm_response = ("llm_response", "模型响应")


ACTION_EXECUTION_ORDER: dict[ChatActionKindEnum, int] = {
    ChatActionKindEnum.system_prompt: 10,
    ChatActionKindEnum.intent_detection: 20,
    ChatActionKindEnum.knowledge_retrieval: 30,
    ChatActionKindEnum.function_call: 40,
    ChatActionKindEnum.tool_call: 50,
    ChatActionKindEnum.mcp_call: 60,
    ChatActionKindEnum.sub_agent_call: 70,
    ChatActionKindEnum.llm_response: 90,
}


def action_execution_order(kind: ChatActionKindEnum) -> int:
    return ACTION_EXECUTION_ORDER.get(kind, 999)


RESOURCE_SELECTION_TOP_LEVEL_KEYS = frozenset(
    {
        "use_system_defaults",
        "use_conversation_defaults",
        "capabilities",
        "actions",
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
    step_io = ("step_io", "步骤输入输出")
    retrieval_hit_list = ("retrieval_hit_list", "检索命中列表")
    extension_event = ("extension_event", "扩展中间事件")
    tool_result = ("tool_result", "工具结果")
    mcp_result = ("mcp_result", "MCP结果")
    capability_plan = ("capability_plan", "能力规划结果")


class ResponseModeEnum(StrEnum):
    text = ("text", "文本输出")
    json = ("json", "JSON输出")


class ToolExecutionPolicyEnum(StrEnum):
    stub = ("stub", "占位执行")
    optional = ("optional", "可选执行")
    required = ("required", "必须执行")


class SelectionModeEnum(StrEnum):
    heuristic = ("heuristic", "启发式")
    llm = ("llm", "模型判定")
    hybrid = ("hybrid", "混合")


class CapabilityKindEnum(StrEnum):
    skill = ("skill", "流程型能力")
    extension = ("extension", "扩展能力")
    sub_agent = ("sub_agent", "子代理能力")


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


class SubAgentCallConfig(StrictModel):
    llm_model_config_id: int | None = Field(default=None, ge=1)
    system_prompt: str = Field(min_length=1, max_length=8000)
    instructions: list[str] = Field(default_factory=list)
    actions: list[ResourceAction] = Field(default_factory=list)


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
    action_summary = ("action_summary", "执行摘要")
    intent_summary = ("intent_summary", "意图摘要")
    function_summary = ("function_summary", "函数摘要")
    conversation_summary = ("conversation_summary", "会话摘要")
    instructions_summary = ("instructions_summary", "额外约束")
    context_policy = ("context_policy", "上下文策略")


DEFAULT_SYSTEM_PROMPT_PLACEHOLDERS: tuple[SystemPromptPlaceholderEnum, ...] = (
    SystemPromptPlaceholderEnum.action_summary,
    SystemPromptPlaceholderEnum.intent_summary,
    SystemPromptPlaceholderEnum.function_summary,
    SystemPromptPlaceholderEnum.conversation_summary,
    SystemPromptPlaceholderEnum.instructions_summary,
    SystemPromptPlaceholderEnum.context_policy,
)


class SystemPromptConfig(StrictModel):
    template_key: SystemPromptTemplateKeyEnum = SystemPromptTemplateKeyEnum.default
    include_placeholders: list[SystemPromptPlaceholderEnum] = Field(default_factory=list)
    highlight_actions: list[ChatActionKindEnum] = Field(default_factory=list)
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
    selection_mode: SelectionModeEnum = SelectionModeEnum.heuristic
    planner_model_config_id: int | None = Field(default=None, ge=1)
    max_selected_tools: int = Field(default=3, ge=1, le=8)
    fail_on_no_match: bool = False
    stop_after_terminal: bool = True


class BaseResourceAction(StrictModel):
    action_id: str | None = Field(default=None, min_length=1, max_length=128)
    name: str | None = Field(default=None, min_length=1, max_length=100)
    source: str | None = Field(default=None, min_length=1, max_length=64)
    enabled: bool = True
    priority: int = Field(default=100, ge=0, le=1000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeRetrievalAction(BaseResourceAction):
    kind: Literal[ChatActionKindEnum.knowledge_retrieval] = ChatActionKindEnum.knowledge_retrieval
    config: KnowledgeRetrievalConfig


class IntentDetectionAction(BaseResourceAction):
    kind: Literal[ChatActionKindEnum.intent_detection] = ChatActionKindEnum.intent_detection
    config: IntentDetectionConfig


class SystemPromptAction(BaseResourceAction):
    kind: Literal[ChatActionKindEnum.system_prompt] = ChatActionKindEnum.system_prompt
    config: SystemPromptConfig


class FunctionCallAction(BaseResourceAction):
    kind: Literal[ChatActionKindEnum.function_call] = ChatActionKindEnum.function_call
    config: FunctionCallConfig


class ToolCallAction(BaseResourceAction):
    kind: Literal[ChatActionKindEnum.tool_call] = ChatActionKindEnum.tool_call
    config: ToolCallConfig = Field(default_factory=ToolCallConfig)


class MCPCallAction(BaseResourceAction):
    kind: Literal[ChatActionKindEnum.mcp_call] = ChatActionKindEnum.mcp_call
    config: MCPCallConfig


class SubAgentCallAction(BaseResourceAction):
    kind: Literal[ChatActionKindEnum.sub_agent_call] = ChatActionKindEnum.sub_agent_call
    config: SubAgentCallConfig


class LLMResponseAction(BaseResourceAction):
    kind: Literal[ChatActionKindEnum.llm_response] = ChatActionKindEnum.llm_response
    config: LLMResponseConfig = Field(default_factory=LLMResponseConfig)


ResourceAction = Annotated[
    IntentDetectionAction
    | SystemPromptAction
    | KnowledgeRetrievalAction
    | FunctionCallAction
    | ToolCallAction
    | MCPCallAction
    | SubAgentCallAction
    | LLMResponseAction,
    Field(discriminator="kind"),
]


class CapabilitySelection(StrictModel):
    capability_id: int | None = Field(default=None, ge=1)
    capability_key: str | None = Field(default=None, min_length=1, max_length=128)
    kind: CapabilityKindEnum | None = None
    enabled: bool = True
    required: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_reference(self) -> CapabilitySelection:
        if self.capability_id is None and self.capability_key is None:
            raise ValueError("capability_id 或 capability_key 至少需要一个")
        return self


class ResourceSelection(StrictModel):
    use_system_defaults: bool = True
    use_conversation_defaults: bool = True
    capabilities: list[CapabilitySelection] = Field(default_factory=list)
    actions: list[ResourceAction] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def actions_by_kind(self) -> dict[ChatActionKindEnum, list[ResourceAction]]:
        grouped: dict[ChatActionKindEnum, list[ResourceAction]] = {}
        for item in self.actions:
            if not item.enabled:
                continue
            grouped.setdefault(item.kind, []).append(item)
        return grouped

    def normalized_actions(self) -> list[ResourceAction]:
        normalized: list[ResourceAction] = []
        seen_ids: set[str] = set()
        has_llm = False
        for item in self.actions:
            if not item.enabled:
                continue
            if item.action_id and item.action_id in seen_ids:
                continue
            if item.action_id:
                seen_ids.add(item.action_id)
            normalized.append(item)
            if item.kind == ChatActionKindEnum.llm_response:
                has_llm = True

        if not has_llm:
            normalized.append(
                LLMResponseAction(
                    action_id="builtin:llm_response",
                    name="llm_response",
                    source="builtin",
                    priority=100,
                ),
            )

        sorted_items = sorted(
            normalized,
            key=lambda item: (
                item.priority,
                action_execution_order(item.kind),
                item.action_id or item.name or "",
            ),
        )
        final_items: list[ResourceAction] = []
        llm_seen = False
        for item in sorted_items:
            if item.kind == ChatActionKindEnum.llm_response:
                if llm_seen:
                    continue
                llm_seen = True
            final_items.append(item)
        return final_items

    def normalized_capabilities(self) -> list[CapabilitySelection]:
        normalized: list[CapabilitySelection] = []
        seen_ids: set[int] = set()
        seen_keys: set[str] = set()
        for item in self.capabilities:
            if not item.enabled:
                continue
            if item.capability_id is not None:
                if item.capability_id in seen_ids:
                    continue
                seen_ids.add(item.capability_id)
            if item.capability_key is not None:
                lowered_key = item.capability_key.casefold()
                if lowered_key in seen_keys:
                    continue
                seen_keys.add(lowered_key)
            normalized.append(item)
        return normalized


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
    input_data_ids: list[int] = Field(default_factory=list)
    output_data_ids: list[int] = Field(default_factory=list)
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


class StepIOPhaseEnum(StrEnum):
    input = ("input", "输入")
    output = ("output", "输出")


class StepIOPayload(StrictModel):
    type: Literal["step_io"] = "step_io"
    phase: StepIOPhaseEnum
    action_id: str = Field(min_length=1, max_length=128)
    action_name: str = Field(min_length=1, max_length=100)
    action_kind: ChatActionKindEnum
    message: str = Field(min_length=1, max_length=2000)
    data: dict[str, Any] = Field(default_factory=dict)


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


class ExtensionEventLevelEnum(StrEnum):
    info = ("info", "信息")
    warning = ("warning", "警告")
    error = ("error", "错误")


class ExtensionEventStageEnum(StrEnum):
    started = ("started", "开始")
    preparing = ("preparing", "准备中")
    rewriting = ("rewriting", "改写中")
    retrieving = ("retrieving", "检索中")
    reranking = ("reranking", "重排中")
    completed = ("completed", "完成")
    no_hit = ("no_hit", "无命中")
    unavailable = ("unavailable", "未实际执行")
    collection_start = ("collection_start", "集合检索开始")
    collection_completed = ("collection_completed", "集合检索完成")
    collection_missing = ("collection_missing", "集合不存在")
    collection_inaccessible = ("collection_inaccessible", "集合不可访问")
    collection_failed = ("collection_failed", "集合检索失败")


class ExtensionEventPayload(StrictModel):
    type: Literal["extension_event"] = "extension_event"
    extension_key: str | None = Field(default=None, min_length=1, max_length=128)
    capability_id: int | None = Field(default=None, ge=1)
    action_id: str = Field(min_length=1, max_length=128)
    action_name: str = Field(min_length=1, max_length=100)
    stage: ExtensionEventStageEnum
    level: ExtensionEventLevelEnum = ExtensionEventLevelEnum.info
    message: str = Field(min_length=1, max_length=2000)
    data: dict[str, Any] = Field(default_factory=dict)


class PromptContextPayload(StrictModel):
    type: Literal["prompt_context"] = "prompt_context"
    template_key: SystemPromptTemplateKeyEnum
    placeholders: list[SystemPromptPlaceholderEnum] = Field(default_factory=list)
    highlighted_actions: list[ChatActionKindEnum] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)
    variable_overrides: dict[str, str] = Field(default_factory=dict)
    applied_action_ids: list[str] = Field(default_factory=list)


class CapabilityPlanCandidate(StrictModel):
    capability_id: int | None = Field(default=None, ge=1)
    capability_key: str = Field(min_length=1, max_length=128)
    capability_kind: CapabilityKindEnum
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    selected: bool = False
    reasons: list[str] = Field(default_factory=list)


class CapabilityPlanPayload(StrictModel):
    type: Literal["capability_plan"] = "capability_plan"
    mode: str = Field(min_length=1, max_length=32)
    summary: str = Field(default="", max_length=1000)
    selected_capability_ids: list[int] = Field(default_factory=list)
    selected_capability_keys: list[str] = Field(default_factory=list)
    candidates: list[CapabilityPlanCandidate] = Field(default_factory=list)


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


class ActionResultDispositionEnum(StrEnum):
    context = ("context", "作为上下文")
    terminal = ("terminal", "作为终态输出")


class ChatContextItemTypeEnum(StrEnum):
    text = ("text", "文本")
    json = ("json", "JSON")
    retrieval = ("retrieval", "检索结果")


class ActionContextItem(StrictModel):
    action_id: str = Field(min_length=1, max_length=128)
    action_kind: ChatActionKindEnum
    action_name: str = Field(min_length=1, max_length=100)
    source: str = Field(min_length=1, max_length=64)
    disposition: ActionResultDispositionEnum = ActionResultDispositionEnum.context
    item_type: ChatContextItemTypeEnum
    title: str | None = Field(default=None, max_length=200)
    priority: int = Field(default=100, ge=0, le=1000)
    text: str | None = None
    data: dict[str, Any] | None = None
    retrievals: list[RetrievalBlock] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActionTerminalOutput(StrictModel):
    action_id: str = Field(min_length=1, max_length=128)
    action_kind: ChatActionKindEnum
    action_name: str = Field(min_length=1, max_length=100)
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
    context_items: list[ActionContextItem] = Field(default_factory=list)
    prompt_context: PromptContextPayload | None = None
    intent_result: IntentRecognitionResult | None = None
    executed_functions: list[FunctionExecutionSummary] = Field(default_factory=list)
    terminal_output: ActionTerminalOutput | None = None

    def ordered_context_items(self) -> list[ActionContextItem]:
        return sorted(
            self.context_items,
            key=lambda item: (item.priority, item.action_kind.value, item.action_id),
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


SkillPlanCandidate = CapabilityPlanCandidate
SkillPlanPayload = CapabilityPlanPayload


SubAgentCallConfig.model_rebuild()
