from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Generic, Literal, TypeVar
from uuid import UUID

from pydantic import Field, JsonValue, TypeAdapter

from core.types import StrEnum
from service.chat.domain.common import (
    ActionResultDispositionEnum,
    ChatActionKindEnum,
    ChatDataKindEnum,
    ChatErrorCodeEnum,
    ChatPayloadTypeEnum,
    ChatRoleEnum,
    ChatWarningCodeEnum,
    ClientCommandEnum,
    PayloadT,
    ProgressLevelEnum,
    RetrievalBlock,
    StrictModel,
    SystemPromptPlaceholderEnum,
    SystemPromptTemplateKeyEnum,
    UsagePayload,
    MessageBundlePayload,
)
from service.chat.domain.resources import ResourceSelection
from service.chat.domain.errors import ChatPayloadError


class ConversationSummary(StrictModel):
    id: int
    agent_key: str = Field(default="orchestrator.default", min_length=1, max_length=128)
    title: str
    user_id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    default_resource_selection: ResourceSelection = Field(default_factory=ResourceSelection)


class TurnSummary(StrictModel):
    id: int
    conversation_id: int
    agent_key: str = Field(default="orchestrator.default", min_length=1, max_length=128)
    seq: int
    status: str
    trigger: str
    request_id: UUID | None = None
    input_root_data_id: int | None = None
    output_root_data_id: int | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    planner_mode: str | None = None
    planner_summary: str | None = None
    selected_capability_keys: list[str] = Field(default_factory=list)
    usage: UsagePayload = Field(default_factory=UsagePayload)
    resource_selection: ResourceSelection = Field(default_factory=ResourceSelection)


class StepSummary(StrictModel):
    id: int
    conversation_id: int
    turn_id: int
    parent_step_id: int | None = None
    kind: str
    capability_key: str | None = None
    operation_key: str | None = None
    name: str
    status: str
    sequence: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class StepIOPhaseEnum(StrEnum):
    input = ("input", "输入")
    output = ("output", "输出")


StepIODataT = TypeVar("StepIODataT")


class StepIOPayload(StrictModel, Generic[StepIODataT]):
    type: Literal["step_io"] = "step_io"
    phase: StepIOPhaseEnum
    action_id: str = Field(min_length=1, max_length=128)
    action_name: str = Field(min_length=1, max_length=100)
    action_kind: ChatActionKindEnum
    message: str = Field(min_length=1, max_length=2000)
    data: StepIODataT | None = None


ToolContentT = TypeVar("ToolContentT")


class ToolResultPayload(StrictModel, Generic[ToolContentT]):
    type: Literal["tool_result"] = "tool_result"
    tool_name: str = Field(min_length=1, max_length=64)
    title: str | None = Field(default=None, max_length=200)
    disposition: ActionResultDispositionEnum = ActionResultDispositionEnum.context
    summary: str | None = None
    content_text: str | None = None
    content: ToolContentT | None = None
    terminal: bool = False


MCPContentT = TypeVar("MCPContentT")


class MCPResultPayload(StrictModel, Generic[MCPContentT]):
    type: Literal["mcp_result"] = "mcp_result"
    server_name: str = Field(min_length=1, max_length=128)
    tool_name: str = Field(min_length=1, max_length=128)
    title: str | None = Field(default=None, max_length=200)
    disposition: ActionResultDispositionEnum = ActionResultDispositionEnum.context
    summary: str | None = None
    content_text: str | None = None
    content: MCPContentT | None = None
    terminal: bool = False


class SubAgentResultPayload(StrictModel):
    type: Literal["sub_agent_result"] = "sub_agent_result"
    agent_key: str | None = Field(default=None, min_length=1, max_length=128)
    agent_name: str | None = Field(default=None, max_length=128)
    output_contract: str | None = Field(default=None, max_length=255)
    terminal: bool = False
    summary: str | None = None
    content_text: str | None = None
    usage: UsagePayload | None = None


class PromptContextPayload(StrictModel):
    type: Literal["prompt_context"] = "prompt_context"
    template_key: SystemPromptTemplateKeyEnum
    placeholders: list[SystemPromptPlaceholderEnum] = Field(default_factory=list)
    highlighted_actions: list[ChatActionKindEnum] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)
    variable_overrides: dict[str, str] = Field(default_factory=dict)
    applied_action_ids: list[str] = Field(default_factory=list)


class WarningPayload(StrictModel):
    message: str
    code: ChatWarningCodeEnum = ChatWarningCodeEnum.warning


class ErrorPayload(StrictModel):
    type: Literal["error"] = "error"
    message: str
    code: ChatErrorCodeEnum = ChatErrorCodeEnum.chat_error
    retryable: bool = False


PersistedToolResultPayload = ToolResultPayload[JsonValue]
PersistedStepIOPayload = StepIOPayload[JsonValue]
PersistedMCPResultPayload = MCPResultPayload[JsonValue]


PersistedPayload = (
    MessageBundlePayload
    | ErrorPayload
    | PromptContextPayload
    | PersistedStepIOPayload
    | PersistedToolResultPayload
    | PersistedMCPResultPayload
    | SubAgentResultPayload
)


def validate_payload_for_type(
    payload_type: ChatPayloadTypeEnum | str,
    payload: Any,
) -> PersistedPayload:
    parsed_type = payload_type if isinstance(payload_type, ChatPayloadTypeEnum) else ChatPayloadTypeEnum(payload_type)
    model_map: dict[ChatPayloadTypeEnum, type[StrictModel]] = {
        ChatPayloadTypeEnum.message_bundle: MessageBundlePayload,
        ChatPayloadTypeEnum.error: ErrorPayload,
        ChatPayloadTypeEnum.prompt_context: PromptContextPayload,
        ChatPayloadTypeEnum.step_io: PersistedStepIOPayload,
        ChatPayloadTypeEnum.tool_result: PersistedToolResultPayload,
        ChatPayloadTypeEnum.mcp_result: PersistedMCPResultPayload,
        ChatPayloadTypeEnum.sub_agent_result: SubAgentResultPayload,
    }
    model_cls = model_map.get(parsed_type)
    if model_cls is None:
        raise ChatPayloadError(f"Unknown payload type: {parsed_type}", payload_type=str(parsed_type))
    return model_cls.model_validate(payload)


class ChatDataSchema(StrictModel, Generic[PayloadT]):
    id: int
    conversation_id: int
    turn_id: int
    step_id: int
    kind: ChatDataKindEnum
    payload_type: ChatPayloadTypeEnum
    payload: PayloadT


class ChatEvent(StrictModel, Generic[PayloadT]):
    id: str
    session_id: UUID | None = None
    conversation_id: int | None = None
    turn_id: int | None = None
    seq: int = Field(ge=0)
    event: str
    ts: datetime
    payload: PayloadT


class AckPayload(StrictModel):
    command: ClientCommandEnum
    accepted: bool = True
    request_id: UUID | None = None


ProgressDataT = TypeVar("ProgressDataT")


class ProgressPayload(StrictModel, Generic[ProgressDataT]):
    stage: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=2000)
    level: ProgressLevelEnum = ProgressLevelEnum.info
    data: ProgressDataT | None = None


class TurnEventPayload(StrictModel):
    turn: TurnSummary


class StepEventPayload(StrictModel):
    step: StepSummary
    data: ChatDataSchema[PersistedPayload] | None = None


class SessionReadyPayload(StrictModel):
    session_id: UUID
    default_agent_key: str = Field(min_length=1, max_length=128)
    available_agent_keys: list[str] = Field(default_factory=list)
    available_capability_keys: list[str] = Field(default_factory=list)


class ConversationStepTrace(StrictModel):
    step: StepSummary
    input: ChatDataSchema[PersistedPayload] | None = None
    output: ChatDataSchema[PersistedPayload] | None = None


class ConversationListItem(StrictModel):
    conversation: ConversationSummary
    latest_user_text: str | None = None
    latest_assistant_text: str | None = None


class ConversationTurnDetail(StrictModel):
    turn: TurnSummary
    input: MessageBundlePayload | None = None
    output: PersistedPayload | None = None
    steps: list[ConversationStepTrace] = Field(default_factory=list)


class ConversationTimeline(StrictModel):
    conversation: ConversationSummary
    turns: list[ConversationTurnDetail] = Field(default_factory=list)


class TurnStartRequest(StrictModel):
    conversation_id: int | None = None
    agent_key: str | None = Field(default=None, min_length=1, max_length=128)
    conversation_title: str | None = Field(default=None, min_length=1, max_length=255)
    request_id: UUID | None = None
    input: MessageBundlePayload
    resource_selection: ResourceSelection = Field(default_factory=ResourceSelection)

    def reuse_bound_conversation(self, conversation_id: int | None) -> TurnStartRequest:
        if conversation_id is None:
            return self
        if self.conversation_id is not None or "conversation_id" in self.model_fields_set:
            return self
        return self.model_copy(update={"conversation_id": conversation_id})


class TurnStartAccepted(StrictModel):
    turn_id: int
    conversation: ConversationSummary


class TurnCancelRequest(StrictModel):
    turn_id: int
    reason: str | None = None


class PingRequest(StrictModel):
    nonce: str | None = None


ClientPayload = TurnStartRequest | TurnCancelRequest | PingRequest


class ClientCommand(StrictModel, Generic[PayloadT]):
    command: ClientCommandEnum
    payload: PayloadT


class TurnStartCommand(ClientCommand[TurnStartRequest]):
    command: Literal["turn.start"] = "turn.start"
    payload: TurnStartRequest


class TurnCancelCommand(ClientCommand[TurnCancelRequest]):
    command: Literal["turn.cancel"] = "turn.cancel"
    payload: TurnCancelRequest


class PingCommand(ClientCommand[PingRequest]):
    command: Literal["ping"] = "ping"
    payload: PingRequest


ValidatedClientCommand = Annotated[
    TurnStartCommand | TurnCancelCommand | PingCommand,
    Field(discriminator="command"),
]

CLIENT_COMMAND_ADAPTER = TypeAdapter(ValidatedClientCommand)


class ChatHistoryItem(StrictModel):
    user_text: str
    assistant_text: str


class ChatContextItemTypeEnum(StrEnum):
    text = ("text", "文本")
    json = ("json", "JSON")
    retrieval = ("retrieval", "检索结果")


class BaseActionContextItem(StrictModel):
    action_id: str = Field(min_length=1, max_length=128)
    action_kind: ChatActionKindEnum
    action_name: str = Field(min_length=1, max_length=100)
    source: str = Field(min_length=1, max_length=64)
    disposition: ActionResultDispositionEnum = ActionResultDispositionEnum.context
    title: str | None = Field(default=None, max_length=200)
    priority: int = Field(default=100, ge=0, le=1000)


class TextContextItem(BaseActionContextItem):
    item_type: Literal[ChatContextItemTypeEnum.text] = ChatContextItemTypeEnum.text
    text: str = Field(min_length=1)


ContextDataT = TypeVar("ContextDataT")


class JsonContextItem(BaseActionContextItem, Generic[ContextDataT]):
    item_type: Literal[ChatContextItemTypeEnum.json] = ChatContextItemTypeEnum.json
    data: ContextDataT


class RetrievalContextItem(BaseActionContextItem):
    item_type: Literal[ChatContextItemTypeEnum.retrieval] = ChatContextItemTypeEnum.retrieval
    retrievals: list[RetrievalBlock] = Field(min_length=1)


PersistedJsonContextItem = JsonContextItem[JsonValue]


ActionContextItem = Annotated[
    TextContextItem | PersistedJsonContextItem | RetrievalContextItem,
    Field(discriminator="item_type"),
]


class ActionTerminalOutput(StrictModel):
    action_id: str = Field(min_length=1, max_length=128)
    action_kind: ChatActionKindEnum
    action_name: str = Field(min_length=1, max_length=100)
    source: str = Field(min_length=1, max_length=64)
    data_id: int | None = None
    payload: MessageBundlePayload


class ToolExecutionSummary(StrictModel):
    tool_name: str = Field(min_length=1, max_length=64)
    title: str | None = Field(default=None, max_length=200)
    disposition: ActionResultDispositionEnum = ActionResultDispositionEnum.context
    summary: str | None = None


class ChatContextEnvelope(StrictModel):
    history: list[ChatHistoryItem] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)
    context_items: list[ActionContextItem] = Field(default_factory=list)
    prompt_context: PromptContextPayload | None = None
    executed_tools: list[ToolExecutionSummary] = Field(default_factory=list)
    terminal_output: ActionTerminalOutput | None = None

    def ordered_context_items(self) -> list[ActionContextItem]:
        return sorted(
            self.context_items,
            key=lambda item: (item.priority, item.action_kind.value, item.action_id),
        )


def parse_client_command(value: Any) -> ValidatedClientCommand:
    return CLIENT_COMMAND_ADAPTER.validate_python(value)


__all__ = [
    "AckPayload",
    "ActionContextItem",
    "ActionTerminalOutput",
    "BaseActionContextItem",
    "CLIENT_COMMAND_ADAPTER",
    "ChatContextEnvelope",
    "ChatContextItemTypeEnum",
    "ChatDataSchema",
    "ChatEvent",
    "ChatHistoryItem",
    "ClientCommand",
    "ClientPayload",
    "ConversationListItem",
    "ConversationStepTrace",
    "ConversationSummary",
    "ConversationTimeline",
    "ConversationTurnDetail",
    "ErrorPayload",
    "JsonContextItem",
    "MCPResultPayload",
    "PersistedJsonContextItem",
    "PersistedMCPResultPayload",
    "PersistedPayload",
    "PersistedStepIOPayload",
    "PingCommand",
    "PersistedToolResultPayload",
    "PingRequest",
    "ProgressPayload",
    "PromptContextPayload",
    "SessionReadyPayload",
    "StepEventPayload",
    "StepIOPayload",
    "StepIODataT",
    "StepIOPhaseEnum",
    "StepSummary",
    "SubAgentResultPayload",
    "TextContextItem",
    "ToolExecutionSummary",
    "ToolResultPayload",
    "TurnCancelCommand",
    "TurnCancelRequest",
    "TurnEventPayload",
    "TurnStartAccepted",
    "TurnStartCommand",
    "TurnStartRequest",
    "TurnSummary",
    "ValidatedClientCommand",
    "WarningPayload",
    "ProgressDataT",
    "MCPContentT",
    "RetrievalContextItem",
    "parse_client_command",
    "validate_payload_for_type",
]
