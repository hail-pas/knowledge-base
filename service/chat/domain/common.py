from __future__ import annotations

from typing import Annotated, Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from core.types import StrEnum
from service.document.schema import (
    DocumentChunkList,
    DocumentGeneratedFaqList,
    DocumentList,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


PayloadT = TypeVar("PayloadT", bound=BaseModel)
ToolArgsT = TypeVar("ToolArgsT")


class ChatRoleEnum(StrEnum):
    user = ("user", "用户")
    assistant = ("assistant", "助手")


class EventNameEnum(StrEnum):
    session_ready = ("session.ready", "会话已就绪")
    ack = ("ack", "确认")
    warning = ("warning", "警告")
    error = ("error", "错误")
    ping = ("ping", "心跳")
    progress = ("progress", "进度")
    turn_started = ("turn.started", "turn已开始")
    turn_completed = ("turn.completed", "turn已完成")
    turn_failed = ("turn.failed", "turn失败")
    turn_canceled = ("turn.canceled", "turn已取消")
    step_started = ("step.started", "step已开始")
    step_completed = ("step.completed", "step已完成")
    step_failed = ("step.failed", "step失败")
    message_delta = ("message.delta", "消息增量")


class ChatWarningCodeEnum(StrEnum):
    warning = ("warning", "通用警告")
    knowledge_retrieval_no_hit = ("knowledge_retrieval_no_hit", "知识库检索无命中")
    knowledge_retrieval_unavailable = ("knowledge_retrieval_unavailable", "知识库检索未实际执行")
    tool_call_skipped = ("tool_call_skipped", "工具调用跳过")


class ChatErrorCodeEnum(StrEnum):
    chat_error = ("chat_error", "通用聊天错误")
    auth_failed = ("auth_failed", "鉴权失败")
    socket_init_failed = ("socket_init_failed", "WebSocket 初始化失败")
    unknown_command = ("unknown_command", "未知命令")
    command_error = ("command_error", "命令执行失败")
    invalid_command_payload = ("invalid_command_payload", "命令载荷非法")


class ChatActionKindEnum(StrEnum):
    system_prompt = ("system_prompt", "系统提示词")
    tool_call = ("tool_call", "工具调用")
    mcp_call = ("mcp_call", "MCP调用")
    sub_agent_call = ("sub_agent_call", "子代理调用")
    llm_response = ("llm_response", "模型响应")


ACTION_EXECUTION_ORDER: dict[ChatActionKindEnum, int] = {
    ChatActionKindEnum.system_prompt: 10,
    ChatActionKindEnum.tool_call: 30,
    ChatActionKindEnum.mcp_call: 40,
    ChatActionKindEnum.sub_agent_call: 50,
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
        "planner",
    },
)


class ClientCommandEnum(StrEnum):
    turn_start = ("turn.start", "启动turn")
    turn_cancel = ("turn.cancel", "取消turn")
    ping = ("ping", "心跳")


class ChatPayloadTypeEnum(StrEnum):
    message_bundle = ("message_bundle", "消息包")
    error = ("error", "错误")
    prompt_context = ("prompt_context", "提示词上下文")
    step_io = ("step_io", "步骤输入输出")
    tool_result = ("tool_result", "工具结果")
    mcp_result = ("mcp_result", "MCP结果")
    sub_agent_result = ("sub_agent_result", "子代理结果")


class ChatDataKindEnum(StrEnum):
    input = ("input", "输入")
    output = ("output", "输出")


class SelectionModeEnum(StrEnum):
    explicit = ("explicit", "显式")
    heuristic = ("heuristic", "启发式")
    llm = ("llm", "模型判定")
    hybrid = ("hybrid", "混合")


class CapabilityKindEnum(StrEnum):
    skill = ("skill", "流程型能力")
    extension = ("extension", "扩展能力")
    sub_agent = ("sub_agent", "子代理能力")


class CapabilityCategoryEnum(StrEnum):
    core = ("core", "核心能力")
    domain = ("domain", "领域能力")
    infra = ("infra", "基础设施能力")
    agent = ("agent", "代理能力")
    guarded = ("guarded", "受控能力")


class CapabilityRuntimeKindEnum(StrEnum):
    local_toolset = ("local_toolset", "本地工具集")
    mcp_toolset = ("mcp_toolset", "MCP工具集")
    agent_delegate = ("agent_delegate", "代理委派")
    agent_handoff = ("agent_handoff", "代理切换")


class AgentRoleEnum(StrEnum):
    orchestrator = ("orchestrator", "编排代理")
    specialist = ("specialist", "专家代理")


class AgentMountModeEnum(StrEnum):
    delegate = ("delegate", "委派")


class ProgressLevelEnum(StrEnum):
    info = ("info", "信息")
    warning = ("warning", "警告")
    error = ("error", "错误")


class ToolResultModeEnum(StrEnum):
    auto = ("auto", "自动")
    context = ("context", "作为上下文")
    terminal = ("terminal", "直接返回")


class CapabilityPlannerModeEnum(StrEnum):
    disabled = ("disabled", "关闭自动规划")
    llm = ("llm", "模型规划")


class SystemPromptTemplateKeyEnum(StrEnum):
    default = ("default", "默认模板")


class SystemPromptPlaceholderEnum(StrEnum):
    action_summary = ("action_summary", "执行摘要")
    tool_summary = ("tool_summary", "工具摘要")
    conversation_summary = ("conversation_summary", "会话摘要")
    instructions_summary = ("instructions_summary", "额外约束")
    context_policy = ("context_policy", "上下文策略")


DEFAULT_SYSTEM_PROMPT_PLACEHOLDERS: tuple[SystemPromptPlaceholderEnum, ...] = (
    SystemPromptPlaceholderEnum.action_summary,
    SystemPromptPlaceholderEnum.tool_summary,
    SystemPromptPlaceholderEnum.conversation_summary,
    SystemPromptPlaceholderEnum.instructions_summary,
    SystemPromptPlaceholderEnum.context_policy,
)


class ActionResultDispositionEnum(StrEnum):
    context = ("context", "作为上下文")
    terminal = ("terminal", "作为终态输出")


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
    data: JsonValue


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


class UsagePayload(StrictModel):
    requests: int = Field(default=0, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class LLMResponseConfig(StrictModel):
    llm_model_config_id: int | None = Field(default=None, ge=1)


class ToolSpec(StrictModel, Generic[ToolArgsT]):
    tool_name: str = Field(min_length=1, max_length=64)
    args: ToolArgsT | None = None
    result_mode: ToolResultModeEnum = ToolResultModeEnum.auto


PersistedToolSpec = ToolSpec[dict[str, JsonValue]]


class ToolCallConfig(StrictModel, Generic[ToolArgsT]):
    tools: list[ToolSpec[ToolArgsT]] = Field(min_length=1)
    selection_mode: SelectionModeEnum = SelectionModeEnum.explicit
    planner_model_config_id: int | None = Field(default=None, ge=1)
    max_selected_tools: int = Field(default=3, ge=1, le=8)
    fail_on_no_match: bool = False
    stop_after_terminal: bool = True


PersistedToolCallConfig = ToolCallConfig[dict[str, JsonValue]]


class MCPCallConfig(StrictModel):
    server_name: str = Field(min_length=1, max_length=128)
    tool_names: list[str] = Field(default_factory=list)
    timeout_ms: int = Field(default=30000, ge=1000, le=300000)


class SystemPromptConfig(StrictModel):
    template_key: SystemPromptTemplateKeyEnum = SystemPromptTemplateKeyEnum.default
    include_placeholders: list[SystemPromptPlaceholderEnum] = Field(default_factory=list)
    highlight_actions: list[ChatActionKindEnum] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)
    variable_overrides: dict[str, str] = Field(default_factory=dict)


__all__ = [
    "ACTION_EXECUTION_ORDER",
    "ActionResultDispositionEnum",
    "AgentMountModeEnum",
    "AgentRoleEnum",
    "CapabilityCategoryEnum",
    "CapabilityKindEnum",
    "CapabilityPlannerModeEnum",
    "CapabilityRuntimeKindEnum",
    "ChatActionKindEnum",
    "ChatDataKindEnum",
    "ChatErrorCodeEnum",
    "ChatPayloadTypeEnum",
    "ChatRoleEnum",
    "ChatWarningCodeEnum",
    "ClientCommandEnum",
    "ContentBlock",
    "DEFAULT_SYSTEM_PROMPT_PLACEHOLDERS",
    "ErrorBlock",
    "EventNameEnum",
    "FileBlock",
    "JsonBlock",
    "LLMResponseConfig",
    "MCPCallConfig",
    "MessageBundlePayload",
    "MessageDeltaPayload",
    "PayloadT",
    "ProgressLevelEnum",
    "RESOURCE_SELECTION_TOP_LEVEL_KEYS",
    "RetrievalBlock",
    "SelectionModeEnum",
    "StrictModel",
    "SystemPromptConfig",
    "SystemPromptPlaceholderEnum",
    "SystemPromptTemplateKeyEnum",
    "TextBlock",
    "ToolArgsT",
    "ToolCallConfig",
    "ToolResultModeEnum",
    "PersistedToolCallConfig",
    "PersistedToolSpec",
    "ToolSpec",
    "UsagePayload",
    "action_execution_order",
]
