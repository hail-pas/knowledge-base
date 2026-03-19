from __future__ import annotations

from typing import Any
from dataclasses import field, dataclass

from service.chat.domain.schema import (
    DEFAULT_SYSTEM_PROMPT_PLACEHOLDERS,
    TurnStartRequest,
    ResourceSelection,
    ChatActionKindEnum,
    SystemPromptConfig,
    ConversationSummary,
    PromptContextPayload,
    SystemPromptPlaceholderEnum,
    SystemPromptTemplateKeyEnum,
)
from service.chat.runtime.context import TurnArtifacts
from service.chat.execution.registry import ExecutionAction


@dataclass(slots=True)
class SystemPromptState:
    template_key: SystemPromptTemplateKeyEnum = SystemPromptTemplateKeyEnum.default
    include_placeholders: list[SystemPromptPlaceholderEnum] = field(default_factory=list)
    highlight_actions: list[ChatActionKindEnum] = field(default_factory=list)
    instructions: list[str] = field(default_factory=list)
    variable_overrides: dict[str, str] = field(default_factory=dict)
    applied_action_ids: list[str] = field(default_factory=list)

    def merge(self, action: ExecutionAction, config: SystemPromptConfig) -> None:
        self.template_key = config.template_key
        self._extend_unique(self.include_placeholders, list(config.include_placeholders))
        self._extend_unique(self.highlight_actions, list(config.highlight_actions))
        self._extend_unique(
            self.instructions,
            [instruction.strip() for instruction in config.instructions if instruction.strip()],
        )
        self.variable_overrides.update(
            {
                key.strip(): value.strip()
                for key, value in config.variable_overrides.items()
                if key.strip() and value.strip()
            },
        )
        if action.action_id not in self.applied_action_ids:
            self.applied_action_ids.append(action.action_id)

    def selected_placeholders(self) -> list[SystemPromptPlaceholderEnum]:
        return list(self.include_placeholders or DEFAULT_SYSTEM_PROMPT_PLACEHOLDERS)

    def to_payload(self) -> PromptContextPayload:
        return PromptContextPayload(
            template_key=self.template_key,
            placeholders=self.selected_placeholders(),
            highlighted_actions=list(self.highlight_actions),
            instructions=list(self.instructions),
            variable_overrides=dict(self.variable_overrides),
            applied_action_ids=list(self.applied_action_ids),
        )

    def _extend_unique(self, target: list[Any], items: list[Any]) -> None:
        for item in items:
            if item not in target:
                target.append(item)


@dataclass(slots=True)
class ChatSessionContext:
    account_id: int | None
    is_staff: bool
    ws_session_id: int | None
    ws_public_session_id: str | None
    conversation: ConversationSummary
    turn_request: TurnStartRequest
    resolved_selection: ResourceSelection
    resolved_actions: list[ExecutionAction]
    artifacts: TurnArtifacts = field(default_factory=TurnArtifacts)
    prompt_state: SystemPromptState = field(default_factory=SystemPromptState)
    state: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sync_prompt_context()

    @property
    def conversation_id(self) -> int:
        return self.conversation.id

    @property
    def conversation_title(self) -> str:
        return self.conversation.title

    @property
    def request_id(self) -> str | None:
        return self.turn_request.request_id

    @property
    def query(self) -> str:
        return self.turn_request.input.text

    def sync_prompt_context(self) -> None:
        self.artifacts.set_prompt_context(self.prompt_state.to_payload())

    def apply_system_prompt_config(self, action: ExecutionAction, config: SystemPromptConfig) -> None:
        self.prompt_state.merge(action, config)
        self.sync_prompt_context()

    def set_state(self, key: str, value: Any) -> None:
        self.state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)
