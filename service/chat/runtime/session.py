from __future__ import annotations

from typing import Any
from uuid import UUID
from dataclasses import field, dataclass

from service.chat.agent.schema import AgentProfileSummary
from service.chat.domain.schema import (
    ChatRequestContext,
    DEFAULT_SYSTEM_PROMPT_PLACEHOLDERS,
    TurnStartRequest,
    ResourceSelection,
    ChatActionKindEnum,
    SystemPromptConfig,
    ConversationSummary,
    PromptContextPayload,
    SystemPromptPlaceholderEnum,
    SystemPromptTemplateKeyEnum,
    MessageBundlePayload,
)
from service.chat.runtime.context import TurnArtifacts
from service.chat.runtime.planning import RuntimeCapabilityDescriptor, RuntimeExecutionPlan
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

    def copy(self) -> SystemPromptState:
        return SystemPromptState(
            template_key=self.template_key,
            include_placeholders=list(self.include_placeholders),
            highlight_actions=list(self.highlight_actions),
            instructions=list(self.instructions),
            variable_overrides=dict(self.variable_overrides),
            applied_action_ids=list(self.applied_action_ids),
        )

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
    request_context: ChatRequestContext
    conversation: ConversationSummary
    turn_request: TurnStartRequest
    resolved_selection: ResourceSelection
    resolved_actions: list[ExecutionAction]
    agent: AgentProfileSummary | None = None
    execution_plan: RuntimeExecutionPlan | None = None
    artifacts: TurnArtifacts = field(default_factory=TurnArtifacts)
    prompt_state: SystemPromptState = field(default_factory=SystemPromptState)
    state: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sync_prompt_context()

    @property
    def account_id(self) -> int | None:
        return self.request_context.account_id

    @property
    def is_staff(self) -> bool:
        return self.request_context.is_staff

    @property
    def session_id(self) -> UUID:
        return self.request_context.require_session_id()

    @property
    def conversation_id(self) -> int:
        return self.conversation.id

    @property
    def conversation_title(self) -> str:
        return self.conversation.title

    @property
    def request_id(self) -> str | None:
        return str(self.turn_request.request_id) if self.turn_request.request_id is not None else None

    @property
    def query(self) -> str:
        return self.turn_request.input.text

    @property
    def selected_capabilities(self) -> list[RuntimeCapabilityDescriptor]:
        return list(self.execution_plan.selected_capabilities) if self.execution_plan is not None else []

    @property
    def selected_capability_keys(self) -> list[str]:
        return list(self.execution_plan.selected_capability_keys) if self.execution_plan is not None else []

    @property
    def planner_summary(self) -> str | None:
        if self.execution_plan is None:
            return None
        return self.execution_plan.summary or None

    def sync_prompt_context(self) -> None:
        self.artifacts.set_prompt_context(self.prompt_state.to_payload())

    def apply_system_prompt_config(self, action: ExecutionAction, config: SystemPromptConfig) -> None:
        self.prompt_state.merge(action, config)
        self.sync_prompt_context()

    def set_state(self, key: str, value: Any) -> None:
        self.state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    @property
    def active_step_ids(self) -> list[int]:
        value = self.state.get("active_step_ids", [])
        return [int(item) for item in value] if isinstance(value, list) else []

    def track_active_step(self, step_id: int) -> None:
        step_ids = self.active_step_ids
        if step_id not in step_ids:
            step_ids.append(step_id)
        self.state["active_step_ids"] = step_ids

    def finish_active_step(self, step_id: int) -> None:
        self.state["active_step_ids"] = [item for item in self.active_step_ids if item != step_id]

    def copy_state_fields_from(
        self,
        source: ChatSessionContext,
        *,
        fields: list[str],
    ) -> None:
        for field_name in fields:
            if not isinstance(field_name, str):
                continue
            normalized_name = field_name.strip()
            if not normalized_name or normalized_name not in source.state:
                continue
            self.state[normalized_name] = source.state[normalized_name]

    def configure_llm_execution(
        self,
        *,
        system_prompt_prefix: str | None = None,
        extra_instructions: list[str] | None = None,
        include_history: bool | None = None,
    ) -> None:
        if system_prompt_prefix is not None:
            self.state["llm_system_prompt_prefix"] = system_prompt_prefix.strip()
        if extra_instructions is not None:
            self.state["llm_extra_instructions"] = [
                item.strip()
                for item in extra_instructions
                if isinstance(item, str) and item.strip()
            ]
        if include_history is not None:
            self.state["llm_include_history"] = include_history

    @property
    def llm_system_prompt_prefix(self) -> str | None:
        value = self.get_state("llm_system_prompt_prefix")
        return value if isinstance(value, str) and value.strip() else None

    @property
    def llm_extra_instructions(self) -> list[str]:
        value = self.get_state("llm_extra_instructions", [])
        return [item for item in value if isinstance(item, str) and item.strip()] if isinstance(value, list) else []

    @property
    def include_history(self) -> bool:
        value = self.get_state("llm_include_history")
        return value if isinstance(value, bool) else True

    def derive(
        self,
        *,
        turn_request: TurnStartRequest | None = None,
        resolved_selection: ResourceSelection | None = None,
        resolved_actions: list[ExecutionAction] | None = None,
        input_payload: MessageBundlePayload | None = None,
        inherit_prompt_state: bool = True,
    ) -> ChatSessionContext:
        next_turn_request = turn_request or self.turn_request
        if input_payload is not None:
            next_turn_request = next_turn_request.model_copy(update={"input": input_payload})
        derived = ChatSessionContext(
            request_context=self.request_context.with_conversation(self.conversation.id),
            conversation=self.conversation,
            turn_request=next_turn_request,
            resolved_selection=resolved_selection if resolved_selection is not None else self.resolved_selection,
            resolved_actions=resolved_actions if resolved_actions is not None else self.resolved_actions,
            agent=self.agent,
            execution_plan=self.execution_plan,
        )
        if inherit_prompt_state:
            derived.prompt_state = self.prompt_state.copy()
            derived.sync_prompt_context()
        return derived
