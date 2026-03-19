from __future__ import annotations

from typing import Any, Iterable
from dataclasses import field, dataclass

from pydantic import BaseModel, TypeAdapter

from ext.ext_tortoise.enums import ChatStepKindEnum
from service.chat.domain.schema import (
    MCPCallConfig,
    ResourceAction,
    ToolCallConfig,
    LLMResponseConfig,
    ResourceSelection,
    ChatActionKindEnum,
    FunctionCallConfig,
    SubAgentCallConfig,
    SystemPromptConfig,
    IntentDetectionConfig,
    KnowledgeRetrievalConfig,
)

_RESOURCE_ACTION_ADAPTER = TypeAdapter(ResourceAction)


@dataclass(slots=True)
class ExecutionActionDefinition:
    kind: ChatActionKindEnum
    step_kind: ChatStepKindEnum
    default_name: str
    config_model: type[BaseModel]


@dataclass(slots=True)
class ExecutionAction:
    action_id: str
    kind: ChatActionKindEnum
    step_kind: ChatStepKindEnum
    name: str
    config: BaseModel
    priority: int
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ExecutionActionRegistry:
    def __init__(self) -> None:
        self._definitions: dict[ChatActionKindEnum, ExecutionActionDefinition] = {}

    def register(
        self,
        kind: ChatActionKindEnum,
        *,
        step_kind: ChatStepKindEnum,
        default_name: str,
        config_model: type[BaseModel],
    ) -> None:
        self._definitions[kind] = ExecutionActionDefinition(
            kind=kind,
            step_kind=step_kind,
            default_name=default_name,
            config_model=config_model,
        )

    def parse_kind(self, kind: ChatActionKindEnum | str) -> ChatActionKindEnum:
        return kind if isinstance(kind, ChatActionKindEnum) else ChatActionKindEnum(kind)

    def get_definition(self, kind: ChatActionKindEnum | str) -> ExecutionActionDefinition:
        parsed_kind = self.parse_kind(kind)
        definition = self._definitions.get(parsed_kind)
        if definition is None:
            raise ValueError(f"Unsupported action kind: {parsed_kind}")
        return definition

    def parse_config(self, kind: ChatActionKindEnum | str, config: BaseModel | dict[str, Any]) -> BaseModel:
        definition = self.get_definition(kind)
        payload = config.model_dump(mode="json") if isinstance(config, BaseModel) else config
        return definition.config_model.model_validate(payload)

    def build_action(
        self,
        kind: ChatActionKindEnum | str,
        *,
        config: BaseModel | dict[str, Any],
        action_id: str | None = None,
        name: str | None = None,
        source: str | None = None,
        enabled: bool = True,
        priority: int = 100,
        metadata: dict[str, Any] | None = None,
    ) -> ResourceAction:
        parsed_kind = self.parse_kind(kind)
        validated_config = self.parse_config(parsed_kind, config)
        return _RESOURCE_ACTION_ADAPTER.validate_python(
            {
                "kind": parsed_kind.value,
                "action_id": action_id,
                "name": name or self.get_definition(parsed_kind).default_name,
                "source": source,
                "enabled": enabled,
                "priority": priority,
                "metadata": metadata or {},
                "config": validated_config.model_dump(mode="json"),
            },
        )

    def assign_inline_action_ids(
        self,
        actions: Iterable[ResourceAction],
        *,
        source: str,
        prefix: str,
    ) -> list[ResourceAction]:
        assigned: list[ResourceAction] = []
        for index, action in enumerate(actions, start=1):
            definition = self.get_definition(action.kind)
            assigned.append(
                action.model_copy(
                    update={
                        "action_id": action.action_id or f"{prefix}:{index}",
                        "name": action.name or str(action.metadata.get("step_name") or definition.default_name),
                        "source": action.source or source,
                    },
                ),
            )
        return assigned

    def build(self, action: ResourceAction) -> ExecutionAction | None:
        try:
            definition = self.get_definition(action.kind)
        except ValueError:
            return None
        return ExecutionAction(
            action_id=action.action_id or f"anonymous:{action.kind.value}",
            kind=action.kind,
            step_kind=definition.step_kind,
            name=str(action.name or action.metadata.get("step_name") or definition.default_name),
            config=action.config,
            priority=action.priority,
            source=str(action.source or "unknown"),
            metadata=action.metadata,
        )

    def normalize_selection(self, selection: ResourceSelection) -> ResourceSelection:
        return ResourceSelection(
            use_system_defaults=selection.use_system_defaults,
            use_conversation_defaults=selection.use_conversation_defaults,
            capabilities=selection.normalized_capabilities(),
            actions=selection.normalized_actions(),
            metadata=selection.metadata,
        )


def create_default_action_registry() -> ExecutionActionRegistry:
    registry = ExecutionActionRegistry()
    registry.register(
        ChatActionKindEnum.system_prompt,
        step_kind=ChatStepKindEnum.system,
        default_name="system_prompt",
        config_model=SystemPromptConfig,
    )
    registry.register(
        ChatActionKindEnum.intent_detection,
        step_kind=ChatStepKindEnum.system,
        default_name="intent_detection",
        config_model=IntentDetectionConfig,
    )
    registry.register(
        ChatActionKindEnum.knowledge_retrieval,
        step_kind=ChatStepKindEnum.retrieval,
        default_name="knowledge_retrieval",
        config_model=KnowledgeRetrievalConfig,
    )
    registry.register(
        ChatActionKindEnum.function_call,
        step_kind=ChatStepKindEnum.tool,
        default_name="function_call",
        config_model=FunctionCallConfig,
    )
    registry.register(
        ChatActionKindEnum.tool_call,
        step_kind=ChatStepKindEnum.tool,
        default_name="tool_call",
        config_model=ToolCallConfig,
    )
    registry.register(
        ChatActionKindEnum.mcp_call,
        step_kind=ChatStepKindEnum.tool,
        default_name="mcp_call",
        config_model=MCPCallConfig,
    )
    registry.register(
        ChatActionKindEnum.sub_agent_call,
        step_kind=ChatStepKindEnum.llm,
        default_name="sub_agent_call",
        config_model=SubAgentCallConfig,
    )
    registry.register(
        ChatActionKindEnum.llm_response,
        step_kind=ChatStepKindEnum.llm,
        default_name="llm_response",
        config_model=LLMResponseConfig,
    )
    return registry
