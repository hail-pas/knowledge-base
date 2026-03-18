from __future__ import annotations

from typing import Any, Iterable, cast
from dataclasses import field, dataclass

from pydantic import BaseModel, TypeAdapter

from ext.ext_tortoise.enums import ChatStepKindEnum
from service.chat.domain.schema import (
    MCPCallConfig,
    ToolCallConfig,
    LLMResponseConfig,
    ResourceSelection,
    FunctionCallConfig,
    ResourceCapability,
    SystemPromptConfig,
    IntentDetectionConfig,
    ChatCapabilityKindEnum,
    KnowledgeRetrievalConfig,
)

_RESOURCE_CAPABILITY_ADAPTER = TypeAdapter(ResourceCapability)


@dataclass(slots=True)
class CapabilityDefinition:
    kind: ChatCapabilityKindEnum
    step_kind: ChatStepKindEnum
    default_name: str
    config_model: type[BaseModel]


@dataclass(slots=True)
class CapabilityDescriptor:
    capability_id: str
    kind: ChatCapabilityKindEnum
    step_kind: ChatStepKindEnum
    name: str
    config: BaseModel
    priority: int
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    profile_id: int | None = None
    binding_id: int | None = None


class CapabilityRegistry:
    def __init__(self) -> None:
        self._definitions: dict[ChatCapabilityKindEnum, CapabilityDefinition] = {}

    def register(
        self,
        kind: ChatCapabilityKindEnum,
        *,
        step_kind: ChatStepKindEnum,
        default_name: str,
        config_model: type[BaseModel],
    ) -> None:
        self._definitions[kind] = CapabilityDefinition(
            kind=kind,
            step_kind=step_kind,
            default_name=default_name,
            config_model=config_model,
        )

    def parse_kind(self, kind: ChatCapabilityKindEnum | str) -> ChatCapabilityKindEnum:
        return kind if isinstance(kind, ChatCapabilityKindEnum) else ChatCapabilityKindEnum(kind)

    def get_definition(self, kind: ChatCapabilityKindEnum | str) -> CapabilityDefinition:
        parsed_kind = self.parse_kind(kind)
        definition = self._definitions.get(parsed_kind)
        if definition is None:
            raise ValueError(f"Unsupported capability kind: {parsed_kind}")
        return definition

    def parse_config(self, kind: ChatCapabilityKindEnum | str, config: BaseModel | dict[str, Any]) -> BaseModel:
        definition = self.get_definition(kind)
        payload = config.model_dump(mode="json") if isinstance(config, BaseModel) else config
        return definition.config_model.model_validate(payload)

    def build_capability(
        self,
        kind: ChatCapabilityKindEnum | str,
        *,
        config: BaseModel | dict[str, Any],
        capability_id: str | None = None,
        profile_id: int | None = None,
        binding_id: int | None = None,
        name: str | None = None,
        source: str | None = None,
        enabled: bool = True,
        priority: int = 100,
        metadata: dict[str, Any] | None = None,
    ) -> ResourceCapability:
        parsed_kind = self.parse_kind(kind)
        validated_config = self.parse_config(parsed_kind, config)
        return _RESOURCE_CAPABILITY_ADAPTER.validate_python(
            {
                "kind": parsed_kind.value,
                "capability_id": capability_id,
                "profile_id": profile_id,
                "binding_id": binding_id,
                "name": name or self.get_definition(parsed_kind).default_name,
                "source": source,
                "enabled": enabled,
                "priority": priority,
                "metadata": metadata or {},
                "config": validated_config.model_dump(mode="json"),
            },
        )

    def assign_inline_capability_ids(
        self,
        capabilities: Iterable[ResourceCapability],
        *,
        source: str,
        prefix: str,
    ) -> list[ResourceCapability]:
        assigned: list[ResourceCapability] = []
        for index, capability in enumerate(capabilities, start=1):
            definition = self.get_definition(capability.kind)
            assigned.append(
                capability.model_copy(
                    update={
                        "capability_id": capability.capability_id or f"{prefix}:{index}",
                        "name": capability.name or str(capability.metadata.get("step_name") or definition.default_name),
                        "source": capability.source or source,
                    },
                ),
            )
        return assigned

    def build(self, capability: ResourceCapability) -> CapabilityDescriptor | None:
        try:
            definition = self.get_definition(capability.kind)
        except ValueError:
            return None
        return CapabilityDescriptor(
            capability_id=capability.capability_id or f"anonymous:{capability.kind.value}",
            kind=capability.kind,
            step_kind=definition.step_kind,
            name=str(capability.name or capability.metadata.get("step_name") or definition.default_name),
            config=capability.config,
            priority=capability.priority,
            source=str(capability.source or "unknown"),
            metadata=capability.metadata,
            profile_id=capability.profile_id,
            binding_id=capability.binding_id,
        )

    def normalize_selection(self, selection: ResourceSelection) -> ResourceSelection:
        return ResourceSelection(
            use_system_defaults=False,
            use_conversation_defaults=False,
            capabilities=selection.normalized_capabilities(),
            metadata=selection.metadata,
        )


def create_default_capability_registry() -> CapabilityRegistry:
    registry = CapabilityRegistry()
    registry.register(
        ChatCapabilityKindEnum.system_prompt,
        step_kind=ChatStepKindEnum.system,
        default_name="system_prompt",
        config_model=SystemPromptConfig,
    )
    registry.register(
        ChatCapabilityKindEnum.intent_detection,
        step_kind=ChatStepKindEnum.system,
        default_name="intent_detection",
        config_model=IntentDetectionConfig,
    )
    registry.register(
        ChatCapabilityKindEnum.knowledge_retrieval,
        step_kind=ChatStepKindEnum.retrieval,
        default_name="knowledge_retrieval",
        config_model=KnowledgeRetrievalConfig,
    )
    registry.register(
        ChatCapabilityKindEnum.function_call,
        step_kind=ChatStepKindEnum.tool,
        default_name="function_call",
        config_model=FunctionCallConfig,
    )
    registry.register(
        ChatCapabilityKindEnum.tool_call,
        step_kind=ChatStepKindEnum.tool,
        default_name="tool_call",
        config_model=ToolCallConfig,
    )
    registry.register(
        ChatCapabilityKindEnum.mcp_call,
        step_kind=ChatStepKindEnum.tool,
        default_name="mcp_call",
        config_model=MCPCallConfig,
    )
    registry.register(
        ChatCapabilityKindEnum.llm_response,
        step_kind=ChatStepKindEnum.llm,
        default_name="llm_response",
        config_model=LLMResponseConfig,
    )
    return registry
