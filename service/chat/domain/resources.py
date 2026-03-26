from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, JsonValue, model_validator

from service.chat.domain.common import (
    AgentMountModeEnum,
    CapabilityCategoryEnum,
    RESOURCE_SELECTION_TOP_LEVEL_KEYS,
    CapabilityPlannerModeEnum,
    CapabilityKindEnum,
    CapabilityRuntimeKindEnum,
    ChatActionKindEnum,
    LLMResponseConfig,
    MCPCallConfig,
    PersistedToolCallConfig,
    StrictModel,
    SystemPromptConfig,
    action_execution_order,
)


class CapabilityPlannerConfig(StrictModel):
    mode: CapabilityPlannerModeEnum = CapabilityPlannerModeEnum.disabled
    planner_model_config_id: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_planner_config(self) -> CapabilityPlannerConfig:
        if self.mode == CapabilityPlannerModeEnum.llm and self.planner_model_config_id is None:
            raise ValueError("llm planner 需要 planner_model_config_id")
        if self.mode == CapabilityPlannerModeEnum.disabled and self.planner_model_config_id is not None:
            raise ValueError("disabled planner 不允许设置 planner_model_config_id")
        return self


class ActionConfigOverride(StrictModel):
    config: dict[str, JsonValue] = Field(default_factory=dict)
    tool_args: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_override(self) -> ActionConfigOverride:
        if not self.config and not self.tool_args:
            raise ValueError("action override 至少需要 config 或 tool_args")
        return self


class ActionCapabilityMetadata(StrictModel):
    capability_id: int | None = Field(default=None, ge=1)
    capability_key: str | None = Field(default=None, min_length=1, max_length=128)
    capability_kind: CapabilityKindEnum | None = None
    capability_category: CapabilityCategoryEnum | None = None
    capability_name: str | None = Field(default=None, min_length=1, max_length=128)
    capability_version: int | None = Field(default=None, ge=1)
    capability_order: int | None = Field(default=None, ge=1)
    capability_required: bool = False
    capability_runtime_kind: CapabilityRuntimeKindEnum | None = None


class ActionDelegationMetadata(StrictModel):
    mount_id: int | None = Field(default=None, ge=1)
    mounted_agent_id: int | None = Field(default=None, ge=1)
    mounted_agent_key: str | None = Field(default=None, min_length=1, max_length=128)
    mounted_agent_name: str | None = Field(default=None, min_length=1, max_length=128)
    mount_mode: AgentMountModeEnum | None = None
    output_contract: str | None = Field(default=None, min_length=1, max_length=255)
    pass_message_history: bool = True
    pass_deps_fields: list[str] = Field(default_factory=list)


class ActionRuntimeMetadata(StrictModel):
    step_name: str | None = Field(default=None, min_length=1, max_length=100)
    emit_intermediate_events: bool = False


class ActionMetadata(StrictModel):
    capability: ActionCapabilityMetadata | None = None
    delegation: ActionDelegationMetadata | None = None
    runtime: ActionRuntimeMetadata = Field(default_factory=ActionRuntimeMetadata)


def merge_action_metadata(*items: ActionMetadata | None) -> ActionMetadata:
    capability_payload: dict[str, JsonValue] = {}
    delegation_payload: dict[str, JsonValue] = {}
    runtime_payload: dict[str, JsonValue] = {}

    for item in items:
        if item is None:
            continue
        if item.capability is not None:
            capability_payload.update(
                item.capability.model_dump(
                    mode="json",
                    exclude_none=True,
                    exclude_unset=True,
                ),
            )
        if item.delegation is not None:
            delegation_payload.update(
                item.delegation.model_dump(
                    mode="json",
                    exclude_none=True,
                    exclude_unset=True,
                ),
            )
        runtime_payload.update(
            item.runtime.model_dump(
                mode="json",
                exclude_none=True,
                exclude_unset=True,
            ),
        )

    return ActionMetadata(
        capability=(
            ActionCapabilityMetadata.model_validate(capability_payload)
            if capability_payload
            else None
        ),
        delegation=(
            ActionDelegationMetadata.model_validate(delegation_payload)
            if delegation_payload
            else None
        ),
        runtime=ActionRuntimeMetadata.model_validate(runtime_payload),
    )


class SubAgentCallConfig(StrictModel):
    llm_model_config_id: int | None = Field(default=None, ge=1)
    system_prompt: str = Field(min_length=1, max_length=8000)
    instructions: list[str] = Field(default_factory=list)
    actions: list[ResourceAction] = Field(default_factory=list)


class BaseResourceAction(StrictModel):
    action_id: str | None = Field(default=None, min_length=1, max_length=128)
    name: str | None = Field(default=None, min_length=1, max_length=100)
    source: str | None = Field(default=None, min_length=1, max_length=64)
    enabled: bool = True
    priority: int = Field(default=100, ge=0, le=1000)
    metadata: ActionMetadata = Field(default_factory=ActionMetadata)


class SystemPromptAction(BaseResourceAction):
    kind: Literal[ChatActionKindEnum.system_prompt] = ChatActionKindEnum.system_prompt
    config: SystemPromptConfig


class ToolCallAction(BaseResourceAction):
    kind: Literal[ChatActionKindEnum.tool_call] = ChatActionKindEnum.tool_call
    config: PersistedToolCallConfig


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
    SystemPromptAction
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
    action_config_overrides: dict[str, ActionConfigOverride] = Field(default_factory=dict)

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
    planner: CapabilityPlannerConfig | None = None

    def normalized_actions(self) -> list[ResourceAction]:
        keyed_actions: dict[tuple[str, str], ResourceAction] = {}
        anonymous_actions: list[ResourceAction] = []
        for item in self.actions:
            if not item.enabled:
                continue
            if item.kind == ChatActionKindEnum.llm_response:
                keyed_actions[("kind", ChatActionKindEnum.llm_response.value)] = item
                continue
            if item.action_id:
                keyed_actions[("action_id", item.action_id)] = item
                continue
            anonymous_actions.append(item)

        normalized = [*keyed_actions.values(), *anonymous_actions]
        has_llm = any(item.kind == ChatActionKindEnum.llm_response for item in normalized)

        if not has_llm:
            normalized.append(
                LLMResponseAction(
                    action_id="builtin:llm_response",
                    name="llm_response",
                    source="builtin",
                    priority=100,
                ),
            )

        return sorted(
            normalized,
            key=lambda item: (
                item.priority,
                action_execution_order(item.kind),
                item.action_id or item.name or "",
            ),
        )

    def normalized_capabilities(self) -> list[CapabilitySelection]:
        normalized_by_identity: dict[str, CapabilitySelection] = {}
        for item in self.capabilities:
            if not item.enabled:
                continue
            if item.capability_key is not None:
                identity = f"key:{item.capability_key.casefold()}"
            elif item.capability_id is not None:
                identity = f"id:{item.capability_id}"
            else:
                continue
            normalized_by_identity[identity] = item
        return list(normalized_by_identity.values())


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


PersistedToolCallConfig.model_rebuild()
SubAgentCallConfig.model_rebuild()


__all__ = [
    "ActionConfigOverride",
    "ActionCapabilityMetadata",
    "ActionDelegationMetadata",
    "ActionMetadata",
    "ActionRuntimeMetadata",
    "BaseResourceAction",
    "CapabilityPlannerConfig",
    "CapabilitySelection",
    "LLMResponseAction",
    "MCPCallAction",
    "ResourceAction",
    "ResourceSelection",
    "SubAgentCallAction",
    "SubAgentCallConfig",
    "SystemPromptAction",
    "ToolCallAction",
    "merge_action_metadata",
    "parse_resource_selection",
]
