from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, model_validator

from core.types import StrEnum
from service.chat.domain.schema import (
    CapabilityCategoryEnum,
    CapabilityKindEnum,
    CapabilityPlannerModeEnum,
    CapabilityRuntimeKindEnum,
    ResourceAction,
    StrictModel,
)


class CapabilityScopeEnum(StrEnum):
    all = ("all", "当前用户可见全部")
    global_only = ("global_only", "仅全局能力包")
    owned_only = ("owned_only", "仅当前用户能力包")


class CapabilityGovernance(StrictModel):
    visible_to_agents: list[str] = Field(default_factory=list)
    requires_deps: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseCapabilityManifest(StrictModel):
    kind: CapabilityKindEnum
    category: CapabilityCategoryEnum = CapabilityCategoryEnum.domain
    runtime_kind: CapabilityRuntimeKindEnum = CapabilityRuntimeKindEnum.local_toolset
    capability_key: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=128)
    description: str = Field(default="", max_length=4000)
    tags: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)
    always_on: bool = False
    governance: CapabilityGovernance = Field(default_factory=CapabilityGovernance)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillCapabilityManifest(BaseCapabilityManifest):
    kind: Literal[CapabilityKindEnum.skill] = CapabilityKindEnum.skill  # type: ignore
    runtime_kind: Literal[CapabilityRuntimeKindEnum.local_toolset] = CapabilityRuntimeKindEnum.local_toolset  # type: ignore
    preferred_extension_keys: list[str] = Field(default_factory=list)
    preferred_sub_agent_keys: list[str] = Field(default_factory=list)
    actions: list[ResourceAction] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_skill_manifest(self) -> SkillCapabilityManifest:
        if (
            not self.instructions
            and not self.actions
            and not self.preferred_extension_keys
            and not self.preferred_sub_agent_keys
        ):
            raise ValueError("skill manifest 需要至少定义 instructions、actions 或 preferred references")
        return self


class ExtensionCapabilityManifest(BaseCapabilityManifest):
    kind: Literal[CapabilityKindEnum.extension] = CapabilityKindEnum.extension  # type: ignore
    runtime_kind: CapabilityRuntimeKindEnum = CapabilityRuntimeKindEnum.local_toolset
    actions: list[ResourceAction] = Field(min_length=1)
    provides_tools: bool = False
    provides_context: bool = False
    provides_actions: bool = True


class SubAgentCapabilityManifest(BaseCapabilityManifest):
    kind: Literal[CapabilityKindEnum.sub_agent] = CapabilityKindEnum.sub_agent  # type: ignore
    category: CapabilityCategoryEnum = CapabilityCategoryEnum.agent
    runtime_kind: Literal[CapabilityRuntimeKindEnum.agent_delegate] = CapabilityRuntimeKindEnum.agent_delegate  # type: ignore
    system_prompt: str = Field(min_length=1, max_length=8000)
    llm_model_config_id: int | None = Field(default=None, ge=1)
    actions: list[ResourceAction] = Field(default_factory=list)


CapabilityManifest = Annotated[
    SkillCapabilityManifest | ExtensionCapabilityManifest | SubAgentCapabilityManifest,
    Field(discriminator="kind"),
]


class CapabilityPackageCreate(StrictModel):
    manifest: CapabilityManifest
    is_enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class CapabilityPackageUpdate(StrictModel):
    manifest: CapabilityManifest | None = None
    is_enabled: bool | None = None
    metadata: dict[str, Any] | None = None


class CapabilityPackageQuery(StrictModel):
    scope: CapabilityScopeEnum = CapabilityScopeEnum.all
    kind: CapabilityKindEnum | None = None
    category: CapabilityCategoryEnum | None = None
    runtime_kind: CapabilityRuntimeKindEnum | None = None
    is_enabled: bool | None = None
    name: str | None = Field(default=None, min_length=1, max_length=128)
    tags: list[str] = Field(default_factory=list)


class CapabilityPackageSummary(StrictModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    id: int
    owner_account_id: int | None = None
    kind: CapabilityKindEnum
    category: CapabilityCategoryEnum
    runtime_kind: CapabilityRuntimeKindEnum
    capability_key: str
    name: str
    description: str
    manifest: CapabilityManifest
    visible_to_agents: list[str] = Field(default_factory=list)
    requires_deps: list[str] = Field(default_factory=list)
    is_enabled: bool
    metadata: dict[str, Any] = Field(default_factory=dict)
    version: int
    created_at: datetime
    updated_at: datetime
