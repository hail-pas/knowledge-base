from __future__ import annotations

from typing import Any, Literal, Annotated
from datetime import datetime

from pydantic import Field, ConfigDict, model_validator

from core.types import StrEnum
from service.chat.domain.schema import StrictModel, ResourceAction, CapabilityKindEnum


class CapabilityScopeEnum(StrEnum):
    all = ("all", "当前用户可见全部")
    global_only = ("global_only", "仅全局能力包")
    owned_only = ("owned_only", "仅当前用户能力包")


class CapabilityRoutingModeEnum(StrEnum):
    heuristic = ("heuristic", "启发式")
    llm = ("llm", "仅模型")
    hybrid = ("hybrid", "混合")


class CapabilityRoutingRule(StrictModel):
    always_on: bool = False
    min_score: float = Field(default=0.15, ge=0.0, le=1.0)
    max_selected: int = Field(default=3, ge=1, le=8)
    keywords: list[str] = Field(default_factory=list)
    all_of: list[str] = Field(default_factory=list)
    any_of: list[str] = Field(default_factory=list)
    excluded_keywords: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)


class BaseCapabilityManifest(StrictModel):
    kind: CapabilityKindEnum
    capability_key: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=128)
    description: str = Field(default="", max_length=4000)
    tags: list[str] = Field(default_factory=list)
    triggers: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)
    routing: CapabilityRoutingRule = Field(default_factory=CapabilityRoutingRule)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillCapabilityManifest(BaseCapabilityManifest):
    kind: Literal[CapabilityKindEnum.skill] = CapabilityKindEnum.skill
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
    kind: Literal[CapabilityKindEnum.extension] = CapabilityKindEnum.extension
    actions: list[ResourceAction] = Field(min_length=1)
    provides_tools: bool = False
    provides_context: bool = False
    provides_actions: bool = True


class SubAgentCapabilityManifest(BaseCapabilityManifest):
    kind: Literal[CapabilityKindEnum.sub_agent] = CapabilityKindEnum.sub_agent
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
    is_enabled: bool | None = None
    name: str | None = Field(default=None, min_length=1, max_length=128)
    tags: list[str] = Field(default_factory=list)


class CapabilityPackageSummary(StrictModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    id: int
    owner_account_id: int | None = None
    kind: CapabilityKindEnum
    capability_key: str
    name: str
    description: str
    manifest: CapabilityManifest
    is_enabled: bool
    metadata: dict[str, Any] = Field(default_factory=dict)
    version: int
    created_at: datetime
    updated_at: datetime


class CapabilityRoutingCandidate(StrictModel):
    capability_id: int
    capability_key: str
    capability_kind: CapabilityKindEnum
    name: str
    score: float = Field(ge=0.0, le=1.0)
    selected: bool = False
    reasons: list[str] = Field(default_factory=list)
    source: str = Field(default="", max_length=64)


class CapabilityRoutingDecision(StrictModel):
    mode: CapabilityRoutingModeEnum
    summary: str = Field(default="", max_length=1000)
    selected_capability_ids: list[int] = Field(default_factory=list)
    candidates: list[CapabilityRoutingCandidate] = Field(default_factory=list)


class CapabilityRegistrySnapshot(StrictModel):
    skills: list[CapabilityPackageSummary] = Field(default_factory=list)
    extensions: list[CapabilityPackageSummary] = Field(default_factory=list)
    sub_agents: list[CapabilityPackageSummary] = Field(default_factory=list)

    @property
    def all_packages(self) -> list[CapabilityPackageSummary]:
        return [*self.skills, *self.extensions, *self.sub_agents]
