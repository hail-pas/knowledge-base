from __future__ import annotations

from typing import Any
from datetime import datetime

from pydantic import Field, ConfigDict

from service.chat.domain.schema import (
    StrictModel,
    AgentRoleEnum,
    AgentMountModeEnum,
    ResourceSelection,
)


class AgentProfileManifest(StrictModel):
    agent_key: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=128)
    role: AgentRoleEnum = AgentRoleEnum.orchestrator
    description: str = Field(default="", max_length=4000)
    system_prompt: str = Field(default="", max_length=8000)
    llm_model_config_id: int | None = Field(default=None, ge=1)
    default_resource_selection: ResourceSelection = Field(default_factory=ResourceSelection)
    capability_keys: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentProfileCreate(StrictModel):
    manifest: AgentProfileManifest
    is_enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentProfileUpdate(StrictModel):
    manifest: AgentProfileManifest | None = None
    is_enabled: bool | None = None
    metadata: dict[str, Any] | None = None


class AgentProfileSummary(StrictModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    id: int
    owner_account_id: int | None = None
    agent_key: str
    name: str
    role: AgentRoleEnum
    description: str
    system_prompt: str
    llm_model_config_id: int | None = None
    manifest: AgentProfileManifest
    is_enabled: bool
    metadata: dict[str, Any] = Field(default_factory=dict)
    version: int
    created_at: datetime
    updated_at: datetime


class AgentMountCreate(StrictModel):
    source_agent_id: int = Field(ge=1)
    mounted_agent_id: int = Field(ge=1)
    mode: AgentMountModeEnum = AgentMountModeEnum.delegate
    purpose: str = Field(default="", max_length=4000)
    trigger_tags: list[str] = Field(default_factory=list)
    pass_message_history: bool = False
    pass_deps_fields: list[str] = Field(default_factory=list)
    output_contract: str | None = Field(default=None, max_length=255)
    mounted_as_capability: str | None = Field(default=None, min_length=1, max_length=128)
    is_enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentMountUpdate(StrictModel):
    mode: AgentMountModeEnum | None = None
    purpose: str | None = Field(default=None, max_length=4000)
    trigger_tags: list[str] | None = None
    pass_message_history: bool | None = None
    pass_deps_fields: list[str] | None = None
    output_contract: str | None = Field(default=None, max_length=255)
    mounted_as_capability: str | None = Field(default=None, min_length=1, max_length=128)
    is_enabled: bool | None = None
    metadata: dict[str, Any] | None = None


class AgentMountSummary(StrictModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    id: int
    source_agent_id: int
    source_agent_key: str
    mounted_agent_id: int
    mounted_agent_key: str
    mounted_agent_name: str
    mode: AgentMountModeEnum
    purpose: str = ""
    trigger_tags: list[str] = Field(default_factory=list)
    pass_message_history: bool = False
    pass_deps_fields: list[str] = Field(default_factory=list)
    output_contract: str | None = None
    mounted_as_capability: str | None = None
    is_enabled: bool
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
