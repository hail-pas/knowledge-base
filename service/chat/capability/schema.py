from __future__ import annotations

from typing import Any
from datetime import datetime

from pydantic import Field, ConfigDict, model_validator

from core.types import StrEnum
from service.chat.domain.schema import StrictModel, ChatCapabilityKindEnum


class ChatCapabilityBindingOwnerEnum(StrEnum):
    system = ("system", "系统默认")
    conversation = ("conversation", "会话")


class ChatCapabilityProfileScopeEnum(StrEnum):
    all = ("all", "当前用户可见全部")
    global_only = ("global_only", "仅全局配置")
    owned_only = ("owned_only", "仅当前用户拥有")


class CapabilityProfileCreate(StrictModel):
    name: str = Field(min_length=1, max_length=128)
    kind: ChatCapabilityKindEnum
    description: str = Field(default="", max_length=2000)
    config: dict[str, Any] = Field(default_factory=dict)
    is_enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class CapabilityProfileUpdate(StrictModel):
    name: str | None = Field(default=None, min_length=1, max_length=128)
    description: str | None = Field(default=None, max_length=2000)
    config: dict[str, Any] | None = None
    is_enabled: bool | None = None
    metadata: dict[str, Any] | None = None


class CapabilityProfileQuery(StrictModel):
    scope: ChatCapabilityProfileScopeEnum = ChatCapabilityProfileScopeEnum.all
    kind: ChatCapabilityKindEnum | None = None
    is_enabled: bool | None = None
    name: str | None = Field(default=None, min_length=1, max_length=128)


class CapabilityProfileSummary(StrictModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    id: int
    owner_account_id: int | None = None
    name: str
    kind: ChatCapabilityKindEnum
    description: str
    config: dict[str, Any] = Field(default_factory=dict)
    is_enabled: bool
    metadata: dict[str, Any] = Field(default_factory=dict)
    version: int
    created_at: datetime
    updated_at: datetime


class CapabilityBindingCreate(StrictModel):
    owner_type: ChatCapabilityBindingOwnerEnum
    owner_id: int | None = Field(default=None, ge=1)
    capability_profile_id: int = Field(ge=1)
    priority: int = Field(default=100, ge=0, le=1000)
    is_enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_owner(self) -> CapabilityBindingCreate:
        if self.owner_type == ChatCapabilityBindingOwnerEnum.system and self.owner_id is not None:
            raise ValueError("system 绑定不需要 owner_id")
        if self.owner_type == ChatCapabilityBindingOwnerEnum.conversation and self.owner_id is None:
            raise ValueError("conversation 绑定需要 owner_id")
        return self


class CapabilityBindingUpdate(StrictModel):
    capability_profile_id: int | None = Field(default=None, ge=1)
    priority: int | None = Field(default=None, ge=0, le=1000)
    is_enabled: bool | None = None
    metadata: dict[str, Any] | None = None


class CapabilityBindingQuery(StrictModel):
    owner_type: ChatCapabilityBindingOwnerEnum | None = None
    owner_id: int | None = Field(default=None, ge=1)
    capability_profile_id: int | None = Field(default=None, ge=1)
    is_enabled: bool | None = None


class CapabilityBindingSummary(StrictModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    id: int
    owner_type: ChatCapabilityBindingOwnerEnum
    owner_id: int | None = None
    capability_profile_id: int
    priority: int
    is_enabled: bool
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class CapabilityBindingDetail(CapabilityBindingSummary):
    profile: CapabilityProfileSummary
