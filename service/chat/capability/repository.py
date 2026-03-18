from __future__ import annotations

from typing import Any, Iterable

from tortoise.queryset import QuerySet
from tortoise.expressions import Q

from ext.ext_tortoise.models.knowledge_base import (
    ChatCapabilityBinding,
    ChatCapabilityProfile,
)


class ChatCapabilityRepository:
    def profile_queryset(self) -> QuerySet[ChatCapabilityProfile]:
        return ChatCapabilityProfile.filter(deleted_at=0)

    def binding_queryset(self) -> QuerySet[ChatCapabilityBinding]:
        return ChatCapabilityBinding.filter(deleted_at=0).prefetch_related("capability_profile")

    async def create_profile(self, **data: Any) -> ChatCapabilityProfile:
        return await ChatCapabilityProfile.create(**data)

    async def get_profile(self, profile_id: int) -> ChatCapabilityProfile | None:
        return await self.profile_queryset().get_or_none(id=profile_id)

    async def get_accessible_profile(
        self,
        profile_id: int,
        *,
        account_id: int | None,
        is_staff: bool,
    ) -> ChatCapabilityProfile | None:
        queryset = self.profile_queryset().filter(id=profile_id)
        if not is_staff:
            if account_id is None:
                queryset = queryset.filter(owner_account_id=None)
            else:
                queryset = queryset.filter(Q(owner_account_id=None) | Q(owner_account_id=account_id))
        return await queryset.first()

    async def list_profiles(
        self,
        *,
        ids: Iterable[int] | None = None,
        kind: str | None = None,
        is_enabled: bool | None = None,
        name_contains: str | None = None,
        account_id: int | None = None,
        is_staff: bool = True,
        scope: str = "all",
    ) -> list[ChatCapabilityProfile]:
        queryset = self.profile_queryset()
        if scope == "global_only":
            queryset = queryset.filter(owner_account_id=None)
        elif scope == "owned_only":
            if account_id is None:
                queryset = queryset.filter(owner_account_id__isnull=False) if is_staff else queryset.filter(id__in=[])
            else:
                queryset = queryset.filter(owner_account_id=account_id)
        elif not is_staff:
            if account_id is None:
                queryset = queryset.filter(owner_account_id=None)
            else:
                queryset = queryset.filter(Q(owner_account_id=None) | Q(owner_account_id=account_id))
        if ids is not None:
            queryset = queryset.filter(id__in=list(ids))
        if kind is not None:
            queryset = queryset.filter(kind=kind)
        if is_enabled is not None:
            queryset = queryset.filter(is_enabled=is_enabled)
        if name_contains:
            queryset = queryset.filter(name__icontains=name_contains)
        return await queryset.order_by("-id")

    async def create_binding(self, **data: Any) -> ChatCapabilityBinding:
        return await ChatCapabilityBinding.create(**data)

    async def get_binding(self, binding_id: int) -> ChatCapabilityBinding | None:
        return await self.binding_queryset().get_or_none(id=binding_id)

    async def list_bindings(
        self,
        *,
        ids: Iterable[int] | None = None,
        owner_type: str | None = None,
        owner_id: int | None = None,
        capability_profile_id: int | None = None,
        is_enabled: bool | None = None,
    ) -> list[ChatCapabilityBinding]:
        queryset = self.binding_queryset()
        if ids is not None:
            queryset = queryset.filter(id__in=list(ids))
        if owner_type is not None:
            queryset = queryset.filter(owner_type=owner_type)
        if owner_id is not None:
            queryset = queryset.filter(owner_id=owner_id)
        if capability_profile_id is not None:
            queryset = queryset.filter(capability_profile_id=capability_profile_id)
        if is_enabled is not None:
            queryset = queryset.filter(is_enabled=is_enabled)
        return await queryset.order_by("priority", "id")
