from __future__ import annotations

from typing import Any, Iterable

from tortoise.queryset import QuerySet
from tortoise.expressions import Q

from service.chat.capability.schema import CapabilityKindEnum
from ext.ext_tortoise.models.knowledge_base import ChatCapabilityPackage


class ChatCapabilityRepository:
    def package_queryset(self) -> QuerySet[ChatCapabilityPackage]:
        return ChatCapabilityPackage.filter(deleted_at=0)

    async def create_package(self, **data: Any) -> ChatCapabilityPackage:
        return await ChatCapabilityPackage.create(**data)

    async def get_package(self, capability_id: int) -> ChatCapabilityPackage | None:
        return await self.package_queryset().get_or_none(id=capability_id)

    async def get_global_package_by_key(
        self,
        *,
        kind: CapabilityKindEnum,
        capability_key: str,
    ) -> ChatCapabilityPackage | None:
        return await self.package_queryset().get_or_none(
            owner_account_id=None,
            kind=kind.value,
            capability_key=capability_key,
        )

    async def get_accessible_package(
        self,
        capability_id: int,
        *,
        account_id: int | None,
        is_staff: bool,
    ) -> ChatCapabilityPackage | None:
        queryset = self.package_queryset().filter(id=capability_id)
        if not is_staff:
            if account_id is None:
                queryset = queryset.filter(owner_account_id=None)
            else:
                queryset = queryset.filter(Q(owner_account_id=None) | Q(owner_account_id=account_id))
        return await queryset.first()

    async def list_packages(
        self,
        *,
        ids: Iterable[int] | None = None,
        kind: CapabilityKindEnum | None = None,
        category: str | None = None,
        runtime_kind: str | None = None,
        is_enabled: bool | None = None,
        name_contains: str | None = None,
        tags: list[str] | None = None,
        account_id: int | None = None,
        is_staff: bool = True,
        scope: str = "all",
    ) -> list[ChatCapabilityPackage]:
        queryset = self.package_queryset()
        if kind is not None:
            queryset = queryset.filter(kind=kind.value)
        if category is not None:
            queryset = queryset.filter(category=category)
        if runtime_kind is not None:
            queryset = queryset.filter(runtime_kind=runtime_kind)
        if scope == "global_only":
            queryset = queryset.filter(owner_account_id=None)
        elif scope == "owned_only":
            if account_id is None:
                queryset = (
                    queryset.filter(id__in=[]) if not is_staff else queryset.filter(owner_account_id__isnull=False)
                )
            else:
                queryset = queryset.filter(owner_account_id=account_id)
        elif not is_staff:
            if account_id is None:
                queryset = queryset.filter(owner_account_id=None)
            else:
                queryset = queryset.filter(Q(owner_account_id=None) | Q(owner_account_id=account_id))

        if ids is not None:
            queryset = queryset.filter(id__in=list(ids))
        if is_enabled is not None:
            queryset = queryset.filter(is_enabled=is_enabled)
        if name_contains:
            queryset = queryset.filter(
                Q(name__icontains=name_contains) | Q(capability_key__icontains=name_contains),
            )

        packages = await queryset.order_by("-id")
        if not tags:
            return packages

        expected = {item.casefold() for item in tags if item.strip()}
        return [
            item
            for item in packages
            if expected.issubset({str(tag).casefold() for tag in (item.manifest or {}).get("tags", [])})
        ]
