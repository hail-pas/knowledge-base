from __future__ import annotations

from typing import Any

from tortoise.expressions import Q
from tortoise.queryset import QuerySet

from ext.ext_tortoise.models.knowledge_base import ChatAgentProfile, ChatAgentMount


class ChatAgentRepository:
    def agent_queryset(self) -> QuerySet[ChatAgentProfile]:
        return ChatAgentProfile.filter(deleted_at=0)

    def mount_queryset(self) -> QuerySet[ChatAgentMount]:
        return ChatAgentMount.filter(deleted_at=0)

    async def create_agent(self, **data: Any) -> ChatAgentProfile:
        return await ChatAgentProfile.create(**data)

    async def get_agent(self, agent_id: int) -> ChatAgentProfile | None:
        return await self.agent_queryset().get_or_none(id=agent_id)

    async def get_agent_by_key(
        self,
        *,
        agent_key: str,
        account_id: int | None,
        is_staff: bool,
    ) -> ChatAgentProfile | None:
        queryset = self.agent_queryset().filter(agent_key=agent_key)
        if not is_staff:
            if account_id is None:
                queryset = queryset.filter(owner_account_id=None)
            else:
                queryset = queryset.filter(Q(owner_account_id=None) | Q(owner_account_id=account_id))
        return await queryset.first()

    async def list_agents(
        self,
        *,
        account_id: int | None,
        is_staff: bool,
        role: str | None = None,
        is_enabled: bool | None = None,
    ) -> list[ChatAgentProfile]:
        queryset = self.agent_queryset()
        if role:
            queryset = queryset.filter(role=role)
        if is_enabled is not None:
            queryset = queryset.filter(is_enabled=is_enabled)
        if not is_staff:
            if account_id is None:
                queryset = queryset.filter(owner_account_id=None)
            else:
                queryset = queryset.filter(Q(owner_account_id=None) | Q(owner_account_id=account_id))
        return await queryset.order_by("-id")

    async def list_agents_by_ids(
        self,
        agent_ids: list[int],
        *,
        account_id: int | None,
        is_staff: bool,
        is_enabled: bool | None = None,
    ) -> list[ChatAgentProfile]:
        if not agent_ids:
            return []
        queryset = self.agent_queryset().filter(id__in=agent_ids)
        if is_enabled is not None:
            queryset = queryset.filter(is_enabled=is_enabled)
        if not is_staff:
            if account_id is None:
                queryset = queryset.filter(owner_account_id=None)
            else:
                queryset = queryset.filter(Q(owner_account_id=None) | Q(owner_account_id=account_id))
        return await queryset.order_by("-id")

    async def create_mount(self, **data: Any) -> ChatAgentMount:
        return await ChatAgentMount.create(**data)

    async def get_mount(self, mount_id: int) -> ChatAgentMount | None:
        return await self.mount_queryset().select_related("source_agent", "mounted_agent").get_or_none(id=mount_id)

    async def list_mounts(
        self,
        *,
        source_agent_id: int | None = None,
        mounted_agent_id: int | None = None,
        is_enabled: bool | None = None,
    ) -> list[ChatAgentMount]:
        queryset = self.mount_queryset().select_related("source_agent", "mounted_agent")
        if source_agent_id is not None:
            queryset = queryset.filter(source_agent_id=source_agent_id)
        if mounted_agent_id is not None:
            queryset = queryset.filter(mounted_agent_id=mounted_agent_id)
        if is_enabled is not None:
            queryset = queryset.filter(is_enabled=is_enabled)
        return await queryset.order_by("-id")
