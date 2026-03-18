from __future__ import annotations

from typing import Any, cast

from loguru import logger
from tortoise.exceptions import IntegrityError

from core.types import ApiException
from service.chat.domain.schema import ResourceSelection, parse_resource_selection
from service.chat.store.repository import ChatRepository
from service.chat.capability.schema import (
    CapabilityBindingQuery,
    CapabilityProfileQuery,
    CapabilityBindingCreate,
    CapabilityBindingDetail,
    CapabilityBindingUpdate,
    CapabilityProfileCreate,
    CapabilityProfileUpdate,
    CapabilityBindingSummary,
    CapabilityProfileSummary,
    ChatCapabilityBindingOwnerEnum,
    ChatCapabilityProfileScopeEnum,
)
from service.chat.capability.registry import CapabilityRegistry
from service.chat.capability.repository import ChatCapabilityRepository
from ext.ext_tortoise.models.knowledge_base import (
    ChatConversation,
    ChatCapabilityBinding,
    ChatCapabilityProfile,
)


class ChatCapabilityService:
    def __init__(
        self,
        registry: CapabilityRegistry,
        repository: ChatCapabilityRepository | None = None,
        chat_repository: ChatRepository | None = None,
    ) -> None:
        self.registry = registry
        self.repository = repository or ChatCapabilityRepository()
        self.chat_repository = chat_repository or ChatRepository()

    async def validate_resource_selection(
        self,
        selection: ResourceSelection,
        *,
        account_id: int | None = None,
        is_staff: bool = True,
    ) -> None:
        await self._resolve_selection_items(
            selection,
            prefix="validate",
            strict=True,
            account_id=account_id,
            is_staff=is_staff,
        )

    async def create_profile(
        self,
        payload: CapabilityProfileCreate,
        *,
        account_id: int | None = None,
        is_staff: bool = True,
    ) -> CapabilityProfileSummary:
        config = self.registry.parse_config(payload.kind, payload.config).model_dump(mode="json")
        try:
            profile = await self.repository.create_profile(
                owner_account_id=None if is_staff or account_id is None else account_id,
                name=payload.name,
                kind=payload.kind.value,
                description=payload.description,
                config=config,
                is_enabled=payload.is_enabled,
                metadata=payload.metadata,
            )
        except IntegrityError as exc:
            raise ApiException("Capability 配置已存在") from exc
        return CapabilityProfileSummary.model_validate(profile)

    async def update_profile(
        self,
        profile_id: int,
        payload: CapabilityProfileUpdate,
        *,
        account_id: int | None = None,
        is_staff: bool = True,
    ) -> CapabilityProfileSummary:
        profile = await self._get_profile_for_write(profile_id, account_id=account_id, is_staff=is_staff)
        if profile is None:
            raise ApiException("Capability 配置不存在")

        update_fields: list[str] = []
        if payload.name is not None:
            profile.name = payload.name
            update_fields.append("name")
        if payload.description is not None:
            profile.description = payload.description
            update_fields.append("description")
        if payload.is_enabled is not None:
            profile.is_enabled = payload.is_enabled
            update_fields.append("is_enabled")
        if payload.metadata is not None:
            profile.metadata = payload.metadata
            update_fields.append("metadata")
        if payload.config is not None:
            profile.config = self.registry.parse_config(profile.kind, payload.config).model_dump(mode="json")
            profile.version += 1
            update_fields.extend(["config", "version"])
        if update_fields:
            try:
                await profile.save(update_fields=update_fields)
            except IntegrityError as exc:
                raise ApiException("Capability 配置已存在") from exc
        return CapabilityProfileSummary.model_validate(profile)

    async def get_profile(
        self,
        profile_id: int,
        *,
        account_id: int | None = None,
        is_staff: bool = True,
    ) -> CapabilityProfileSummary:
        profile = await self._get_profile_for_read(profile_id, account_id=account_id, is_staff=is_staff)
        if profile is None:
            raise ApiException("Capability 配置不存在")
        return CapabilityProfileSummary.model_validate(profile)

    async def list_profiles(
        self,
        query: CapabilityProfileQuery,
        *,
        account_id: int | None = None,
        is_staff: bool = True,
    ) -> list[CapabilityProfileSummary]:
        profiles = await self.repository.list_profiles(
            scope=(query.scope.value if isinstance(query.scope, ChatCapabilityProfileScopeEnum) else str(query.scope)),
            kind=query.kind and query.kind.value,
            is_enabled=query.is_enabled,
            name_contains=query.name,
            account_id=account_id,
            is_staff=is_staff,
        )
        return [CapabilityProfileSummary.model_validate(profile) for profile in profiles]

    async def delete_profile(
        self,
        profile_id: int,
        *,
        account_id: int | None = None,
        is_staff: bool = True,
    ) -> int:
        profile = await self._get_profile_for_write(profile_id, account_id=account_id, is_staff=is_staff)
        if profile is None:
            raise ApiException("Capability 配置不存在")
        await profile.delete()
        return 1

    async def create_binding(
        self,
        payload: CapabilityBindingCreate,
        *,
        account_id: int | None = None,
        is_staff: bool = True,
    ) -> CapabilityBindingDetail:
        await self._validate_binding_owner(
            payload.owner_type,
            payload.owner_id,
            account_id=account_id,
            is_staff=is_staff,
        )
        profile = await self._get_profile_for_read(
            payload.capability_profile_id,
            account_id=account_id,
            is_staff=is_staff,
        )
        if profile is None:
            raise ApiException("Capability 配置不存在")
        self._ensure_binding_profile_compatible(payload.owner_type, profile, is_staff=is_staff)

        exists = await self.repository.binding_queryset().get_or_none(
            owner_type=payload.owner_type.value,
            owner_id=payload.owner_id,
            capability_profile_id=payload.capability_profile_id,
        )
        if exists is not None:
            raise ApiException("Capability 绑定已存在")

        try:
            binding = await self.repository.create_binding(
                owner_type=payload.owner_type.value,
                owner_id=payload.owner_id,
                capability_profile_id=payload.capability_profile_id,
                priority=payload.priority,
                is_enabled=payload.is_enabled,
                metadata=payload.metadata,
            )
        except IntegrityError as exc:
            raise ApiException("Capability 绑定已存在") from exc
        await binding.fetch_related("capability_profile")
        return self._serialize_binding(binding)

    async def update_binding(
        self,
        binding_id: int,
        payload: CapabilityBindingUpdate,
        *,
        account_id: int | None = None,
        is_staff: bool = True,
    ) -> CapabilityBindingDetail:
        binding = await self.repository.get_binding(binding_id)
        if binding is None:
            raise ApiException("Capability 绑定不存在")
        await self._ensure_binding_access(binding, account_id=account_id, is_staff=is_staff)

        update_fields: list[str] = []
        current_profile_id = cast(int, cast(Any, binding).capability_profile_id)
        if payload.capability_profile_id is not None and payload.capability_profile_id != current_profile_id:
            profile = await self._get_profile_for_read(
                payload.capability_profile_id,
                account_id=account_id,
                is_staff=is_staff,
            )
            if profile is None:
                raise ApiException("Capability 配置不存在")
            self._ensure_binding_profile_compatible(
                ChatCapabilityBindingOwnerEnum(binding.owner_type),
                profile,
                is_staff=is_staff,
            )
            exists = await self.repository.binding_queryset().get_or_none(
                owner_type=binding.owner_type,
                owner_id=binding.owner_id,
                capability_profile_id=payload.capability_profile_id,
            )
            if exists is not None and exists.id != binding.id:
                raise ApiException("Capability 绑定已存在")
            binding.capability_profile_id = payload.capability_profile_id
            update_fields.append("capability_profile_id")
        if payload.priority is not None:
            binding.priority = payload.priority
            update_fields.append("priority")
        if payload.is_enabled is not None:
            binding.is_enabled = payload.is_enabled
            update_fields.append("is_enabled")
        if payload.metadata is not None:
            binding.metadata = payload.metadata
            update_fields.append("metadata")
        if update_fields:
            try:
                await binding.save(update_fields=update_fields)
            except IntegrityError as exc:
                raise ApiException("Capability 绑定已存在") from exc
        await binding.fetch_related("capability_profile")
        return self._serialize_binding(binding)

    async def get_binding(
        self,
        binding_id: int,
        *,
        account_id: int | None = None,
        is_staff: bool = True,
    ) -> CapabilityBindingDetail:
        binding = await self.repository.get_binding(binding_id)
        if binding is None:
            raise ApiException("Capability 绑定不存在")
        await self._ensure_binding_access(binding, account_id=account_id, is_staff=is_staff)
        return self._serialize_binding(binding)

    async def list_bindings(
        self,
        query: CapabilityBindingQuery,
        *,
        account_id: int | None = None,
        is_staff: bool = True,
    ) -> list[CapabilityBindingDetail]:
        if query.owner_type == ChatCapabilityBindingOwnerEnum.conversation and query.owner_id is not None:
            await self._validate_binding_owner(
                query.owner_type,
                query.owner_id,
                account_id=account_id,
                is_staff=is_staff,
            )
        bindings = await self.repository.list_bindings(
            owner_type=query.owner_type and query.owner_type.value,
            owner_id=query.owner_id,
            capability_profile_id=query.capability_profile_id,
            is_enabled=query.is_enabled,
        )
        visible: list[CapabilityBindingDetail] = []
        for binding in bindings:
            try:
                await self._ensure_binding_access(binding, account_id=account_id, is_staff=is_staff)
            except ApiException:
                continue
            visible.append(self._serialize_binding(binding))
        return visible

    async def delete_binding(
        self,
        binding_id: int,
        *,
        account_id: int | None = None,
        is_staff: bool = True,
    ) -> int:
        binding = await self.repository.get_binding(binding_id)
        if binding is None:
            raise ApiException("Capability 绑定不存在")
        await self._ensure_binding_access(binding, account_id=account_id, is_staff=is_staff)
        await binding.delete()
        return 1

    async def resolve_resource_selection(
        self,
        *,
        conversation: ChatConversation | None,
        request_selection: ResourceSelection,
        account_id: int | None = None,
        is_staff: bool = True,
    ) -> ResourceSelection:
        resolved_capabilities = []
        metadata: dict[str, Any] = {}

        if request_selection.use_system_defaults:
            for binding in await self.repository.list_bindings(
                owner_type=ChatCapabilityBindingOwnerEnum.system.value,
                owner_id=None,
                is_enabled=True,
            ):
                capability = self._binding_to_capability(binding)
                if capability is not None:
                    resolved_capabilities.append(capability)

        if request_selection.use_conversation_defaults and conversation is not None:
            for binding in await self.repository.list_bindings(
                owner_type=ChatCapabilityBindingOwnerEnum.conversation.value,
                owner_id=conversation.id,
                is_enabled=True,
            ):
                capability = self._binding_to_capability(binding)
                if capability is not None:
                    resolved_capabilities.append(capability)

            conversation_selection = parse_resource_selection(conversation.default_resource_config)
            metadata.update(conversation_selection.metadata)
            resolved_capabilities.extend(
                await self._resolve_selection_items(
                    conversation_selection,
                    prefix=f"conversation:{conversation.id}",
                    strict=False,
                    account_id=account_id,
                    is_staff=is_staff,
                ),
            )

        metadata.update(request_selection.metadata)
        resolved_capabilities.extend(
            await self._resolve_selection_items(
                request_selection,
                prefix=f"request:{conversation.id if conversation else 'standalone'}",
                strict=True,
                account_id=account_id,
                is_staff=is_staff,
            ),
        )

        return self.registry.normalize_selection(
            ResourceSelection(
                use_system_defaults=False,
                use_conversation_defaults=False,
                capabilities=resolved_capabilities,
                metadata=metadata,
            ),
        )

    async def _resolve_selection_items(
        self,
        selection: ResourceSelection,
        *,
        prefix: str,
        strict: bool,
        account_id: int | None,
        is_staff: bool,
    ) -> list:
        resolved = self.registry.assign_inline_capability_ids(
            selection.capabilities,
            source="inline",
            prefix=f"{prefix}:inline",
        )
        resolved.extend(
            await self._materialize_profiles(
                selection.capability_profile_ids,
                strict=strict,
                account_id=account_id,
                is_staff=is_staff,
            ),
        )
        resolved.extend(
            await self._materialize_bindings(
                selection.capability_binding_ids,
                strict=strict,
                account_id=account_id,
                is_staff=is_staff,
            ),
        )
        return resolved

    async def _materialize_profiles(
        self,
        profile_ids: list[int],
        *,
        strict: bool,
        account_id: int | None,
        is_staff: bool,
    ) -> list:
        if not profile_ids:
            return []
        profiles = await self.repository.list_profiles(
            ids=profile_ids,
            account_id=account_id,
            is_staff=is_staff,
        )
        profile_map = {profile.id: profile for profile in profiles}
        capabilities = []
        for profile_id in profile_ids:
            profile = profile_map.get(profile_id)
            if profile is None:
                if strict:
                    raise ApiException(f"Capability 配置不存在: {profile_id}")
                continue
            if not profile.is_enabled:
                if strict:
                    raise ApiException(f"Capability 配置已禁用: {profile_id}")
                continue
            capabilities.append(
                self.registry.build_capability(
                    profile.kind,
                    config=profile.config,
                    capability_id=f"profile:{profile.id}",
                    profile_id=profile.id,
                    name=profile.name,
                    source="profile",
                    enabled=True,
                    priority=100,
                    metadata=profile.metadata or {},
                ),
            )
        return capabilities

    async def _materialize_bindings(
        self,
        binding_ids: list[int],
        *,
        strict: bool,
        account_id: int | None,
        is_staff: bool,
    ) -> list:
        if not binding_ids:
            return []
        bindings = await self.repository.list_bindings(ids=binding_ids)
        binding_map = {binding.id: binding for binding in bindings}
        capabilities = []
        for binding_id in binding_ids:
            binding = binding_map.get(binding_id)
            if binding is None:
                if strict:
                    raise ApiException(f"Capability 绑定不存在: {binding_id}")
                continue
            try:
                await self._ensure_binding_access(binding, account_id=account_id, is_staff=is_staff)
            except ApiException:
                if strict:
                    raise ApiException(f"Capability 绑定不存在: {binding_id}") from None
                continue
            capability = self._binding_to_capability(binding)
            if capability is None:
                if strict:
                    raise ApiException(f"Capability 绑定不可用: {binding_id}")
                continue
            capabilities.append(capability)
        return capabilities

    def _binding_to_capability(self, binding: ChatCapabilityBinding):
        profile = getattr(binding, "capability_profile", None)
        if profile is None:
            return None
        if not binding.is_enabled or not profile.is_enabled:
            return None
        return self.registry.build_capability(
            profile.kind,
            config=profile.config,
            capability_id=f"binding:{binding.id}",
            profile_id=profile.id,
            binding_id=binding.id,
            name=profile.name,
            source=f"binding:{binding.owner_type}",
            enabled=True,
            priority=binding.priority,
            metadata={**(profile.metadata or {}), **(binding.metadata or {})},
        )

    async def _validate_binding_owner(
        self,
        owner_type: ChatCapabilityBindingOwnerEnum,
        owner_id: int | None,
        *,
        account_id: int | None,
        is_staff: bool,
    ) -> None:
        if owner_type == ChatCapabilityBindingOwnerEnum.system:
            if account_id is not None and not is_staff:
                raise ApiException("仅 staff 可管理 system binding")
            return
        if owner_type == ChatCapabilityBindingOwnerEnum.conversation:
            if owner_id is None:
                raise ApiException("conversation 绑定需要 owner_id")
            conversation = (
                await self.chat_repository.get_conversation(owner_id)
                if account_id is None and not is_staff
                else await self.chat_repository.get_accessible_conversation(
                    owner_id,
                    account_id=account_id,
                    is_staff=is_staff,
                )
            )
            if conversation is None:
                raise ApiException("会话不存在")
            return
        raise ApiException(f"Unsupported owner_type: {owner_type}")

    async def _ensure_binding_access(
        self,
        binding: ChatCapabilityBinding,
        *,
        account_id: int | None,
        is_staff: bool,
    ) -> None:
        if account_id is None and not is_staff:
            return
        owner_type = ChatCapabilityBindingOwnerEnum(binding.owner_type)
        if owner_type == ChatCapabilityBindingOwnerEnum.system:
            if not is_staff:
                raise ApiException("仅 staff 可管理 system binding")
            return
        await self._validate_binding_owner(
            owner_type,
            binding.owner_id,
            account_id=account_id,
            is_staff=is_staff,
        )

    async def _get_profile_for_read(
        self,
        profile_id: int,
        *,
        account_id: int | None,
        is_staff: bool,
    ) -> ChatCapabilityProfile | None:
        if account_id is None and is_staff:
            return await self.repository.get_profile(profile_id)
        return await self.repository.get_accessible_profile(
            profile_id,
            account_id=account_id,
            is_staff=is_staff,
        )

    async def _get_profile_for_write(
        self,
        profile_id: int,
        *,
        account_id: int | None,
        is_staff: bool,
    ) -> ChatCapabilityProfile | None:
        profile = await self.repository.get_profile(profile_id)
        if profile is None:
            return None
        if is_staff:
            return profile
        if account_id is None:
            return profile if profile.owner_account_id is None else None
        return profile if profile.owner_account_id == account_id else None

    def _ensure_binding_profile_compatible(
        self,
        owner_type: ChatCapabilityBindingOwnerEnum,
        profile: ChatCapabilityProfile,
        *,
        is_staff: bool,
    ) -> None:
        if owner_type == ChatCapabilityBindingOwnerEnum.system:
            if not is_staff:
                raise ApiException("仅 staff 可管理 system binding")
            if profile.owner_account_id is not None:
                raise ApiException("system binding 仅可绑定全局 Capability 配置")

    def _serialize_binding(self, binding: ChatCapabilityBinding) -> CapabilityBindingDetail:
        profile = getattr(binding, "capability_profile", None)
        if profile is None:
            logger.warning("Capability binding missing prefetched profile: binding_id={}", binding.id)
            raise ApiException("Capability 绑定缺少配置详情")
        summary = CapabilityBindingSummary.model_validate(binding)
        return CapabilityBindingDetail(
            **summary.model_dump(),
            profile=CapabilityProfileSummary.model_validate(profile),
        )
