from __future__ import annotations

from copy import deepcopy
from typing import Any

from loguru import logger
from tortoise.exceptions import IntegrityError

from core.types import ApiException
from service.chat.domain.schema import (
    ResourceAction,
    ResourceSelection,
    ChatActionKindEnum,
    SubAgentCallConfig,
    SystemPromptConfig,
    CapabilitySelection,
)
from service.chat.capability.router import ChatCapabilityRouter
from service.chat.capability.schema import (
    CapabilityKindEnum,
    CapabilityScopeEnum,
    BaseCapabilityManifest,
    CapabilityPackageQuery,
    CapabilityPackageCreate,
    CapabilityPackageUpdate,
    SkillCapabilityManifest,
    CapabilityPackageSummary,
    CapabilityRoutingDecision,
    CapabilityRoutingModeEnum,
    CapabilityRegistrySnapshot,
    SubAgentCapabilityManifest,
    ExtensionCapabilityManifest,
)
from service.chat.execution.registry import (
    ExecutionActionRegistry,
    create_default_action_registry,
)
from service.chat.capability.repository import ChatCapabilityRepository
from ext.ext_tortoise.models.knowledge_base import ChatCapabilityPackage


class ChatCapabilityService:
    BUILTIN_KB_RETRIEVAL_KEY = "knowledge_base_retrieval"

    def __init__(
        self,
        *,
        repository: ChatCapabilityRepository | None = None,
        action_registry: ExecutionActionRegistry | None = None,
        router: ChatCapabilityRouter | None = None,
    ) -> None:
        self.repository = repository or ChatCapabilityRepository()
        self.action_registry = action_registry or create_default_action_registry()
        self.router = router or ChatCapabilityRouter()

    async def create_package(
        self,
        payload: CapabilityPackageCreate,
        *,
        account_id: int | None = None,
        is_staff: bool = True,
    ) -> CapabilityPackageSummary:
        manifest = self._normalize_manifest(payload.manifest)
        try:
            package = await self.repository.create_package(
                owner_account_id=None if is_staff or account_id is None else account_id,
                kind=manifest.kind.value,
                capability_key=manifest.capability_key,
                name=manifest.name,
                description=manifest.description,
                manifest=manifest.model_dump(mode="json"),
                is_enabled=payload.is_enabled,
                metadata=payload.metadata,
            )
        except IntegrityError as exc:
            raise ApiException("Capability package 已存在") from exc
        return self._serialize_package(package)

    async def update_package(
        self,
        capability_id: int,
        payload: CapabilityPackageUpdate,
        *,
        account_id: int | None = None,
        is_staff: bool = True,
    ) -> CapabilityPackageSummary:
        package = await self._get_package_for_write(capability_id, account_id=account_id, is_staff=is_staff)
        if package is None:
            raise ApiException("Capability package 不存在")

        update_fields: list[str] = []
        if payload.manifest is not None:
            manifest = self._normalize_manifest(payload.manifest)
            package.kind = manifest.kind.value
            package.capability_key = manifest.capability_key
            package.name = manifest.name
            package.description = manifest.description
            package.manifest = manifest.model_dump(mode="json")
            package.version += 1
            update_fields.extend(["kind", "capability_key", "name", "description", "manifest", "version"])
        if payload.is_enabled is not None:
            package.is_enabled = payload.is_enabled
            update_fields.append("is_enabled")
        if payload.metadata is not None:
            package.metadata = payload.metadata
            update_fields.append("metadata")
        if update_fields:
            try:
                await package.save(update_fields=update_fields)
            except IntegrityError as exc:
                raise ApiException("Capability package 已存在") from exc
        return self._serialize_package(package)

    async def get_package(
        self,
        capability_id: int,
        *,
        account_id: int | None = None,
        is_staff: bool = True,
    ) -> CapabilityPackageSummary:
        package = await self.repository.get_accessible_package(capability_id, account_id=account_id, is_staff=is_staff)
        if package is None:
            raise ApiException("Capability package 不存在")
        return self._serialize_package(package)

    async def list_packages(
        self,
        query: CapabilityPackageQuery,
        *,
        account_id: int | None = None,
        is_staff: bool = True,
    ) -> list[CapabilityPackageSummary]:
        await self.ensure_builtin_packages()
        packages = await self.repository.list_packages(
            scope=(query.scope.value if isinstance(query.scope, CapabilityScopeEnum) else str(query.scope)),
            kind=query.kind,
            is_enabled=query.is_enabled,
            name_contains=query.name,
            tags=query.tags,
            account_id=account_id,
            is_staff=is_staff,
        )
        return [self._serialize_package(item) for item in packages]

    async def delete_package(
        self,
        capability_id: int,
        *,
        account_id: int | None = None,
        is_staff: bool = True,
    ) -> int:
        package = await self._get_package_for_write(capability_id, account_id=account_id, is_staff=is_staff)
        if package is None:
            raise ApiException("Capability package 不存在")
        await package.delete()
        return 1

    async def build_registry_snapshot(
        self,
        *,
        account_id: int | None = None,
        is_staff: bool = True,
    ) -> CapabilityRegistrySnapshot:
        await self.ensure_builtin_packages()
        packages = await self.list_packages(
            CapabilityPackageQuery(is_enabled=True),
            account_id=account_id,
            is_staff=is_staff,
        )
        return CapabilityRegistrySnapshot(
            skills=[item for item in packages if item.kind == CapabilityKindEnum.skill],
            extensions=[item for item in packages if item.kind == CapabilityKindEnum.extension],
            sub_agents=[item for item in packages if item.kind == CapabilityKindEnum.sub_agent],
        )

    async def resolve_turn_capabilities(
        self,
        *,
        query: str,
        resource_selection: ResourceSelection,
        account_id: int | None = None,
        is_staff: bool = True,
    ) -> tuple[ResourceSelection, CapabilityRoutingDecision]:
        registry = await self.build_registry_snapshot(account_id=account_id, is_staff=is_staff)
        registry_map_by_id = {item.id: item for item in registry.all_packages}
        forced_capability_keys = set()
        required_capability_keys = set()
        for item in resource_selection.capabilities:
            if not item.enabled:
                continue
            if item.capability_key:
                forced_capability_keys.add(item.capability_key)
                if item.required:
                    required_capability_keys.add(item.capability_key)
                continue
            if item.capability_id and item.capability_id in registry_map_by_id:
                resolved_key = registry_map_by_id[item.capability_id].capability_key
                forced_capability_keys.add(resolved_key)
                if item.required:
                    required_capability_keys.add(resolved_key)
        self._validate_required_capabilities(
            required_capability_keys=required_capability_keys,
            packages=registry.all_packages,
        )
        router_config = self._resolve_router_config(resource_selection)
        decision = await self.router.route(
            query=query,
            packages=registry.all_packages,
            forced_capability_keys=forced_capability_keys,
            required_capability_keys=required_capability_keys,
            mode_override=router_config["mode"],
            planner_model_config_id=router_config["planner_model_config_id"],
        )
        logger.info(
            "Capability routing resolved: query={!r}, forced_keys={}, required_keys={}, selected_keys={}, mode={}",
            query[:200],
            sorted(forced_capability_keys),
            sorted(required_capability_keys),
            [item.capability_key for item in registry.all_packages if item.id in set(decision.selected_capability_ids)],
            decision.mode.value,
        )

        package_map = {item.id: item for item in registry.all_packages}
        selected_packages = [package_map[item] for item in decision.selected_capability_ids if item in package_map]
        selection_map = self._build_selection_map(resource_selection.normalized_capabilities())

        actions: list[ResourceAction] = []
        for order, package in enumerate(selected_packages, start=1):
            actions.extend(
                self._materialize_package_actions(
                    package,
                    package_order=order,
                    selection=selection_map.get(package.id) or selection_map.get(package.capability_key),
                ),
            )

        actions.extend(resource_selection.actions)
        selection = self.action_registry.normalize_selection(
            ResourceSelection(
                use_system_defaults=resource_selection.use_system_defaults,
                use_conversation_defaults=resource_selection.use_conversation_defaults,
                capabilities=resource_selection.normalized_capabilities(),
                actions=actions,
                metadata={
                    **resource_selection.metadata,
                    "selected_capability_ids": decision.selected_capability_ids,
                    "selected_capability_keys": [item.capability_key for item in selected_packages],
                    "capability_routing": decision.model_dump(mode="json"),
                },
            ),
        )
        return selection, decision

    def _materialize_package_actions(
        self,
        package: CapabilityPackageSummary,
        *,
        package_order: int,
        selection: CapabilitySelection | None = None,
    ) -> list[ResourceAction]:
        manifest = package.manifest
        package_metadata = {
            "capability_id": package.id,
            "capability_key": package.capability_key,
            "capability_kind": package.kind.value,
            "capability_name": package.name,
            "capability_version": package.version,
            "capability_order": package_order,
            "capability_required": bool(selection.required) if selection is not None else False,
        }

        if isinstance(manifest, SkillCapabilityManifest):
            actions = [
                action.model_copy(
                    update={
                        "action_id": action.action_id or f"capability:{package.id}:action:{index}",
                        "name": action.name or f"{package.capability_key}:{action.kind.value}",
                        "source": "capability:skill",
                        "config": self._resolve_action_config(action, selection=selection),
                        "metadata": {
                            **package_metadata,
                            **(selection.metadata if selection is not None else {}),
                            **action.metadata,
                        },
                    },
                )
                for index, action in enumerate(manifest.actions, start=1)
            ]
            if manifest.instructions:
                actions.append(
                    self.action_registry.build_action(
                        ChatActionKindEnum.system_prompt,
                        config=SystemPromptConfig(instructions=list(manifest.instructions)),
                        action_id=f"capability:{package.id}:instructions",
                        name=f"{package.capability_key}:instructions",
                        source="capability:skill",
                        priority=min([item.priority for item in manifest.actions] or [100]) - 1,
                        metadata=package_metadata,
                    ),
                )
            return actions

        if isinstance(manifest, ExtensionCapabilityManifest):
            actions = []
            for index, action in enumerate(manifest.actions, start=1):
                actions.append(
                    action.model_copy(
                        update={
                            "action_id": action.action_id or f"capability:{package.id}:action:{index}",
                            "name": action.name or f"{package.capability_key}:{action.kind.value}",
                            "source": "capability:extension",
                            "config": self._resolve_action_config(action, selection=selection),
                            "metadata": {
                                **package_metadata,
                                **(selection.metadata if selection is not None else {}),
                                **action.metadata,
                            },
                        },
                    ),
                )
            if manifest.instructions:
                actions.append(
                    self.action_registry.build_action(
                        ChatActionKindEnum.system_prompt,
                        config=SystemPromptConfig(instructions=list(manifest.instructions)),
                        action_id=f"capability:{package.id}:instructions",
                        name=f"{package.capability_key}:instructions",
                        source="capability:extension",
                        priority=min([item.priority for item in manifest.actions] or [100]) - 1,
                        metadata=package_metadata,
                    ),
                )
            return actions

        if isinstance(manifest, SubAgentCapabilityManifest):
            return [
                self.action_registry.build_action(
                    ChatActionKindEnum.sub_agent_call,
                    config=SubAgentCallConfig(
                        llm_model_config_id=manifest.llm_model_config_id,
                        system_prompt=manifest.system_prompt,
                        instructions=list(manifest.instructions),
                        actions=manifest.actions,
                    ),
                    action_id=f"capability:{package.id}:sub-agent",
                    name=package.capability_key,
                    source="capability:sub_agent",
                    priority=100,
                    metadata=package_metadata,
                ),
            ]
        return []

    def _normalize_manifest(self, manifest: BaseCapabilityManifest) -> BaseCapabilityManifest:
        actions = getattr(manifest, "actions", None)
        if not isinstance(actions, list):
            return manifest
        normalized_actions = self.action_registry.assign_inline_action_ids(
            actions,
            source=f"capability_manifest:{manifest.kind.value}",
            prefix=f"capability_manifest:{manifest.capability_key}",
        )
        return manifest.model_copy(update={"actions": normalized_actions})

    async def _get_package_for_write(
        self,
        capability_id: int,
        *,
        account_id: int | None,
        is_staff: bool,
    ) -> ChatCapabilityPackage | None:
        package = await self.repository.get_package(capability_id)
        if package is None:
            return None
        if is_staff:
            return package
        if account_id is None:
            return package if package.owner_account_id is None else None
        return package if package.owner_account_id == account_id else None

    def _serialize_package(self, package: ChatCapabilityPackage) -> CapabilityPackageSummary:
        manifest_payload = {
            "kind": package.kind,
            "capability_key": package.capability_key,
            "name": package.name,
            "description": package.description,
            **(package.manifest or {}),
        }
        manifest = self._manifest_model_from_kind(str(package.kind)).model_validate(manifest_payload)
        return CapabilityPackageSummary(
            id=package.id,
            owner_account_id=package.owner_account_id,
            kind=CapabilityKindEnum(str(package.kind)),
            capability_key=package.capability_key,
            name=package.name,
            description=package.description,
            manifest=manifest,
            is_enabled=package.is_enabled,
            metadata=package.metadata or {},
            version=package.version,
            created_at=package.created_at,
            updated_at=package.updated_at,
        )

    def _manifest_model_from_kind(self, kind: str) -> type[BaseCapabilityManifest]:
        parsed_kind = CapabilityKindEnum(kind)
        if parsed_kind == CapabilityKindEnum.skill:
            return SkillCapabilityManifest
        if parsed_kind == CapabilityKindEnum.extension:
            return ExtensionCapabilityManifest
        return SubAgentCapabilityManifest

    async def ensure_builtin_packages(self) -> None:
        existing = await self.repository.get_global_package_by_key(
            kind=CapabilityKindEnum.extension,
            capability_key=self.BUILTIN_KB_RETRIEVAL_KEY,
        )
        if existing is not None:
            return
        manifest = self._normalize_manifest(self._build_builtin_kb_retrieval_manifest())
        await self.repository.create_package(
            owner_account_id=None,
            kind=manifest.kind.value,
            capability_key=manifest.capability_key,
            name=manifest.name,
            description=manifest.description,
            manifest=manifest.model_dump(mode="json"),
            is_enabled=True,
            metadata={"builtin": True},
        )

    def _build_builtin_kb_retrieval_manifest(self) -> ExtensionCapabilityManifest:
        return ExtensionCapabilityManifest.model_validate(
            {
                "kind": "extension",
                "capability_key": self.BUILTIN_KB_RETRIEVAL_KEY,
                "name": "知识库检索",
                "description": "按需执行知识库检索，将命中内容作为上下文提供给聊天回复。",
                "tags": ["rag", "retrieval", "knowledge-base"],
                "triggers": ["grounded_answer", "citation_required", "knowledge_retrieval"],
                "constraints": ["do_not_guess"],
                "routing": {
                    "keywords": ["知识库", "文档", "检索", "根据文档", "RAG", "资料里"],
                    "min_score": 0.2,
                    "max_selected": 1,
                },
                "instructions": [
                    "当启用知识库检索时，优先基于检索到的结果回答；证据不足时明确说明。",
                ],
                "provides_context": True,
                "provides_actions": True,
                "actions": [
                    {
                        "kind": "knowledge_retrieval",
                        "priority": 30,
                        "name": "knowledge_base_retrieval",
                        "config": {
                            "collection_ids": [1],
                            "top_k": 5,
                        },
                        "metadata": {
                            "builtin": True,
                            "supports_runtime_override": True,
                            "emit_intermediate_events": True,
                        },
                    },
                ],
            },
        )

    def _build_selection_map(self, selections: list[CapabilitySelection]) -> dict[int | str, CapabilitySelection]:
        mapping: dict[int | str, CapabilitySelection] = {}
        for item in selections:
            if item.capability_id is not None:
                mapping[item.capability_id] = item
            if item.capability_key is not None:
                mapping[item.capability_key] = item
        return mapping

    def _resolve_action_config(
        self,
        action: ResourceAction,
        *,
        selection: CapabilitySelection | None,
    ):
        if selection is None:
            return action.config
        overrides = selection.metadata.get("action_config_overrides", {})
        if not isinstance(overrides, dict):
            return action.config

        override_payload = overrides.get(action.kind.value)
        if not isinstance(override_payload, dict):
            return action.config

        base_payload = deepcopy(action.config.model_dump(mode="json"))
        base_payload.update(override_payload)
        return self.action_registry.parse_config(action.kind, base_payload)

    def _validate_required_capabilities(
        self,
        *,
        required_capability_keys: set[str],
        packages: list[CapabilityPackageSummary],
    ) -> None:
        if not required_capability_keys:
            return
        visible_keys = {item.capability_key for item in packages}
        missing = sorted(required_capability_keys - visible_keys)
        if missing:
            raise ApiException(f"必需 capability package 不存在或不可见: {', '.join(missing)}")

    def _resolve_router_config(self, resource_selection: ResourceSelection) -> dict[str, Any]:
        payload = resource_selection.metadata.get("capability_router", {})
        if not isinstance(payload, dict):
            return {"mode": None, "planner_model_config_id": None}
        mode = payload.get("mode")
        planner_model_config_id = payload.get("planner_model_config_id")
        return {
            "mode": CapabilityRoutingModeEnum(mode) if isinstance(mode, str) else None,
            "planner_model_config_id": (
                int(planner_model_config_id)
                if isinstance(planner_model_config_id, int) and planner_model_config_id > 0
                else None
            ),
        }
