from __future__ import annotations

from copy import deepcopy
from typing import Any

from loguru import logger
from tortoise.exceptions import IntegrityError

from core.types import ApiException
from ext.ext_tortoise.models.knowledge_base import ChatCapabilityPackage
from service.chat.capability.repository import ChatCapabilityRepository
from service.chat.capability.schema import (
    BaseCapabilityManifest,
    CapabilityKindEnum,
    CapabilityPackageCreate,
    CapabilityPackageQuery,
    CapabilityPackageSummary,
    CapabilityPackageUpdate,
    CapabilityScopeEnum,
    ExtensionCapabilityManifest,
    SkillCapabilityManifest,
    SubAgentCapabilityManifest,
)
from service.chat.domain.schema import (
    ActionConfigOverride,
    ActionCapabilityMetadata,
    ActionMetadata,
    CapabilityCategoryEnum,
    CapabilityRuntimeKindEnum,
    CapabilitySelection,
    ChatActionKindEnum,
    ResourceAction,
    SubAgentCallConfig,
    SystemPromptConfig,
    merge_action_metadata,
)
from service.chat.execution.registry import ExecutionActionRegistry, create_default_action_registry


class ChatCapabilityService:
    BUILTIN_CORE_DATETIME_KEY = "core_datetime"
    BUILTIN_KB_RETRIEVAL_KEY = "knowledge_base_retrieval"
    BUILTIN_GUARDED_WORK_ORDER_KEY = "guarded_work_order_create"

    def __init__(
        self,
        *,
        repository: ChatCapabilityRepository | None = None,
        action_registry: ExecutionActionRegistry | None = None,
    ) -> None:
        self.repository = repository or ChatCapabilityRepository()
        self.action_registry = action_registry or create_default_action_registry()

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
                category=manifest.category.value,
                runtime_kind=manifest.runtime_kind.value,
                capability_key=manifest.capability_key,
                name=manifest.name,
                description=manifest.description,
                manifest=manifest.model_dump(mode="json"),
                visible_to_agents=manifest.governance.visible_to_agents,
                requires_deps=manifest.governance.requires_deps,
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
            package.kind = manifest.kind.value  # type: ignore
            package.category = manifest.category.value  # type: ignore
            package.runtime_kind = manifest.runtime_kind.value  # type: ignore
            package.capability_key = manifest.capability_key
            package.name = manifest.name
            package.description = manifest.description
            package.manifest = manifest.model_dump(mode="json")
            package.visible_to_agents = manifest.governance.visible_to_agents
            package.requires_deps = manifest.governance.requires_deps
            package.version += 1
            update_fields.extend(
                [
                    "kind",
                    "category",
                    "runtime_kind",
                    "capability_key",
                    "name",
                    "description",
                    "manifest",
                    "visible_to_agents",
                    "requires_deps",
                    "version",
                ],
            )
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
            category=query.category.value if query.category is not None else None,
            runtime_kind=query.runtime_kind.value if query.runtime_kind is not None else None,
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

    def compile_package_actions(
        self,
        package: CapabilityPackageSummary,
        *,
        package_order: int = 100,
        selection: CapabilitySelection | None = None,
    ) -> list[ResourceAction]:
        return self._materialize_package_actions(
            package,
            package_order=package_order,
            selection=selection,
        )

    def _materialize_package_actions(
        self,
        package: CapabilityPackageSummary,
        *,
        package_order: int,
        selection: CapabilitySelection | None = None,
    ) -> list[ResourceAction]:
        manifest = package.manifest
        package_metadata = ActionCapabilityMetadata(
            capability_id=package.id,
            capability_key=package.capability_key,
            capability_kind=package.kind,
            capability_category=package.category,
            capability_name=package.name,
            capability_version=package.version,
            capability_order=package_order,
            capability_required=bool(selection.required) if selection is not None else False,
            capability_runtime_kind=package.runtime_kind,
        )

        if isinstance(manifest, SkillCapabilityManifest):
            return self._materialize_inline_package_actions(
                package=package,
                package_metadata=package_metadata,
                selection=selection,
                actions=manifest.actions,
                instructions=manifest.instructions,
                source="capability:skill",
            )

        if isinstance(manifest, ExtensionCapabilityManifest):
            return self._materialize_inline_package_actions(
                package=package,
                package_metadata=package_metadata,
                selection=selection,
                actions=manifest.actions,
                instructions=manifest.instructions,
                source="capability:extension",
            )

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
                    metadata=ActionMetadata(capability=package_metadata),
                ),
            ]
        return []

    def _materialize_inline_package_actions(
        self,
        *,
        package: CapabilityPackageSummary,
        package_metadata: ActionCapabilityMetadata,
        selection: CapabilitySelection | None,
        actions: list[ResourceAction],
        instructions: list[str],
        source: str,
    ) -> list[ResourceAction]:
        resolved_actions = [
            action.model_copy(
                update={
                    "action_id": action.action_id or f"capability:{package.id}:action:{index}",
                    "name": action.name or f"{package.capability_key}:{action.kind.value}",
                    "source": source,
                    "config": self._resolve_action_config(action, selection=selection),
                    "metadata": merge_action_metadata(
                        ActionMetadata(capability=package_metadata),
                        action.metadata,
                    ),
                },
            )
            for index, action in enumerate(actions, start=1)
        ]
        if instructions:
            resolved_actions.append(
                self.action_registry.build_action(
                    ChatActionKindEnum.system_prompt,
                    config=SystemPromptConfig(instructions=list(instructions)),
                    action_id=f"capability:{package.id}:instructions",
                    name=f"{package.capability_key}:instructions",
                    source=source,
                    priority=min([item.priority for item in actions] or [100]) - 1,
                    metadata=ActionMetadata(capability=package_metadata),
                ),
            )
        return resolved_actions

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
            "category": getattr(package, "category", CapabilityCategoryEnum.domain.value),
            "runtime_kind": getattr(package, "runtime_kind", CapabilityRuntimeKindEnum.local_toolset.value),
            "capability_key": package.capability_key,
            "name": package.name,
            "description": package.description,
            "governance": {
                "visible_to_agents": list(getattr(package, "visible_to_agents", []) or []),
                "requires_deps": list(getattr(package, "requires_deps", []) or []),
            },
            **(package.manifest or {}),
        }
        manifest = self._manifest_model_from_kind(str(package.kind)).model_validate(manifest_payload)
        return CapabilityPackageSummary(
            id=package.id,
            owner_account_id=package.owner_account_id,
            kind=CapabilityKindEnum(str(package.kind)),
            category=manifest.category,
            runtime_kind=manifest.runtime_kind,
            capability_key=package.capability_key,
            name=package.name,
            description=package.description,
            manifest=manifest,  # type: ignore
            visible_to_agents=list(manifest.governance.visible_to_agents),
            requires_deps=list(manifest.governance.requires_deps),
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
        builtins = [
            self._build_builtin_core_datetime_manifest(),
            self._build_builtin_kb_retrieval_manifest(),
            self._build_builtin_guarded_work_order_manifest(),
        ]
        for builtin in builtins:
            existing = await self.repository.get_global_package_by_key(
                kind=builtin.kind,
                capability_key=builtin.capability_key,
            )
            if existing is not None:
                continue
            manifest = self._normalize_manifest(builtin)
            try:
                await self.repository.create_package(
                    owner_account_id=None,
                    kind=manifest.kind.value,
                    category=manifest.category.value,
                    runtime_kind=manifest.runtime_kind.value,
                    capability_key=manifest.capability_key,
                    name=manifest.name,
                    description=manifest.description,
                    manifest=manifest.model_dump(mode="json"),
                    visible_to_agents=manifest.governance.visible_to_agents,
                    requires_deps=manifest.governance.requires_deps,
                    is_enabled=True,
                    metadata={"builtin": True},
                )
            except IntegrityError:
                logger.info("Builtin capability package already created concurrently: {}", manifest.capability_key)

    def _build_builtin_core_datetime_manifest(self) -> SkillCapabilityManifest:
        return SkillCapabilityManifest.model_validate(
            {
                "kind": "skill",
                "category": "core",
                "runtime_kind": "local_toolset",
                "capability_key": self.BUILTIN_CORE_DATETIME_KEY,
                "name": "当前时间",
                "description": "提供时间日期等平台基础能力。",
                "tags": ["time", "date", "clock"],
                "instructions": ["涉及当前时间或日期时，优先使用时间能力。"],
                "actions": [
                    {
                        "kind": "tool_call",
                        "priority": 10,
                        "config": {
                            "tools": [{"tool_name": "current_datetime"}],
                            "stop_after_terminal": True,
                        },
                    },
                ],
            },
        )

    def _build_builtin_kb_retrieval_manifest(self) -> ExtensionCapabilityManifest:
        return ExtensionCapabilityManifest.model_validate(
            {
                "kind": "extension",
                "category": "infra",
                "runtime_kind": "local_toolset",
                "capability_key": self.BUILTIN_KB_RETRIEVAL_KEY,
                "name": "知识库检索",
                "description": "按需执行知识库检索，将命中内容作为上下文提供给聊天回复。",
                "tags": ["rag", "retrieval", "knowledge-base"],
                "constraints": ["do_not_guess"],
                "instructions": [
                    "当启用知识库检索时，优先基于检索到的结果回答；证据不足时明确说明。",
                ],
                "provides_context": True,
                "provides_actions": True,
                "actions": [
                    {
                        "kind": "tool_call",
                        "priority": 30,
                        "name": "knowledge_base_search",
                        "config": {
                            "tools": [
                                {
                                    "tool_name": "knowledge_base_search",
                                    "args": {
                                        "collection_ids": [1],
                                        "top_k": 5,
                                    },
                                },
                            ],
                        },
                        "metadata": {
                            "runtime": {
                                "emit_intermediate_events": True,
                            },
                        },
                    },
                ],
            },
        )

    def _build_builtin_guarded_work_order_manifest(self) -> SkillCapabilityManifest:
        return SkillCapabilityManifest.model_validate(
            {
                "kind": "skill",
                "category": "guarded",
                "runtime_kind": "local_toolset",
                "capability_key": self.BUILTIN_GUARDED_WORK_ORDER_KEY,
                "name": "模拟创建工单",
                "description": "演示受控写操作能力，不再带审批链。",
                "tags": ["guarded", "write", "ticket", "work-order"],
                "instructions": ["该能力属于受控写操作，执行前需要明确用户意图。"],
                "actions": [
                    {
                        "kind": "tool_call",
                        "priority": 20,
                        "config": {
                            "tools": [{"tool_name": "create_work_order"}],
                            "stop_after_terminal": True,
                        },
                    },
                ],
            },
        )

    def _resolve_action_config(
        self,
        action: ResourceAction,
        *,
        selection: CapabilitySelection | None,
    ):
        if selection is None:
            return action.config
        overrides = selection.action_config_overrides
        if not overrides:
            return action.config

        override_payload = self._resolve_override_payload(action, overrides)
        if override_payload is None:
            return action.config

        base_payload = deepcopy(action.config.model_dump(mode="json"))
        merged_payload = self._deep_merge_dicts(
            base_payload,
            override_payload.config,
        )
        if action.kind == ChatActionKindEnum.tool_call and override_payload.tool_args:
            merged_payload = self._apply_tool_args_override(merged_payload, override_payload.tool_args)
        return self.action_registry.parse_config(action.kind, merged_payload)

    def _resolve_override_payload(
        self,
        action: ResourceAction,
        overrides: dict[str, ActionConfigOverride],
    ) -> ActionConfigOverride | None:
        for key in [action.action_id, action.name]:
            if not key:
                continue
            payload = overrides.get(key)
            if payload is not None:
                return payload
        return None

    def _apply_tool_args_override(
        self,
        payload: dict[str, Any],
        tool_args_override: dict[str, Any],
    ) -> dict[str, Any]:
        tools = payload.get("tools")
        if not isinstance(tools, list):
            return payload
        next_payload = deepcopy(payload)
        next_tools: list[dict[str, Any]] = []
        for item in tools:
            if not isinstance(item, dict):
                next_tools.append(item)
                continue
            tool_name = item.get("tool_name")
            override_args = tool_args_override.get(tool_name) if isinstance(tool_name, str) else None
            if not isinstance(override_args, dict):
                next_tools.append(item)
                continue
            next_tools.append(
                {
                    **item,
                    "args": self._deep_merge_dicts(
                        item.get("args") if isinstance(item.get("args"), dict) else {},
                        override_args,
                    ),
                },
            )
        next_payload["tools"] = next_tools
        return next_payload

    def _deep_merge_dicts(
        self,
        base: dict[str, Any],
        override: dict[str, Any],
    ) -> dict[str, Any]:
        merged = deepcopy(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._deep_merge_dicts(merged[key], value)
                continue
            merged[key] = deepcopy(value)
        return merged

    def build_ready_query(self) -> CapabilityPackageQuery:
        return CapabilityPackageQuery(is_enabled=True)
