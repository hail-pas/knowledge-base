from __future__ import annotations

from dataclasses import dataclass

from core.types import ApiException
from service.chat.agent.schema import AgentMountSummary, AgentProfileSummary
from service.chat.agent.service import ChatAgentService
from service.chat.capability.schema import (
    CapabilityPackageSummary,
    SkillCapabilityManifest,
)
from service.chat.capability.service import ChatCapabilityService
from service.chat.domain.schema import (
    ActionDelegationMetadata,
    ActionMetadata,
    CapabilityPlannerConfig,
    CapabilityCategoryEnum,
    CapabilityKindEnum,
    CapabilityRuntimeKindEnum,
    CapabilitySelection,
    ChatActionKindEnum,
    ResourceSelection,
    SubAgentCallConfig,
)
from service.chat.execution.registry import ExecutionActionRegistry, create_default_action_registry
from service.chat.runtime.planning import (
    CapabilityCatalogSnapshot,
    CapabilityPlanner,
    RuntimeCapabilityDescriptor,
    RuntimeExecutionPlan,
)


@dataclass(slots=True)
class ResolvedTurnResources:
    agent: AgentProfileSummary
    mounts: list[AgentMountSummary]
    resource_selection: ResourceSelection
    runtime_plan: RuntimeExecutionPlan
    catalog_snapshot: CapabilityCatalogSnapshot


class ChatPlatformService:
    AGENT_CAPABILITY_ID_OFFSET = 1_000_000

    def __init__(
        self,
        *,
        capability_service: ChatCapabilityService,
        agent_service: ChatAgentService,
        action_registry: ExecutionActionRegistry | None = None,
        planner: CapabilityPlanner | None = None,
    ) -> None:
        self.capability_service = capability_service
        self.agent_service = agent_service
        self.action_registry = action_registry or create_default_action_registry()
        self.planner = planner or CapabilityPlanner()

    async def resolve_turn_resources(
        self,
        *,
        query: str,
        requested_agent_key: str | None,
        conversation_selection: ResourceSelection,
        request_selection: ResourceSelection,
        account_id: int | None,
        is_staff: bool,
    ) -> ResolvedTurnResources:
        await self.capability_service.ensure_builtin_packages()
        await self.agent_service.ensure_builtin_agents()
        agent = await self.agent_service.get_agent_by_key(
            requested_agent_key or self.agent_service.DEFAULT_ORCHESTRATOR_KEY,
            account_id=account_id,
            is_staff=is_staff,
        )
        merged_selection = self.merge_resource_selection(
            self.expand_agent_default_selection(agent),
            conversation_selection,
            request_selection,
        )
        mounts = await self.agent_service.get_mounts_for_agent(agent_id=agent.id, is_enabled=True)
        catalog_snapshot = await self.build_catalog_snapshot(
            agent=agent,
            mounts=mounts,
            resource_selection=merged_selection,
            account_id=account_id,
            is_staff=is_staff,
        )
        planner_config = self.resolve_planner_config(merged_selection)
        runtime_plan = await self.planner.build_plan(
            query=query,
            catalog=catalog_snapshot,
            allow_system_defaults=merged_selection.use_system_defaults,
            planner_mode=planner_config.mode,
            planner_model_config_id=planner_config.planner_model_config_id,
        )
        resolved_selection = self.action_registry.normalize_selection(
            ResourceSelection(
                use_system_defaults=merged_selection.use_system_defaults,
                use_conversation_defaults=merged_selection.use_conversation_defaults,
                capabilities=merged_selection.normalized_capabilities(),
                actions=runtime_plan.actions,
                planner=merged_selection.planner,
            ),
        )
        runtime_plan = runtime_plan.model_copy(update={"actions": resolved_selection.actions})
        return ResolvedTurnResources(
            agent=agent,
            mounts=mounts,
            resource_selection=resolved_selection,
            runtime_plan=runtime_plan,
            catalog_snapshot=catalog_snapshot,
        )

    async def build_catalog_snapshot(
        self,
        *,
        agent: AgentProfileSummary,
        mounts: list[AgentMountSummary],
        resource_selection: ResourceSelection,
        account_id: int | None,
        is_staff: bool,
    ) -> CapabilityCatalogSnapshot:
        packages = await self.capability_service.list_packages(
            query=self.capability_service.build_ready_query(),
            account_id=account_id,
            is_staff=is_staff,
        )
        visible_packages = [
            item
            for item in packages
            if not item.visible_to_agents or agent.agent_key in set(item.visible_to_agents)
        ]
        selection_map = self.build_selection_map(resource_selection.normalized_capabilities())
        descriptors_by_key: dict[str, RuntimeCapabilityDescriptor] = {}
        for package in visible_packages:
            selection = selection_map.get(package.id) or selection_map.get(package.capability_key)
            descriptor = self.build_package_descriptor(package=package, selection=selection)
            descriptors_by_key[descriptor.capability_key] = descriptor

        mounted_agents = {
            item.id: item
            for item in await self.agent_service.list_agents_by_ids(
                [mount.mounted_agent_id for mount in mounts],
                account_id=account_id,
                is_staff=is_staff,
                is_enabled=True,
            )
        }
        for mount in mounts:
            mounted_agent = mounted_agents.get(mount.mounted_agent_id)
            if mounted_agent is None:
                raise ApiException("Agent 不存在")
            selection = (
                selection_map.get(mount.mounted_as_capability or mount.mounted_agent_key)
                or selection_map.get(self.AGENT_CAPABILITY_ID_OFFSET + mount.id)
            )
            descriptor = self.build_mount_descriptor(
                mount=mount,
                mounted_agent=mounted_agent,
                selection=selection,
            )
            descriptors_by_key[descriptor.capability_key] = descriptor

        self.validate_required_capabilities(
            descriptors=list(descriptors_by_key.values()),
            selections=resource_selection.normalized_capabilities(),
        )
        return CapabilityCatalogSnapshot(
            descriptors=list(descriptors_by_key.values()),
            inline_actions=list(resource_selection.actions),
        )

    def expand_agent_default_selection(self, agent: AgentProfileSummary) -> ResourceSelection:
        selection = agent.manifest.default_resource_selection
        capability_keys = [
            item.strip()
            for item in agent.manifest.capability_keys
            if isinstance(item, str) and item.strip()
        ]
        if not capability_keys:
            return selection
        return ResourceSelection(
            use_system_defaults=selection.use_system_defaults,
            use_conversation_defaults=selection.use_conversation_defaults,
            capabilities=[
                *selection.capabilities,
                *[
                    CapabilitySelection(capability_key=capability_key)
                    for capability_key in capability_keys
                ],
            ],
            actions=selection.actions,
            planner=selection.planner,
        )

    def merge_resource_selection(
        self,
        agent_defaults: ResourceSelection,
        conversation_selection: ResourceSelection,
        request_selection: ResourceSelection,
    ) -> ResourceSelection:
        capabilities = list(agent_defaults.capabilities) if request_selection.use_system_defaults else []
        actions = list(agent_defaults.actions) if request_selection.use_system_defaults else []
        planner = agent_defaults.planner if request_selection.use_system_defaults else None
        if request_selection.use_conversation_defaults:
            capabilities.extend(conversation_selection.capabilities)
            actions.extend(conversation_selection.actions)
            planner = conversation_selection.planner or planner
        capabilities.extend(request_selection.capabilities)
        actions.extend(request_selection.actions)
        planner = request_selection.planner or planner
        return self.action_registry.normalize_selection(
            ResourceSelection(
                use_system_defaults=request_selection.use_system_defaults,
                use_conversation_defaults=request_selection.use_conversation_defaults,
                capabilities=capabilities,
                actions=actions,
                planner=planner,
            ),
        )

    def build_package_descriptor(
        self,
        *,
        package: CapabilityPackageSummary,
        selection: CapabilitySelection | None,
    ) -> RuntimeCapabilityDescriptor:
        manifest = package.manifest
        preferred_capability_keys: list[str] = []
        if isinstance(manifest, SkillCapabilityManifest):
            preferred_capability_keys = [
                *manifest.preferred_extension_keys,
                *manifest.preferred_sub_agent_keys,
            ]
        return RuntimeCapabilityDescriptor(
            capability_id=package.id,
            capability_key=package.capability_key,
            capability_kind=package.kind,
            category=package.category,
            runtime_kind=package.runtime_kind,
            name=package.name,
            capability_version=package.version,
            description=package.description,
            tags=list(manifest.tags),
            constraints=list(manifest.constraints),
            instructions=list(manifest.instructions),
            preferred_capability_keys=[item for item in preferred_capability_keys if item],
            actions=self.capability_service.compile_package_actions(
                package,
                selection=selection,
            ),
            explicit=selection is not None,
            required=bool(selection.required) if selection is not None else False,
            always_on=manifest.always_on,
        )

    def build_mount_descriptor(
        self,
        *,
        mount: AgentMountSummary,
        mounted_agent: AgentProfileSummary,
        selection: CapabilitySelection | None,
    ) -> RuntimeCapabilityDescriptor:
        capability_key = mount.mounted_as_capability or mount.mounted_agent_key
        nested_selection = self.expand_agent_default_selection(mounted_agent)
        has_system_prompt_action = any(
            action.kind == ChatActionKindEnum.system_prompt
            for action in nested_selection.actions
        )
        return RuntimeCapabilityDescriptor(
            capability_id=self.AGENT_CAPABILITY_ID_OFFSET + mount.id,
            capability_key=capability_key,
            capability_kind=CapabilityKindEnum.sub_agent,
            category=CapabilityCategoryEnum.agent,
            runtime_kind=CapabilityRuntimeKindEnum.agent_delegate,
            name=mounted_agent.name,
            description=mount.purpose or mounted_agent.description,
            tags=[],
            constraints=[mount.output_contract] if mount.output_contract else [],
            instructions=[],
            actions=[
                self.action_registry.build_action(
                    ChatActionKindEnum.sub_agent_call,
                    config=SubAgentCallConfig(
                        llm_model_config_id=mounted_agent.manifest.llm_model_config_id,
                        system_prompt=(
                            mounted_agent.system_prompt
                            if mounted_agent.system_prompt and not has_system_prompt_action
                            else "你是被委派执行的专业代理，请完成当前子任务并返回结论。"
                        ),
                        actions=list(nested_selection.actions),
                    ),
                    action_id=f"mount:{mount.id}:delegate",
                    name=capability_key,
                    source="agent_mount",
                    priority=100,
                    metadata=ActionMetadata(
                        delegation=ActionDelegationMetadata(
                            mount_id=mount.id,
                            mounted_agent_id=mounted_agent.id,
                            mounted_agent_key=mount.mounted_agent_key,
                            mounted_agent_name=mounted_agent.name,
                            mount_mode=mount.mode,
                            output_contract=mount.output_contract,
                            pass_message_history=mount.pass_message_history,
                            pass_deps_fields=list(mount.pass_deps_fields),
                        ),
                    ),
                ),
            ],
            explicit=selection is not None,
            required=bool(selection.required) if selection is not None else False,
            always_on=False,
        )

    def build_selection_map(
        self,
        selections: list[CapabilitySelection],
    ) -> dict[int | str, CapabilitySelection]:
        mapping: dict[int | str, CapabilitySelection] = {}
        for item in selections:
            if item.capability_id is not None:
                mapping[item.capability_id] = item
            if item.capability_key is not None:
                mapping[item.capability_key] = item
        return mapping

    def validate_required_capabilities(
        self,
        *,
        descriptors: list[RuntimeCapabilityDescriptor],
        selections: list[CapabilitySelection],
    ) -> None:
        required_keys = {
            item.capability_key
            for item in selections
            if item.required and item.capability_key
        }
        visible_keys = {item.capability_key for item in descriptors}
        missing = sorted(required_keys - visible_keys)
        if missing:
            raise ApiException(f"必需 capability package 不存在或不可见: {', '.join(missing)}")

    def resolve_planner_config(self, selection: ResourceSelection) -> CapabilityPlannerConfig:
        return selection.planner or CapabilityPlannerConfig()
