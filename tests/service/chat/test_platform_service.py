import pytest

from service.chat.agent.schema import AgentProfileCreate, AgentProfileManifest
from service.chat.agent.service import ChatAgentService
from service.chat.capability.service import ChatCapabilityService
from service.chat.domain.schema import (
    AgentRoleEnum,
    CapabilityCategoryEnum,
    CapabilityKindEnum,
    CapabilityPlannerConfig,
    CapabilityPlannerModeEnum,
    CapabilityRuntimeKindEnum,
    CapabilitySelection,
    ChatActionKindEnum,
    ResourceSelection,
)
from service.chat.execution.registry import create_default_action_registry
from service.chat.platform_service import ChatPlatformService


@pytest.mark.asyncio
async def test_platform_service_keeps_explicit_delegate_agent_mount_as_capability() -> None:
    action_registry = create_default_action_registry()
    capability_service = ChatCapabilityService(action_registry=action_registry)
    agent_service = ChatAgentService(action_registry=action_registry)
    platform_service = ChatPlatformService(
        capability_service=capability_service,
        agent_service=agent_service,
        action_registry=action_registry,
    )

    resolved = await platform_service.resolve_turn_resources(
        query="请计算 9 * (2 + 1)",
        requested_agent_key=None,
        conversation_selection=ResourceSelection(),
        request_selection=ResourceSelection(
            capabilities=[
                CapabilitySelection(capability_key=agent_service.DEFAULT_SPECIALIST_CAPABILITY_KEY),
            ],
        ),
        account_id=1,
        is_staff=True,
    )

    assert resolved.agent.agent_key == agent_service.DEFAULT_ORCHESTRATOR_KEY
    assert any(
        action.kind == ChatActionKindEnum.sub_agent_call
        and action.metadata.delegation is not None
        and action.metadata.delegation.mount_id is not None
        for action in resolved.resource_selection.actions
    )


@pytest.mark.asyncio
async def test_platform_service_keeps_guarded_capability_as_regular_selected_capability() -> None:
    action_registry = create_default_action_registry()
    capability_service = ChatCapabilityService(action_registry=action_registry)
    agent_service = ChatAgentService(action_registry=action_registry)
    platform_service = ChatPlatformService(
        capability_service=capability_service,
        agent_service=agent_service,
        action_registry=action_registry,
    )

    resolved = await platform_service.resolve_turn_resources(
        query="请创建工单处理 VPN 权限问题",
        requested_agent_key=None,
        conversation_selection=ResourceSelection(),
        request_selection=ResourceSelection(
            capabilities=[
                CapabilitySelection(
                    capability_key=capability_service.BUILTIN_GUARDED_WORK_ORDER_KEY,
                    kind=CapabilityKindEnum.skill,
                ),
            ],
        ),
        account_id=1,
        is_staff=True,
    )

    guarded_action = next(
        item
        for item in resolved.resource_selection.actions
        if item.metadata.capability is not None
        and item.metadata.capability.capability_key == capability_service.BUILTIN_GUARDED_WORK_ORDER_KEY
    )

    assert guarded_action.metadata.capability is not None
    assert guarded_action.metadata.capability.capability_category == CapabilityCategoryEnum.guarded
    assert (
        guarded_action.metadata.capability.capability_runtime_kind
        == CapabilityRuntimeKindEnum.local_toolset
    )


@pytest.mark.asyncio
async def test_platform_service_expands_agent_default_capability_keys() -> None:
    action_registry = create_default_action_registry()
    capability_service = ChatCapabilityService(action_registry=action_registry)
    agent_service = ChatAgentService(action_registry=action_registry)
    platform_service = ChatPlatformService(
        capability_service=capability_service,
        agent_service=agent_service,
        action_registry=action_registry,
    )

    agent = await agent_service.create_agent(
        AgentProfileCreate(
            manifest=AgentProfileManifest(
                agent_key="custom.datetime.orchestrator",
                name="Datetime Orchestrator",
                role=AgentRoleEnum.orchestrator,
                capability_keys=[capability_service.BUILTIN_CORE_DATETIME_KEY],
                default_resource_selection=ResourceSelection(),
            ),
        ),
        account_id=1,
        is_staff=True,
    )

    resolved = await platform_service.resolve_turn_resources(
        query="你好",
        requested_agent_key=agent.agent_key,
        conversation_selection=ResourceSelection(),
        request_selection=ResourceSelection(),
        account_id=1,
        is_staff=True,
    )

    assert any(
        item.metadata.capability is not None
        and item.metadata.capability.capability_key == capability_service.BUILTIN_CORE_DATETIME_KEY
        for item in resolved.resource_selection.actions
    )


@pytest.mark.asyncio
async def test_platform_service_can_disable_system_defaults_for_agent_capabilities() -> None:
    action_registry = create_default_action_registry()
    capability_service = ChatCapabilityService(action_registry=action_registry)
    agent_service = ChatAgentService(action_registry=action_registry)
    platform_service = ChatPlatformService(
        capability_service=capability_service,
        agent_service=agent_service,
        action_registry=action_registry,
    )

    agent = await agent_service.create_agent(
        AgentProfileCreate(
            manifest=AgentProfileManifest(
                agent_key="custom.no-defaults.orchestrator",
                name="No Defaults Orchestrator",
                role=AgentRoleEnum.orchestrator,
                capability_keys=[capability_service.BUILTIN_CORE_DATETIME_KEY],
                default_resource_selection=ResourceSelection(),
            ),
        ),
        account_id=1,
        is_staff=True,
    )

    resolved = await platform_service.resolve_turn_resources(
        query="现在几点",
        requested_agent_key=agent.agent_key,
        conversation_selection=ResourceSelection(),
        request_selection=ResourceSelection(use_system_defaults=False),
        account_id=1,
        is_staff=True,
    )

    assert all(
        item.metadata.capability is None
        or item.metadata.capability.capability_key != capability_service.BUILTIN_CORE_DATETIME_KEY
        for item in resolved.resource_selection.actions
    )
    assert [item.kind.value for item in resolved.resource_selection.normalized_actions()] == ["llm_response"]


@pytest.mark.asyncio
async def test_platform_service_does_not_auto_select_mount_from_query() -> None:
    action_registry = create_default_action_registry()
    capability_service = ChatCapabilityService(action_registry=action_registry)
    agent_service = ChatAgentService(action_registry=action_registry)
    platform_service = ChatPlatformService(
        capability_service=capability_service,
        agent_service=agent_service,
        action_registry=action_registry,
    )

    resolved = await platform_service.resolve_turn_resources(
        query="请计算 9 * (2 + 1)",
        requested_agent_key=None,
        conversation_selection=ResourceSelection(),
        request_selection=ResourceSelection(),
        account_id=1,
        is_staff=True,
    )

    assert all(item.kind != ChatActionKindEnum.sub_agent_call for item in resolved.resource_selection.actions)


@pytest.mark.asyncio
async def test_platform_service_keeps_explicit_mount_selection_when_system_defaults_off() -> None:
    action_registry = create_default_action_registry()
    capability_service = ChatCapabilityService(action_registry=action_registry)
    agent_service = ChatAgentService(action_registry=action_registry)
    platform_service = ChatPlatformService(
        capability_service=capability_service,
        agent_service=agent_service,
        action_registry=action_registry,
    )

    resolved = await platform_service.resolve_turn_resources(
        query="你好",
        requested_agent_key=None,
        conversation_selection=ResourceSelection(),
        request_selection=ResourceSelection(
            use_system_defaults=False,
            capabilities=[
                CapabilitySelection(capability_key=agent_service.DEFAULT_SPECIALIST_CAPABILITY_KEY),
            ],
        ),
        account_id=1,
        is_staff=True,
    )

    assert any(item.kind == ChatActionKindEnum.sub_agent_call for item in resolved.resource_selection.actions)


def test_platform_service_merge_resource_selection_prefers_request_planner() -> None:
    action_registry = create_default_action_registry()
    capability_service = ChatCapabilityService(action_registry=action_registry)
    agent_service = ChatAgentService(action_registry=action_registry)
    platform_service = ChatPlatformService(
        capability_service=capability_service,
        agent_service=agent_service,
        action_registry=action_registry,
    )

    merged = platform_service.merge_resource_selection(
        ResourceSelection(
            planner=CapabilityPlannerConfig(
                mode=CapabilityPlannerModeEnum.llm,
                planner_model_config_id=7,
            ),
        ),
        ResourceSelection(),
        ResourceSelection(
            planner=CapabilityPlannerConfig(
                mode=CapabilityPlannerModeEnum.disabled,
            ),
        ),
    )

    assert merged.planner == CapabilityPlannerConfig(mode=CapabilityPlannerModeEnum.disabled)
