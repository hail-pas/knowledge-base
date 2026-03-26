import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from tortoise.exceptions import IntegrityError

from service.chat.application.service import ChatApplicationService
from service.chat.agent.service import ChatAgentService
from service.chat.capability.schema import (
    CapabilityKindEnum,
    CapabilityPackageCreate,
    CapabilityPackageQuery,
    SkillCapabilityManifest,
)
from service.chat.capability.service import ChatCapabilityService
from service.chat.execution.registry import create_default_action_registry
from service.chat.domain.schema import (
    CapabilitySelection,
    ChatRequestContext,
    ChatRoleEnum,
    MessageBundlePayload,
    ResourceSelection,
    TextBlock,
    TurnStartRequest,
)
from service.chat.store.repository import ChatRepository
from ext.ext_tortoise.models.knowledge_base import ChatData, ChatStep, ChatTurn


def _request_context(*, account_id: int = 1, is_staff: bool = True) -> ChatRequestContext:
    return ChatRequestContext(
        account=SimpleNamespace(id=account_id, is_staff=is_staff),
        session_id=uuid4(),
    )


@pytest.mark.asyncio
async def test_capability_service_exposes_builtin_kb_retrieval_extension() -> None:
    service = ChatCapabilityService(action_registry=create_default_action_registry())

    packages = await service.list_packages(CapabilityPackageQuery())

    assert any(item.capability_key == service.BUILTIN_KB_RETRIEVAL_KEY for item in packages)


@pytest.mark.asyncio
async def test_capability_service_ensure_builtin_packages_tolerates_concurrent_create() -> None:
    class RaceRepository:
        async def get_global_package_by_key(self, *, kind, capability_key):
            return None

        async def create_package(self, **data):
            raise IntegrityError

    service = ChatCapabilityService(
        repository=RaceRepository(),  # type: ignore[arg-type]
        action_registry=create_default_action_registry(),
    )

    await service.ensure_builtin_packages()


@pytest.mark.asyncio
async def test_agent_service_ensure_builtin_agents_tolerates_concurrent_create() -> None:
    class RaceRepository:
        def __init__(self) -> None:
            self._agents: dict[str, SimpleNamespace] = {}
            self._next_id = 1

        async def get_agent_by_key(self, *, agent_key, account_id, is_staff):
            return self._agents.get(agent_key)

        async def create_agent(self, **data):
            agent = SimpleNamespace(
                id=self._next_id,
                agent_key=data["agent_key"],
                name=data["name"],
            )
            self._next_id += 1
            self._agents[data["agent_key"]] = agent
            raise IntegrityError

        async def list_mounts(self, *, source_agent_id, mounted_agent_id, is_enabled):
            return []

        async def create_mount(self, **data):
            raise IntegrityError

    service = ChatAgentService(
        repository=RaceRepository(),  # type: ignore[arg-type]
        action_registry=create_default_action_registry(),
    )

    await service.ensure_builtin_agents()


@pytest.mark.asyncio
async def test_capability_service_compile_package_actions_applies_runtime_override() -> None:
    service = ChatCapabilityService(action_registry=create_default_action_registry())
    packages = await service.list_packages(service.build_ready_query())
    package = next(
        item for item in packages if item.capability_key == service.BUILTIN_KB_RETRIEVAL_KEY
    )

    retrieval_action = next(
        item
        for item in service.compile_package_actions(
            package,
            selection=CapabilitySelection(
                capability_key=service.BUILTIN_KB_RETRIEVAL_KEY,
                kind=CapabilityKindEnum.extension,
                action_config_overrides={
                    "knowledge_base_search": {
                        "tool_args": {
                            "knowledge_base_search": {
                                "collection_ids": [11, 12],
                                "top_k": 8,
                            },
                        },
                    },
                },
            ),
        )
        if item.kind.value == "tool_call"
    )

    assert retrieval_action.name == "knowledge_base_search"
    assert retrieval_action.config.tools[0].args["collection_ids"] == [11, 12]
    assert retrieval_action.config.tools[0].args["top_k"] == 8


@pytest.mark.asyncio
async def test_capability_service_compile_package_actions_marks_required_capability_actions() -> None:
    service = ChatCapabilityService(action_registry=create_default_action_registry())
    packages = await service.list_packages(service.build_ready_query())
    package = next(
        item for item in packages if item.capability_key == service.BUILTIN_KB_RETRIEVAL_KEY
    )

    retrieval_action = next(
        item
        for item in service.compile_package_actions(
            package,
            selection=CapabilitySelection(
                capability_key=service.BUILTIN_KB_RETRIEVAL_KEY,
                kind=CapabilityKindEnum.extension,
                required=True,
                action_config_overrides={
                    "knowledge_base_search": {
                        "tool_args": {
                            "knowledge_base_search": {
                                "collection_ids": [7],
                                "top_k": 3,
                            },
                        },
                    },
                },
            ),
        )
        if item.kind.value == "tool_call"
    )

    assert retrieval_action.metadata.capability is not None
    assert retrieval_action.metadata.capability.capability_required is True
    assert retrieval_action.config.tools[0].args["collection_ids"] == [7]
    assert retrieval_action.config.tools[0].args["top_k"] == 3


@pytest.mark.asyncio
async def test_application_service_executes_selected_capability_and_persists_turn_steps() -> None:
    app_service = ChatApplicationService()
    capability_key = "turn_only_calculator"
    await app_service.capability_service.create_package(
        CapabilityPackageCreate(
            manifest=SkillCapabilityManifest.model_validate(
                {
                    "kind": "skill",
                    "capability_key": capability_key,
                    "name": "专属计算流程",
                    "description": "处理数学计算表达式",
                    "instructions": ["优先使用计算能力完成结果，不需要额外解释。"],
                    "actions": [
                        {
                            "kind": "tool_call",
                            "config": {"tools": [{"tool_name": "calculate_expression"}]},
                        },
                    ],
                },
            ),
        ),
        is_staff=True,
    )

    accepted = await app_service.start_turn(
        TurnStartRequest(
            conversation_id=None,
            input=MessageBundlePayload(role=ChatRoleEnum.user, blocks=[TextBlock(text="请计算 2 + 3 * 4")]),
            resource_selection=ResourceSelection(
                capabilities=[
                    CapabilitySelection(capability_key=capability_key, kind=CapabilityKindEnum.skill),
                ],
            ),
        ),
        context=_request_context(),
        send_event=lambda event: asyncio.sleep(0),
    )
    await app_service.runtime.running_turns[accepted.turn_id].task

    turn = await ChatTurn.get(id=accepted.turn_id)
    steps = await ChatStep.filter(turn_id=accepted.turn_id, deleted_at=0).order_by("sequence")
    output = await ChatData.get(id=turn.output_root_data_id)

    assert str(turn.status) == "completed"
    assert any(step.name == "user_message" for step in steps)
    assert any(step.name == "turn_only_calculator:instructions" for step in steps)
    assert any(step.name == "calculate_expression" for step in steps)
    assert output.payload["terminal"] is True
    assert output.payload["tool_name"] == "calculate_expression"
    assert "14" in output.payload["content_text"]


@pytest.mark.asyncio
async def test_application_service_prepare_turn_keeps_request_selection_turn_local() -> None:
    app_service = ChatApplicationService()

    accepted = await app_service.prepare_turn(
        TurnStartRequest(
            conversation_id=None,
            input=MessageBundlePayload(
                role=ChatRoleEnum.user,
                blocks=[TextBlock(text="请计算 5 + 6")],
            ),
            resource_selection=ResourceSelection.model_validate(
                {
                    "actions": [
                        {
                            "kind": "tool_call",
                            "priority": 10,
                            "config": {"tools": [{"tool_name": "calculate_expression"}]},
                        },
                    ],
                },
            ),
        ),
        context=_request_context(),
    )

    conversation = await app_service.get_conversation(
        accepted.conversation.id,
        context=_request_context(),
    )
    turn = await ChatTurn.get(id=accepted.turn_id)

    assert accepted.turn_id in app_service.runtime.pending_turns
    assert accepted.turn_id not in app_service.runtime.running_turns
    assert conversation.default_resource_selection.actions == []
    assert turn is not None
    assert any(action["kind"] == "tool_call" for action in turn.resource_selection["actions"])

    await app_service.launch_prepared_turn(accepted.turn_id, send_event=lambda event: asyncio.sleep(0))
    await app_service.runtime.running_turns[accepted.turn_id].task


@pytest.mark.asyncio
async def test_application_service_executes_default_mount_delegate_as_sub_agent() -> None:
    app_service = ChatApplicationService()

    accepted = await app_service.start_turn(
        TurnStartRequest(
            conversation_id=None,
            input=MessageBundlePayload(
                role=ChatRoleEnum.user,
                blocks=[TextBlock(text="请计算 9 * (2 + 1)")],
            ),
            resource_selection=ResourceSelection(
                capabilities=[
                    CapabilitySelection(
                        capability_key=app_service.agent_service.DEFAULT_SPECIALIST_CAPABILITY_KEY,
                        kind=CapabilityKindEnum.sub_agent,
                    ),
                ],
            ),
        ),
        context=_request_context(),
        send_event=lambda event: asyncio.sleep(0),
    )
    await app_service.runtime.running_turns[accepted.turn_id].task

    turn = await ChatTurn.get(id=accepted.turn_id)
    steps = await ChatStep.filter(turn_id=accepted.turn_id, deleted_at=0).order_by("sequence")
    sub_agent_data = await ChatData.filter(
        turn_id=accepted.turn_id,
        payload_type="sub_agent_result",
        deleted_at=0,
    ).first()
    output = await ChatData.get(id=turn.output_root_data_id)

    assert str(turn.status) == "completed"
    delegate_step = next(
        step
        for step in steps
        if step.name == app_service.agent_service.DEFAULT_SPECIALIST_CAPABILITY_KEY
    )
    assert any(
        step.name == "calculate_expression" and step.parent_step_id == delegate_step.id
        for step in steps
    )
    assert sub_agent_data is not None
    assert sub_agent_data.payload["agent_key"] == app_service.agent_service.DEFAULT_SPECIALIST_KEY
    assert sub_agent_data.payload["terminal"] is True
    assert "27" in output.payload["content_text"]
