import asyncio

import pytest

from service.chat.application.service import ChatApplicationService
from service.chat.capability.router import ChatCapabilityRouter
from service.chat.capability.schema import (
    CapabilityKindEnum,
    CapabilityPackageCreate,
    CapabilityPackageQuery,
    ExtensionCapabilityManifest,
    SkillCapabilityManifest,
)
from service.chat.capability.service import ChatCapabilityService
from service.chat.execution.registry import create_default_action_registry
from service.chat.domain.schema import (
    CapabilitySelection,
    ChatRoleEnum,
    MessageBundlePayload,
    ResourceSelection,
    TextBlock,
    TurnStartRequest,
)
from service.chat.store.repository import ChatRepository
from ext.ext_tortoise.models.knowledge_base import ChatData, ChatStep, ChatTurn


@pytest.mark.asyncio
async def test_capability_service_routes_query_to_best_matching_package() -> None:
    service = ChatCapabilityService(
        action_registry=create_default_action_registry(),
        router=ChatCapabilityRouter(),
    )
    policy_capability = await service.create_package(
        CapabilityPackageCreate(
            manifest=ExtensionCapabilityManifest.model_validate(
                {
                    "kind": "extension",
                    "capability_key": "policy_lookup",
                    "name": "政策检索",
                    "description": "查询报销制度与员工政策",
                    "tags": ["policy"],
                    "routing": {"keywords": ["政策", "报销", "制度"]},
                    "actions": [
                        {
                            "kind": "knowledge_retrieval",
                            "config": {"collection_ids": [101], "top_k": 5},
                        },
                    ],
                },
            ),
        ),
    )
    calculator_capability = await service.create_package(
        CapabilityPackageCreate(
            manifest=ExtensionCapabilityManifest.model_validate(
                {
                    "kind": "extension",
                    "capability_key": "exclusive_calculator",
                    "name": "专属计算器",
                    "description": "处理数学计算表达式",
                    "routing": {"keywords": ["专属计算", "算一下"]},
                    "actions": [
                        {
                            "kind": "function_call",
                            "config": {"tools": [{"tool_name": "calculate_expression"}]},
                        },
                    ],
                },
            ),
        ),
    )

    selection, decision = await service.resolve_turn_capabilities(
        query="请帮我专属计算 12 * (7 + 3)",
        resource_selection=ResourceSelection(),
    )

    assert decision.selected_capability_ids == [calculator_capability.id]
    assert selection.metadata["selected_capability_ids"] == [calculator_capability.id]
    assert [item.kind.value for item in selection.normalized_actions()] == ["function_call", "llm_response"]
    assert policy_capability.id not in decision.selected_capability_ids


@pytest.mark.asyncio
async def test_capability_service_honors_explicit_capability_reference() -> None:
    service = ChatCapabilityService(
        action_registry=create_default_action_registry(),
        router=ChatCapabilityRouter(),
    )
    calculator_capability = await service.create_package(
        CapabilityPackageCreate(
            manifest=ExtensionCapabilityManifest.model_validate(
                {
                    "kind": "extension",
                    "capability_key": "forced_calculator",
                    "name": "强制计算器",
                    "description": "处理数学计算表达式",
                    "routing": {"keywords": ["不会命中"]},
                    "actions": [
                        {
                            "kind": "function_call",
                            "config": {"tools": [{"tool_name": "calculate_expression"}]},
                        },
                    ],
                },
            ),
        ),
    )

    selection, decision = await service.resolve_turn_capabilities(
        query="你好",
        resource_selection=ResourceSelection(
            capabilities=[
                CapabilitySelection(
                    capability_id=calculator_capability.id,
                    kind=CapabilityKindEnum.extension,
                ),
            ],
        ),
    )

    assert decision.selected_capability_ids == [calculator_capability.id]
    assert selection.normalized_capabilities()[0].capability_id == calculator_capability.id
    assert [item.kind.value for item in selection.normalized_actions()] == ["function_call", "llm_response"]


@pytest.mark.asyncio
async def test_capability_router_falls_back_to_plain_chat_when_no_packages() -> None:
    decision = await ChatCapabilityRouter().route(
        query="你好，帮我简单介绍一下你能做什么",
        packages=[],
    )

    assert decision.selected_capability_ids == []
    assert decision.summary == "当前未安装可用 capability package，已回退到基础聊天"


@pytest.mark.asyncio
async def test_capability_service_exposes_builtin_kb_retrieval_extension() -> None:
    service = ChatCapabilityService(
        action_registry=create_default_action_registry(),
        router=ChatCapabilityRouter(),
    )

    packages = await service.list_packages(CapabilityPackageQuery())

    assert any(item.capability_key == service.BUILTIN_KB_RETRIEVAL_KEY for item in packages)


@pytest.mark.asyncio
async def test_capability_service_applies_runtime_override_for_builtin_kb_retrieval() -> None:
    service = ChatCapabilityService(
        action_registry=create_default_action_registry(),
        router=ChatCapabilityRouter(),
    )

    selection, decision = await service.resolve_turn_capabilities(
        query="你好",
        resource_selection=ResourceSelection(
            capabilities=[
                CapabilitySelection(
                    capability_key=service.BUILTIN_KB_RETRIEVAL_KEY,
                    kind=CapabilityKindEnum.extension,
                    metadata={
                        "action_config_overrides": {
                            "knowledge_retrieval": {
                                "collection_ids": [11, 12],
                                "top_k": 8,
                            },
                        },
                    },
                ),
            ],
        ),
    )

    retrieval_action = next(item for item in selection.normalized_actions() if item.kind.value == "knowledge_retrieval")

    assert decision.selected_capability_ids
    assert retrieval_action.config.collection_ids == [11, 12]
    assert retrieval_action.config.top_k == 8


@pytest.mark.asyncio
async def test_capability_service_marks_required_capability_actions() -> None:
    service = ChatCapabilityService(
        action_registry=create_default_action_registry(),
        router=ChatCapabilityRouter(),
    )

    selection, _decision = await service.resolve_turn_capabilities(
        query="何为数据一致性",
        resource_selection=ResourceSelection(
            capabilities=[
                CapabilitySelection(
                    capability_key=service.BUILTIN_KB_RETRIEVAL_KEY,
                    kind=CapabilityKindEnum.extension,
                    required=True,
                    metadata={
                        "action_config_overrides": {
                            "knowledge_retrieval": {
                                "collection_ids": [7],
                                "top_k": 3,
                            },
                        },
                    },
                ),
            ],
        ),
    )

    retrieval_action = next(item for item in selection.normalized_actions() if item.kind.value == "knowledge_retrieval")

    assert retrieval_action.metadata["capability_required"] is True
    assert retrieval_action.config.collection_ids == [7]
    assert retrieval_action.config.top_k == 3


@pytest.mark.asyncio
async def test_application_service_executes_selected_capability_and_persists_capability_plan() -> None:
    app_service = ChatApplicationService()
    calculator_capability = await app_service.capability_service.create_package(
        CapabilityPackageCreate(
            manifest=SkillCapabilityManifest.model_validate(
                {
                    "kind": "skill",
                    "capability_key": "turn_only_calculator",
                    "name": "专属计算流程",
                    "description": "处理数学计算表达式",
                    "routing": {"keywords": ["回合专用计算"]},
                    "instructions": ["优先使用计算能力完成结果，不需要额外解释。"],
                    "actions": [
                        {
                            "kind": "function_call",
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
            input=MessageBundlePayload(role=ChatRoleEnum.user, blocks=[TextBlock(text="回合专用计算 2 + 3 * 4")]),
        ),
        ws_session_db_id=None,
        ws_public_session_id="ws-capability-test",
        account_id=1,
        is_staff=True,
        send_event=lambda event: asyncio.sleep(0),
    )
    await app_service.runtime.running_turns[accepted.turn_id].task

    turn = await ChatTurn.get(id=accepted.turn_id)
    steps = await ChatStep.filter(turn_id=accepted.turn_id, deleted_at=0).order_by("sequence")
    plan_data = await ChatData.filter(turn_id=accepted.turn_id, payload_type="capability_plan", deleted_at=0).first()
    output = await ChatData.get(id=turn.output_root_data_id)

    assert str(turn.status) == "completed"
    assert [step.name for step in steps] == [
        "turn_root",
        "capability_routing",
        "capability:turn_only_calculator",
        "turn_only_calculator:instructions",
        "function_call",
        "llm_response",
    ]
    assert plan_data is not None
    assert plan_data.payload["selected_capability_ids"] == [calculator_capability.id]
    assert "14" in output.payload["blocks"][0]["text"]
