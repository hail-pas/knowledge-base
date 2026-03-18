from datetime import UTC, datetime

import pytest

from core.types import ApiException
from service.chat.domain.schema import ChatCapabilityKindEnum, ResourceSelection
from service.chat.store.repository import ChatRepository
from service.chat.capability.registry import create_default_capability_registry
from service.chat.capability.service import ChatCapabilityService
from service.chat.capability.repository import ChatCapabilityRepository
from service.chat.capability.schema import (
    CapabilityBindingCreate,
    CapabilityProfileCreate,
    ChatCapabilityBindingOwnerEnum,
)
from ext.ext_tortoise.enums import ChatTurnStatusEnum, ChatTurnTriggerEnum


@pytest.mark.asyncio
async def test_resolve_resource_selection_merges_system_conversation_and_request_sources() -> None:
    chat_repository = ChatRepository()
    registry = create_default_capability_registry()
    capability_service = ChatCapabilityService(
        registry,
        repository=ChatCapabilityRepository(),
        chat_repository=chat_repository,
    )

    conversation = await chat_repository.create_conversation(
        title="capability-conversation",
        user_id=1,
        resource_selection=ResourceSelection.model_validate(
            {
                "capabilities": [
                    {
                        "capability_id": "conv:retrieval",
                        "kind": "knowledge_retrieval",
                        "priority": 20,
                        "config": {"collection_ids": [11], "top_k": 3},
                    },
                ],
            },
        ),
    )

    system_profile = await capability_service.create_profile(
        CapabilityProfileCreate(
            name="system-tool",
            kind=ChatCapabilityKindEnum.tool_call,
            config={"policy": "stub", "tool_names": ["weather"]},
        ),
    )
    request_profile = await capability_service.create_profile(
        CapabilityProfileCreate(
            name="request-retrieval",
            kind=ChatCapabilityKindEnum.knowledge_retrieval,
            config={"collection_ids": [22], "top_k": 4},
        ),
    )
    conversation_profile = await capability_service.create_profile(
        CapabilityProfileCreate(
            name="conversation-retrieval",
            kind=ChatCapabilityKindEnum.knowledge_retrieval,
            config={"collection_ids": [33], "top_k": 2},
        ),
    )
    system_binding = await capability_service.create_binding(
        CapabilityBindingCreate(
            owner_type=ChatCapabilityBindingOwnerEnum.system,
            capability_profile_id=system_profile.id,
            priority=5,
        ),
    )
    conversation_binding = await capability_service.create_binding(
        CapabilityBindingCreate(
            owner_type=ChatCapabilityBindingOwnerEnum.conversation,
            owner_id=conversation.id,
            capability_profile_id=conversation_profile.id,
            priority=15,
        ),
    )

    resolved = await capability_service.resolve_resource_selection(
        conversation=conversation,
        request_selection=ResourceSelection.model_validate(
            {
                "capability_profile_ids": [request_profile.id],
                "capabilities": [
                    {
                        "capability_id": "request:tool",
                        "kind": "tool_call",
                        "priority": 50,
                        "config": {"policy": "optional", "tool_names": ["search"]},
                    },
                ],
            },
        ),
    )

    normalized = resolved.normalized_capabilities()

    assert [item.capability_id for item in normalized] == [
        f"binding:{system_binding.id}",
        f"binding:{conversation_binding.id}",
        "conv:retrieval",
        "request:tool",
        f"profile:{request_profile.id}",
        "builtin:llm_response",
    ]
    assert normalized[0].source == "binding:system"
    assert normalized[1].source == "binding:conversation"
    request_profile_capability = next(
        item for item in normalized if item.capability_id == f"profile:{request_profile.id}"
    )
    assert request_profile_capability.profile_id == request_profile.id


@pytest.mark.asyncio
async def test_finalize_turn_clears_active_turn_and_sets_head_on_completion() -> None:
    repository = ChatRepository()
    conversation = await repository.create_conversation(
        title="turn-finalize",
        user_id=1,
        resource_selection=ResourceSelection(),
    )
    turn = await repository.create_turn(
        conversation_id=conversation.id,
        request_id="req-complete",
        trigger=ChatTurnTriggerEnum.user,
        resource_selection=ResourceSelection(),
    )

    await repository.finalize_turn(
        turn,
        status=ChatTurnStatusEnum.completed,
        finished_at=datetime.now(UTC),
        output_root_data_id=123,
        usage={"requests": 1},
        set_head=True,
    )
    refreshed = await repository.get_conversation(conversation.id)

    assert refreshed is not None
    assert refreshed.active_turn_id is None
    assert refreshed.head_turn_id == turn.id


@pytest.mark.asyncio
async def test_finalize_turn_failure_keeps_previous_head_turn() -> None:
    repository = ChatRepository()
    conversation = await repository.create_conversation(
        title="turn-failure",
        user_id=1,
        resource_selection=ResourceSelection(),
    )
    first_turn = await repository.create_turn(
        conversation_id=conversation.id,
        request_id="req-1",
        trigger=ChatTurnTriggerEnum.user,
        resource_selection=ResourceSelection(),
    )
    await repository.finalize_turn(
        first_turn,
        status=ChatTurnStatusEnum.completed,
        finished_at=datetime.now(UTC),
        output_root_data_id=1,
        usage={"requests": 1},
        set_head=True,
    )

    second_turn = await repository.create_turn(
        conversation_id=conversation.id,
        request_id="req-2",
        trigger=ChatTurnTriggerEnum.user,
        resource_selection=ResourceSelection(),
    )
    await repository.finalize_turn(
        second_turn,
        status=ChatTurnStatusEnum.failed,
        finished_at=datetime.now(UTC),
        error_message="boom",
        set_head=False,
    )
    refreshed = await repository.get_conversation(conversation.id)

    assert refreshed is not None
    assert refreshed.active_turn_id is None
    assert refreshed.head_turn_id == first_turn.id


@pytest.mark.asyncio
async def test_create_turn_rejects_concurrent_active_turns() -> None:
    repository = ChatRepository()
    conversation = await repository.create_conversation(
        title="turn-concurrency",
        user_id=1,
        resource_selection=ResourceSelection(),
    )
    await repository.create_turn(
        conversation_id=conversation.id,
        request_id="req-active",
        trigger=ChatTurnTriggerEnum.user,
        resource_selection=ResourceSelection(),
    )

    with pytest.raises(ApiException, match="当前会话已有进行中的 turn"):
        await repository.create_turn(
            conversation_id=conversation.id,
            request_id="req-second",
            trigger=ChatTurnTriggerEnum.user,
            resource_selection=ResourceSelection(),
        )
