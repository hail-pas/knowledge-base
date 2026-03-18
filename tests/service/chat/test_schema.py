from datetime import UTC, datetime

from pydantic import ValidationError

from service.chat.domain.schema import (
    AckPayload,
    ClientCommand,
    ClientCommandEnum,
    ChatCapabilityKindEnum,
    ChatErrorCodeEnum,
    ChatEvent,
    ChatWarningCodeEnum,
    DataEventPayload,
    ErrorPayload,
    FunctionCallConfig,
    IntentDetectionConfig,
    MessageBundlePayload,
    ResourceSelection,
    RetrievalBlock,
    RetrievalListPayload,
    SystemPromptConfig,
    TextBlock,
    TurnStartRequest,
    WarningPayload,
    parse_resource_selection,
)


def test_message_bundle_text_projection():
    payload = MessageBundlePayload.model_validate(
        {
            "type": "message_bundle",
            "role": "user",
            "blocks": [
                {"type": "text", "text": "你好"},
                {"type": "text", "text": "请总结文档"},
            ],
        },
    )

    assert payload.text == "你好\n请总结文档"


def test_generic_event_payload_strict_validation():
    event = ChatEvent[DataEventPayload[RetrievalListPayload]].model_validate(
        {
            "id": "evt_1",
            "session_id": "session_1",
            "conversation_id": 1,
            "turn_id": 2,
            "seq": 3,
            "event": "data.created",
            "ts": "2026-03-17T12:00:00Z",
            "payload": {
                "data": {
                    "id": 11,
                    "turn_id": 2,
                    "step_id": 5,
                    "kind": "reference",
                    "payload_type": "retrieval_hit_list",
                    "payload": {
                        "items": [
                            {
                                "type": "retrieval_hit",
                                "source_id": "chunk_1",
                                "collection_id": 1,
                                "document_id": 2,
                                "score": 0.9,
                                "snippet": "命中片段",
                            },
                        ],
                    },
                },
            },
        },
    )

    assert event.payload.data.payload.items[0] == RetrievalBlock(
        source_id="chunk_1",
        collection_id=1,
        document_id=2,
        score=0.9,
        snippet="命中片段",
    )


def test_resource_selection_constraints():
    selection = ResourceSelection.model_validate(
        {
            "capabilities": [
                {
                    "kind": "knowledge_retrieval",
                    "priority": 10,
                    "config": {"collection_ids": [1, 2], "top_k": 4},
                },
            ],
        },
    )

    normalized = selection.normalized_capabilities()
    assert normalized[0].kind == ChatCapabilityKindEnum.knowledge_retrieval
    assert normalized[0].config.collection_ids == [1, 2]
    assert normalized[1].kind == ChatCapabilityKindEnum.llm_response


def test_resource_selection_rejects_unknown_capability_kind():
    try:
        ResourceSelection.model_validate(
            {
                "capabilities": [
                    {
                        "kind": "unknown_capability",
                        "priority": 10,
                        "config": {},
                    },
                ],
            },
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("expected validation failure for invalid capability kind")


def test_resource_selection_uses_explicit_capability_execution_order():
    selection = ResourceSelection.model_validate(
        {
            "capabilities": [
                {"kind": "function_call", "priority": 10, "config": {"tools": [{"tool_name": "session_context"}]}},
                {
                    "kind": "knowledge_retrieval",
                    "priority": 10,
                    "config": {"collection_ids": [1], "top_k": 2},
                },
                {
                    "kind": "intent_detection",
                    "priority": 10,
                    "config": {"intents": [{"intent": "weather", "keywords": ["天气"]}]},
                },
                {"kind": "system_prompt", "priority": 10, "config": {}},
                {"kind": "llm_response", "priority": 10, "config": {}},
            ],
        },
    )

    normalized = selection.normalized_capabilities()

    assert [item.kind for item in normalized] == [
        ChatCapabilityKindEnum.system_prompt,
        ChatCapabilityKindEnum.intent_detection,
        ChatCapabilityKindEnum.knowledge_retrieval,
        ChatCapabilityKindEnum.function_call,
        ChatCapabilityKindEnum.llm_response,
    ]
    assert isinstance(normalized[0].config, SystemPromptConfig)
    assert isinstance(normalized[1].config, IntentDetectionConfig)
    assert isinstance(normalized[3].config, FunctionCallConfig)


def test_message_bundle_rejects_unknown_block_shape():
    try:
        MessageBundlePayload.model_validate(
            {
                "type": "message_bundle",
                "role": "user",
                "blocks": [{"type": "text", "bad_field": "oops"}],
            },
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("expected validation failure for invalid content block")


def test_error_payload_can_be_embedded_in_event():
    event = ChatEvent[ErrorPayload](
        id="evt_error",
        event="error",
        seq=1,
        ts=datetime(2026, 3, 17, 12, 0, tzinfo=UTC),
        payload=ErrorPayload(message="boom"),
    )

    assert event.payload.code == ChatErrorCodeEnum.chat_error


def test_error_and_warning_payload_codes_use_enum():
    error_payload = ErrorPayload(message="boom")
    warning_payload = WarningPayload.model_validate({"message": "skip", "code": "tool_call_skipped"})

    assert error_payload.code == ChatErrorCodeEnum.chat_error
    assert warning_payload.code == ChatWarningCodeEnum.tool_call_skipped


def test_parse_resource_selection_discards_legacy_flat_payload():
    selection = parse_resource_selection(
        {
            "collection_ids": [3, "5", "bad"],
            "top_k": 6,
            "llm_model_config_id": 9,
            "response_mode": "text",
            "tool_policy": "optional",
        },
    )

    normalized = selection.normalized_capabilities()

    assert len(normalized) == 1
    assert normalized[0].kind == ChatCapabilityKindEnum.llm_response


def test_parse_resource_selection_ignores_legacy_keys_when_new_shape_exists():
    selection = parse_resource_selection(
        {
            "capability_profile_ids": [7],
            "metadata": {"source": "saved"},
            "collection_ids": [3],
            "top_k": 6,
            "tool_policy": "optional",
        },
    )

    assert selection.capability_profile_ids == [7]
    assert selection.metadata == {"source": "saved"}
    assert len(selection.normalized_capabilities()) == 1
    assert selection.normalized_capabilities()[0].kind == ChatCapabilityKindEnum.llm_response


def test_client_command_uses_enum_and_typed_payload():
    command = ClientCommand[TurnStartRequest].model_validate(
        {
            "command": "turn.start",
            "payload": {
                "input": {
                    "role": "user",
                    "blocks": [{"type": "text", "text": "你好"}],
                },
            },
        },
    )

    assert command.command == ClientCommandEnum.turn_start
    assert command.payload.input.text == "你好"


def test_ack_payload_uses_command_enum():
    payload = AckPayload.model_validate({"command": "turn.start", "accepted": True})

    assert payload.command == ClientCommandEnum.turn_start
