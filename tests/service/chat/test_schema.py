from datetime import UTC, datetime

from pydantic import ValidationError

from service.chat.domain.schema import (
    AckPayload,
    CapabilitySelection,
    ClientCommand,
    ClientCommandEnum,
    ChatActionKindEnum,
    CapabilityKindEnum,
    ChatErrorCodeEnum,
    ChatEvent,
    ChatWarningCodeEnum,
    DataEventPayload,
    ErrorPayload,
    ExtensionEventStageEnum,
    ExtensionEventPayload,
    FunctionCallConfig,
    IntentDetectionConfig,
    MessageBundlePayload,
    ResourceSelection,
    RetrievalBlock,
    RetrievalListPayload,
    StepIOPayload,
    StepIOPhaseEnum,
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


def test_extension_event_payload_supports_nested_step_data_validation():
    event = ChatEvent[DataEventPayload[ExtensionEventPayload]].model_validate(
        {
            "id": "evt_2",
            "session_id": "session_1",
            "conversation_id": 1,
            "turn_id": 2,
            "seq": 4,
            "event": "data.created",
            "ts": "2026-03-17T12:00:01Z",
            "payload": {
                "data": {
                    "id": 12,
                    "turn_id": 2,
                    "step_id": 6,
                    "kind": "intermediate",
                    "payload_type": "extension_event",
                    "payload": {
                        "type": "extension_event",
                        "extension_key": "knowledge_base_retrieval",
                        "capability_id": 9,
                        "action_id": "capability:9:action:1",
                        "action_name": "knowledge_base_retrieval",
                        "stage": "reranking",
                        "level": "info",
                        "message": "正在 rerank 检索结果",
                        "data": {"candidate_count": 12},
                    },
                },
            },
        },
    )

    assert event.payload.data.payload.stage == ExtensionEventStageEnum.reranking
    assert event.payload.data.payload.extension_key == "knowledge_base_retrieval"


def test_step_io_payload_uses_enum_and_strict_shape():
    payload = StepIOPayload.model_validate(
        {
            "type": "step_io",
            "phase": "input",
            "action_id": "capability:1:action:1",
            "action_name": "knowledge_base_retrieval",
            "action_kind": "knowledge_retrieval",
            "message": "知识库检索输入",
            "data": {"collection_ids": [1], "top_k": 5},
        },
    )

    assert payload.phase == StepIOPhaseEnum.input


def test_resource_selection_constraints():
    selection = ResourceSelection.model_validate(
        {
            "capabilities": [{"capability_key": "grounded_qa", "kind": "skill"}],
            "actions": [
                {
                    "kind": "knowledge_retrieval",
                    "priority": 10,
                    "config": {"collection_ids": [1, 2], "top_k": 4},
                },
            ],
        },
    )

    normalized = selection.normalized_actions()
    normalized_capabilities = selection.normalized_capabilities()
    assert normalized_capabilities == [
        CapabilitySelection(capability_key="grounded_qa", kind=CapabilityKindEnum.skill),
    ]
    assert normalized[0].kind == ChatActionKindEnum.knowledge_retrieval
    assert normalized[0].config.collection_ids == [1, 2]
    assert normalized[1].kind == ChatActionKindEnum.llm_response


def test_resource_selection_rejects_unknown_action_kind():
    try:
        ResourceSelection.model_validate(
            {
                "actions": [
                    {
                        "kind": "unknown_action",
                        "priority": 10,
                        "config": {},
                    },
                ],
            },
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("expected validation failure for invalid action kind")


def test_resource_selection_uses_explicit_action_execution_order():
    selection = ResourceSelection.model_validate(
        {
            "actions": [
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

    normalized = selection.normalized_actions()

    assert [item.kind for item in normalized] == [
        ChatActionKindEnum.system_prompt,
        ChatActionKindEnum.intent_detection,
        ChatActionKindEnum.knowledge_retrieval,
        ChatActionKindEnum.function_call,
        ChatActionKindEnum.llm_response,
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

    normalized = selection.normalized_actions()

    assert len(normalized) == 1
    assert normalized[0].kind == ChatActionKindEnum.llm_response


def test_parse_resource_selection_ignores_unknown_legacy_keys():
    selection = parse_resource_selection(
        {
            "metadata": {"source": "saved"},
            "capabilities": [
                {"capability_id": 3, "kind": "extension"},
            ],
            "capability_profile_ids": [7],
            "capability_binding_ids": [3],
            "actions": [
                {
                    "kind": "function_call",
                    "priority": 10,
                    "config": {"tools": [{"tool_name": "session_context"}]},
                },
            ],
        },
    )

    assert selection.metadata == {"source": "saved"}
    assert selection.normalized_capabilities() == [
        CapabilitySelection(capability_id=3, kind=CapabilityKindEnum.extension),
    ]
    assert "capability_profile_ids" not in selection.model_dump(mode="json")
    assert "capability_binding_ids" not in selection.model_dump(mode="json")
    assert [item.kind for item in selection.normalized_actions()] == [
        ChatActionKindEnum.function_call,
        ChatActionKindEnum.llm_response,
    ]


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
