from types import SimpleNamespace
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import TypeAdapter, ValidationError

from service.chat.domain.schema import (
    ActionContextItem,
    ActionConfigOverride,
    ActionMetadata,
    CapabilityPlannerConfig,
    CapabilityCategoryEnum,
    CapabilityKindEnum,
    CapabilityPlannerModeEnum,
    CapabilityRuntimeKindEnum,
    CapabilitySelection,
    ChatActionKindEnum,
    ChatErrorCodeEnum,
    ChatEvent,
    ChatRequestContext,
    ChatWarningCodeEnum,
    ErrorPayload,
    EventNameEnum,
    MCPResultPayload,
    MessageBundlePayload,
    ResourceSelection,
    ProgressPayload,
    StepEventPayload,
    StepIOPayload,
    StepIOPhaseEnum,
    StrictModel,
    SystemPromptConfig,
    TextBlock,
    ToolCallConfig,
    ToolResultPayload,
    ToolSpec,
    WarningPayload,
    parse_client_command,
    parse_resource_selection,
)
from service.chat.runtime.planning import RuntimeCapabilityDescriptor


def test_message_bundle_text_projection() -> None:
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


def test_step_event_payload_supports_generic_tool_result_content() -> None:
    event = ChatEvent[StepEventPayload].model_validate(
        {
            "id": "evt_1",
            "session_id": str(uuid4()),
            "conversation_id": 1,
            "turn_id": 2,
            "seq": 3,
            "event": EventNameEnum.step_completed.value,
            "ts": "2026-03-17T12:00:00Z",
            "payload": {
                "step": {
                    "id": 11,
                    "conversation_id": 1,
                    "turn_id": 2,
                    "kind": "tool",
                    "name": "knowledge_base_search",
                    "status": "completed",
                    "sequence": 20,
                },
                "data": {
                    "id": 12,
                    "conversation_id": 1,
                    "turn_id": 2,
                    "step_id": 11,
                    "kind": "output",
                    "payload_type": "tool_result",
                    "payload": {
                        "type": "tool_result",
                        "tool_name": "custom_tool",
                        "disposition": "context",
                        "content": [
                            "alpha",
                            {"nested": {"count": 2}},
                        ],
                    },
                },
            },
        },
    )

    assert event.payload.data is not None
    assert event.payload.data.payload.content == ["alpha", {"nested": {"count": 2}}]


def test_tool_result_payload_supports_typed_generic_content() -> None:
    class CalculationResult(StrictModel):
        expression: str
        result: int

    payload = ToolResultPayload[CalculationResult].model_validate(
        {
            "type": "tool_result",
            "tool_name": "calculate_expression",
            "disposition": "terminal",
            "summary": "计算结果：6 * 7 = 42",
            "content": {"expression": "6 * 7", "result": 42},
            "terminal": True,
        },
    )

    assert payload.content == CalculationResult(expression="6 * 7", result=42)


def test_mcp_result_payload_supports_typed_generic_content() -> None:
    class CatalogItem(StrictModel):
        id: int
        name: str

    payload = MCPResultPayload[list[CatalogItem]].model_validate(
        {
            "type": "mcp_result",
            "server_name": "knowledge_base",
            "tool_name": "collection_catalog",
            "summary": "返回 1 条 MCP 结果",
            "content": [{"id": 1, "name": "公开集合A"}],
        },
    )

    assert payload.content == [CatalogItem(id=1, name="公开集合A")]


def test_tool_spec_supports_typed_generic_args() -> None:
    class SearchArgs(StrictModel):
        collection_ids: list[int]
        top_k: int

    spec = ToolSpec[SearchArgs].model_validate(
        {
            "tool_name": "knowledge_base_search",
            "args": {"collection_ids": [1, 2], "top_k": 4},
        },
    )

    assert spec.args == SearchArgs(collection_ids=[1, 2], top_k=4)


def test_action_context_item_rejects_mismatched_variant_shape() -> None:
    try:
        TypeAdapter(ActionContextItem).validate_python(
            {
                "action_id": "act:1",
                "action_kind": "tool_call",
                "action_name": "knowledge_base_search",
                "source": "builtin",
                "item_type": "text",
                "retrievals": [
                    {
                        "type": "retrieval_hit",
                        "source_id": "chunk_1",
                        "collection_id": 1,
                        "score": 0.9,
                        "snippet": "命中片段",
                    },
                ],
            },
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("expected validation failure for invalid context item variant")


def test_step_io_payload_uses_enum_and_strict_shape() -> None:
    class ToolStepInput(StrictModel):
        tool_name: str
        tool_args: dict[str, int | list[int]]

    payload = StepIOPayload[ToolStepInput].model_validate(
        {
            "type": "step_io",
            "phase": "input",
            "action_id": "capability:1:action:1",
            "action_name": "knowledge_base_search",
            "action_kind": "tool_call",
            "message": "知识库检索工具输入",
            "data": {"tool_name": "knowledge_base_search", "tool_args": {"collection_ids": [1], "top_k": 5}},
        },
    )

    assert payload.phase == StepIOPhaseEnum.input
    assert payload.data == ToolStepInput(
        tool_name="knowledge_base_search",
        tool_args={"collection_ids": [1], "top_k": 5},
    )


def test_progress_payload_supports_typed_generic_data() -> None:
    class ProgressData(StrictModel):
        collection_id: int
        hit_count: int

    payload = ProgressPayload[ProgressData].model_validate(
        {
            "stage": "collection_completed",
            "message": "collection 1 检索完成",
            "level": "info",
            "data": {"collection_id": 1, "hit_count": 3},
        },
    )

    assert payload.data == ProgressData(collection_id=1, hit_count=3)


def test_resource_selection_accepts_explicit_planner_config() -> None:
    selection = ResourceSelection.model_validate(
        {
            "planner": {
                "mode": "llm",
                "planner_model_config_id": 9,
            },
        },
    )

    assert selection.planner == CapabilityPlannerConfig(
        mode=CapabilityPlannerModeEnum.llm,
        planner_model_config_id=9,
    )


def test_action_config_override_supports_explicit_tool_args_patch() -> None:
    override = ActionConfigOverride.model_validate(
        {
            "tool_args": {
                "knowledge_base_search": {
                    "collection_ids": [1, 2],
                    "top_k": 4,
                },
            },
        },
    )

    assert override.tool_args["knowledge_base_search"]["collection_ids"] == [1, 2]


def test_resource_action_metadata_uses_explicit_nested_schema() -> None:
    selection = ResourceSelection.model_validate(
        {
            "actions": [
                {
                    "kind": "sub_agent_call",
                    "name": "delegate.writer",
                    "metadata": {
                        "capability": {
                            "capability_key": "writer_delegate",
                            "capability_category": "agent",
                            "capability_runtime_kind": "agent_delegate",
                        },
                        "delegation": {
                            "mounted_agent_key": "agent.writer",
                            "output_contract": "terminal_text",
                        },
                        "runtime": {
                            "step_name": "writer_delegate_step",
                        },
                    },
                    "config": {
                        "system_prompt": "你是代理。",
                    },
                },
            ],
        },
    )

    metadata = selection.actions[0].metadata

    assert isinstance(metadata, ActionMetadata)
    assert metadata.capability is not None
    assert metadata.capability.capability_category == CapabilityCategoryEnum.agent
    assert metadata.capability.capability_runtime_kind == CapabilityRuntimeKindEnum.agent_delegate
    assert metadata.delegation is not None
    assert metadata.delegation.mounted_agent_key == "agent.writer"
    assert metadata.runtime.step_name == "writer_delegate_step"


def test_resource_action_metadata_rejects_legacy_flat_shape() -> None:
    try:
        ResourceSelection.model_validate(
            {
                "actions": [
                    {
                        "kind": "sub_agent_call",
                        "metadata": {
                            "output_contract": "terminal_text",
                        },
                        "config": {
                            "system_prompt": "你是代理。",
                        },
                    },
                ],
            },
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("expected validation failure for legacy flat action metadata")


def test_capability_selection_rejects_legacy_metadata_field() -> None:
    try:
        CapabilitySelection.model_validate(
            {
                "capability_key": "grounded_qa",
                "metadata": {"legacy": True},
            },
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("expected validation failure for legacy capability metadata")


def test_resource_selection_rejects_legacy_metadata_field() -> None:
    try:
        ResourceSelection.model_validate(
            {
                "metadata": {"legacy": True},
            },
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("expected validation failure for legacy resource metadata")


def test_capability_selection_rejects_empty_action_override() -> None:
    try:
        CapabilitySelection.model_validate(
            {
                "capability_key": "grounded_qa",
                "action_config_overrides": {
                    "knowledge_base_search": {},
                },
            },
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("expected validation failure for empty action override")


def test_resource_selection_constraints() -> None:
    selection = ResourceSelection.model_validate(
        {
            "capabilities": [{"capability_key": "grounded_qa", "kind": "skill"}],
            "actions": [
                {
                    "kind": "tool_call",
                    "priority": 10,
                    "config": {
                        "tools": [
                            {
                                "tool_name": "knowledge_base_search",
                                "args": {"collection_ids": [1, 2], "top_k": 4},
                            },
                        ],
                    },
                },
            ],
        },
    )

    normalized = selection.normalized_actions()
    normalized_capabilities = selection.normalized_capabilities()
    assert normalized_capabilities == [
        CapabilitySelection(capability_key="grounded_qa", kind=CapabilityKindEnum.skill),
    ]
    assert normalized[0].kind == ChatActionKindEnum.tool_call
    assert normalized[0].config.tools[0].args["collection_ids"] == [1, 2]
    assert normalized[1].kind == ChatActionKindEnum.llm_response


def test_resource_selection_rejects_unknown_action_kind() -> None:
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


def test_resource_selection_uses_explicit_action_execution_order() -> None:
    selection = ResourceSelection.model_validate(
        {
            "actions": [
                {
                    "kind": "tool_call",
                    "priority": 10,
                    "config": {
                        "tools": [
                            {
                                "tool_name": "knowledge_base_search",
                                "args": {"collection_ids": [1], "top_k": 2},
                            },
                        ],
                    },
                },
                {"kind": "system_prompt", "priority": 10, "config": {}},
                {"kind": "tool_call", "priority": 10, "config": {"tools": [{"tool_name": "session_context"}]}},
                {"kind": "llm_response", "priority": 10, "config": {}},
            ],
        },
    )

    normalized = selection.normalized_actions()

    assert [item.kind for item in normalized] == [
        ChatActionKindEnum.system_prompt,
        ChatActionKindEnum.tool_call,
        ChatActionKindEnum.tool_call,
        ChatActionKindEnum.llm_response,
    ]
    assert isinstance(normalized[0].config, SystemPromptConfig)
    assert isinstance(normalized[1].config, ToolCallConfig)
    assert isinstance(normalized[2].config, ToolCallConfig)


def test_resource_selection_uses_last_matching_capability_selection() -> None:
    selection = ResourceSelection(
        capabilities=[
            CapabilitySelection(capability_key="grounded_qa"),
            CapabilitySelection(
                capability_key="grounded_qa",
                required=True,
                action_config_overrides={
                    "knowledge_base_search": ActionConfigOverride(
                        tool_args={
                            "knowledge_base_search": {"top_k": 8},
                        },
                    ),
                },
            ),
        ],
    )

    normalized = selection.normalized_capabilities()

    assert len(normalized) == 1
    assert normalized[0].required is True
    override = normalized[0].action_config_overrides["knowledge_base_search"]
    assert override.tool_args["knowledge_base_search"]["top_k"] == 8


def test_resource_selection_uses_last_matching_action_by_action_id() -> None:
    selection = ResourceSelection.model_validate(
        {
            "actions": [
                {
                    "action_id": "request:inline:tool",
                    "kind": "tool_call",
                    "priority": 10,
                    "config": {
                        "tools": [{"tool_name": "knowledge_base_search", "args": {"top_k": 3}}],
                    },
                },
                {
                    "action_id": "request:inline:tool",
                    "kind": "tool_call",
                    "priority": 20,
                    "config": {
                        "tools": [{"tool_name": "knowledge_base_search", "args": {"top_k": 8}}],
                    },
                },
            ],
        },
    )

    normalized = selection.normalized_actions()
    tool_action = next(item for item in normalized if item.action_id == "request:inline:tool")

    assert tool_action.priority == 20
    assert tool_action.config.tools[0].args["top_k"] == 8


def test_runtime_capability_descriptor_rejects_legacy_metadata_field() -> None:
    try:
        RuntimeCapabilityDescriptor.model_validate(
            {
                "capability_key": "grounded_qa",
                "capability_kind": "extension",
                "category": "infra",
                "runtime_kind": "local_toolset",
                "name": "Grounded QA",
                "metadata": {"legacy": True},
            },
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("expected validation failure for legacy descriptor metadata")


def test_message_bundle_rejects_unknown_block_shape() -> None:
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


def test_error_and_warning_payload_codes_use_enum() -> None:
    error_payload = ErrorPayload(message="boom")
    warning_payload = WarningPayload.model_validate({"message": "skip", "code": "tool_call_skipped"})

    assert error_payload.code == ChatErrorCodeEnum.chat_error
    assert warning_payload.code == ChatWarningCodeEnum.tool_call_skipped


def test_parse_resource_selection_discards_legacy_flat_payload() -> None:
    selection = parse_resource_selection(
        {
            "capability_profile_ids": [1, 2, 3],
            "capability_binding_ids": [4, 5],
            "metadata": {"legacy": True},
            "foo": "bar",
        },
    )

    assert selection == ResourceSelection()


def test_error_payload_can_be_embedded_in_event() -> None:
    event = ChatEvent[ErrorPayload](
        id="evt_error",
        event="error",
        seq=1,
        ts=datetime(2026, 3, 17, 12, 0, tzinfo=UTC),
        payload=ErrorPayload(message="boom"),
    )

    assert event.payload.code == ChatErrorCodeEnum.chat_error


def test_turn_start_request_reuses_bound_conversation_only_when_field_missing() -> None:
    missing_conversation = parse_client_command(
        {
            "command": "turn.start",
            "payload": {
                "input": {
                    "role": "user",
                    "blocks": [{"type": "text", "text": "继续当前对话"}],
                },
            },
        },
    ).payload
    explicit_null_conversation = parse_client_command(
        {
            "command": "turn.start",
            "payload": {
                "conversation_id": None,
                "input": {
                    "role": "user",
                    "blocks": [{"type": "text", "text": "创建新对话"}],
                },
            },
        },
    ).payload

    assert missing_conversation.reuse_bound_conversation(9).conversation_id == 9
    assert explicit_null_conversation.reuse_bound_conversation(9).conversation_id is None


def test_parse_client_command_rejects_extra_payload_fields() -> None:
    try:
        parse_client_command(
            {
                "command": "turn.start",
                "payload": {
                    "input": {
                        "role": "user",
                        "blocks": [{"type": "text", "text": "hello"}],
                    },
                    "unexpected": True,
                },
            },
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("expected validation failure for extra payload fields")


def test_chat_request_context_wraps_account_and_session_metadata() -> None:
    context = ChatRequestContext(
        account=SimpleNamespace(id=7, is_staff=True),
        session_id=uuid4(),
        conversation_id=3,
    )

    assert context.account_id == 7
    assert context.is_staff is True
    assert context.require_session_id() == context.session_id
    assert context.with_conversation(8).conversation_id == 8
