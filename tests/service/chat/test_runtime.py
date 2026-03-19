import asyncio
from types import SimpleNamespace

import pytest

from service.chat.runtime.engine import ChatRuntime
from service.chat.store.repository import ChatRepository
from service.chat.domain.schema import (
    ChatRoleEnum,
    ExtensionEventPayload,
    ExtensionEventStageEnum,
    FunctionCallResultModeEnum,
    FunctionToolSpec,
    MessageBundlePayload,
    ResourceSelection,
    StepIOPayload,
    TextBlock,
    TurnStartRequest,
    UsagePayload,
)
from service.chat.runtime.function_tools import FunctionToolExecutionResult, FunctionToolRegistry
from ext.ext_tortoise.models.knowledge_base import ChatData, ChatStep, ChatTurn


class DummyRepository:
    pass


@pytest.mark.asyncio
async def test_clear_current_task_cancellation_allows_cleanup() -> None:
    runtime = ChatRuntime(repository=DummyRepository())  # type: ignore[arg-type]
    cleanup_done: list[str] = []

    async def worker() -> None:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            runtime.clear_current_task_cancellation()
            await asyncio.sleep(0)
            cleanup_done.append("done")

    task = asyncio.create_task(worker())
    await asyncio.sleep(0)
    task.cancel()
    await task

    assert cleanup_done == ["done"]


def test_usage_payload_is_json_serializable_for_checkpoint_snapshot() -> None:
    usage = UsagePayload(requests=1, input_tokens=2, output_tokens=3, total_tokens=5)

    snapshot = {"response_text": "ok", "usage": usage.model_dump(mode="json")}

    assert snapshot["usage"]["total_tokens"] == 5


@pytest.mark.asyncio
async def test_terminal_output_short_circuits_non_llm_capabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = ChatRepository()
    function_tools = FunctionToolRegistry()

    async def execute_terminal_tool(session, spec) -> FunctionToolExecutionResult:
        payload = MessageBundlePayload(
            role=ChatRoleEnum.assistant,
            blocks=[TextBlock(text="terminal answer")],
        )
        return FunctionToolExecutionResult(
            tool_name=spec.tool_name,
            title=spec.title,
            summary="return terminal output",
            text=payload.text,
            prefer_terminal=True,
            terminal_payload=payload,
        )

    function_tools.register(
        "terminal_tool",
        title="Terminal Tool",
        matcher=lambda query, session: 1.0,
        executor=execute_terminal_tool,
        default_result_mode=FunctionCallResultModeEnum.terminal,
    )
    runtime = ChatRuntime(repository=repository, function_tool_registry=function_tools)

    conversation = await repository.create_conversation(
        title="terminal-short-circuit",
        user_id=1,
        resource_selection=ResourceSelection(),
    )
    conversation_summary = await repository.summarize_conversation(conversation)

    async def fail_retrieval(**kwargs):
        raise AssertionError("retrieval action should be skipped after terminal output")

    monkeypatch.setattr(runtime.execution_manager, "retrieve_context", fail_retrieval)

    turn_id = await runtime.execute_turn(
        ws_session_id=None,
        ws_public_session_id="ws-public",
        conversation=conversation_summary,
        turn_request=TurnStartRequest(
            conversation_id=conversation.id,
            request_id="req-terminal",
                input=MessageBundlePayload(
                    role=ChatRoleEnum.user,
                    blocks=[TextBlock(text="直接给我当前答案")],
                ),
                resource_selection=ResourceSelection.model_validate(
                    {
                        "actions": [
                            {
                                "kind": "function_call",
                                "priority": 10,
                            "config": {"tools": [{"tool_name": "terminal_tool"}]},
                        },
                        {
                            "kind": "knowledge_retrieval",
                            "priority": 20,
                            "config": {"collection_ids": [999], "top_k": 1},
                        },
                    ],
                },
            ),
        ),
        account_id=1,
        is_staff=False,
        send_event=lambda event: asyncio.sleep(0),
    )

    task = runtime.running_turns[turn_id].task
    await task

    turn = await ChatTurn.get(id=turn_id)
    steps = await ChatStep.filter(turn_id=turn_id, deleted_at=0).order_by("sequence")

    assert str(turn.status) == "completed"
    assert [step.name for step in steps] == ["turn_root", "function_call", "llm_response"]


@pytest.mark.asyncio
async def test_required_capability_group_executes_before_terminal_standalone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = ChatRepository()
    function_tools = FunctionToolRegistry()

    async def execute_terminal_tool(session, spec) -> FunctionToolExecutionResult:
        payload = MessageBundlePayload(
            role=ChatRoleEnum.assistant,
            blocks=[TextBlock(text="terminal answer")],
        )
        return FunctionToolExecutionResult(
            tool_name=spec.tool_name,
            title=spec.title,
            summary="return terminal output",
            text=payload.text,
            prefer_terminal=True,
            terminal_payload=payload,
        )

    function_tools.register(
        "terminal_tool",
        title="Terminal Tool",
        matcher=lambda query, session: 1.0,
        executor=execute_terminal_tool,
        default_result_mode=FunctionCallResultModeEnum.terminal,
    )
    runtime = ChatRuntime(repository=repository, function_tool_registry=function_tools)

    conversation = await repository.create_conversation(
        title="required-capability-first",
        user_id=1,
        resource_selection=ResourceSelection(),
    )
    conversation_summary = await repository.summarize_conversation(conversation)
    called_collection_ids: list[list[int]] = []

    async def fake_retrieve_context(**kwargs):
        called_collection_ids.append(list(kwargs["collection_ids"]))
        return []

    monkeypatch.setattr(runtime.execution_manager, "retrieve_context", fake_retrieve_context)

    turn_id = await runtime.execute_turn(
        ws_session_id=None,
        ws_public_session_id="ws-public",
        conversation=conversation_summary,
        turn_request=TurnStartRequest(
            conversation_id=conversation.id,
            request_id="req-required-capability",
            input=MessageBundlePayload(
                role=ChatRoleEnum.user,
                blocks=[TextBlock(text="何为数据一致性")],
            ),
            resource_selection=ResourceSelection.model_validate(
                {
                    "actions": [
                        {
                            "kind": "function_call",
                            "priority": 10,
                            "config": {"tools": [{"tool_name": "terminal_tool"}]},
                        },
                        {
                            "kind": "knowledge_retrieval",
                            "priority": 20,
                            "name": "knowledge_base_retrieval",
                            "config": {"collection_ids": [321], "top_k": 2},
                            "metadata": {
                                "capability_id": 1,
                                "capability_key": "knowledge_base_retrieval",
                                "capability_name": "知识库检索",
                                "capability_kind": "extension",
                                "capability_order": 1,
                                "capability_required": True,
                            },
                        },
                    ],
                },
            ),
            metadata={
                "capability_plan": {
                    "mode": "heuristic",
                    "summary": "required capability first",
                    "selected_capability_ids": [1],
                    "candidates": [
                        {
                            "capability_id": 1,
                            "capability_key": "knowledge_base_retrieval",
                            "capability_kind": "extension",
                            "score": 1.0,
                            "selected": True,
                            "reasons": ["required by selection"],
                        },
                    ],
                },
            },
        ),
        account_id=1,
        is_staff=False,
        send_event=lambda event: asyncio.sleep(0),
    )

    await runtime.running_turns[turn_id].task

    turn = await ChatTurn.get(id=turn_id)
    steps = await ChatStep.filter(turn_id=turn_id, deleted_at=0).order_by("sequence")

    assert str(turn.status) == "completed"
    assert called_collection_ids == [[321]]
    assert [step.name for step in steps] == [
        "turn_root",
        "capability_routing",
        "capability:knowledge_base_retrieval",
        "knowledge_base_retrieval",
        "function_call",
        "llm_response",
    ]
    retrieval_step = next(step for step in steps if step.name == "knowledge_base_retrieval")
    assert retrieval_step.input_data_ids


@pytest.mark.asyncio
async def test_required_retrieval_fails_when_no_collection_is_actually_searched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = ChatRepository()
    runtime = ChatRuntime(repository=repository)

    conversation = await repository.create_conversation(
        title="required-retrieval-unavailable",
        user_id=1,
        resource_selection=ResourceSelection(),
    )
    conversation_summary = await repository.summarize_conversation(conversation)
    events = []

    async def fake_retrieve_context(**kwargs):
        return {
            "retrievals": [],
            "requested_collection_ids": [321],
            "searched_collection_ids": [],
            "missing_collection_ids": [321],
            "inaccessible_collection_ids": [],
            "failed_collection_ids": [],
        }

    async def collect_event(event) -> None:
        events.append(event)

    monkeypatch.setattr(runtime.execution_manager, "retrieve_context", fake_retrieve_context)

    turn_id = await runtime.execute_turn(
        ws_session_id=None,
        ws_public_session_id="ws-public",
        conversation=conversation_summary,
        turn_request=TurnStartRequest(
            conversation_id=conversation.id,
            request_id="req-required-retrieval-unavailable",
            input=MessageBundlePayload(
                role=ChatRoleEnum.user,
                blocks=[TextBlock(text="何为数据一致性")],
            ),
            resource_selection=ResourceSelection.model_validate(
                {
                    "actions": [
                        {
                            "kind": "knowledge_retrieval",
                            "priority": 20,
                            "name": "knowledge_base_retrieval",
                            "config": {"collection_ids": [321], "top_k": 2},
                            "metadata": {
                                "capability_id": 1,
                                "capability_key": "knowledge_base_retrieval",
                                "capability_name": "知识库检索",
                                "capability_kind": "extension",
                                "capability_order": 1,
                                "capability_required": True,
                            },
                        },
                    ],
                },
            ),
            metadata={
                "capability_plan": {
                    "mode": "heuristic",
                    "summary": "required capability only",
                    "selected_capability_ids": [1],
                    "candidates": [
                        {
                            "capability_id": 1,
                            "capability_key": "knowledge_base_retrieval",
                            "capability_kind": "extension",
                            "score": 1.0,
                            "selected": True,
                            "reasons": ["required by selection"],
                        },
                    ],
                },
            },
        ),
        account_id=1,
        is_staff=False,
        send_event=collect_event,
    )

    await runtime.running_turns[turn_id].task

    turn = await ChatTurn.get(id=turn_id)

    assert str(turn.status) == "failed"
    assert any(
        event.event == "warning" and event.payload.code == "knowledge_retrieval_unavailable"
        for event in events
    )
    assert any(event.event == "turn.failed" for event in events)


@pytest.mark.asyncio
async def test_retrieval_no_hit_emits_warning_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = ChatRepository()
    runtime = ChatRuntime(repository=repository)

    conversation = await repository.create_conversation(
        title="retrieval-no-hit-warning",
        user_id=1,
        resource_selection=ResourceSelection(),
    )
    conversation_summary = await repository.summarize_conversation(conversation)
    events = []

    async def fake_retrieve_context(**kwargs):
        return {
            "retrievals": [],
            "requested_collection_ids": [321],
            "searched_collection_ids": [321],
            "missing_collection_ids": [],
            "inaccessible_collection_ids": [],
            "failed_collection_ids": [],
        }

    async def collect_event(event) -> None:
        events.append(event)

    monkeypatch.setattr(runtime.execution_manager, "retrieve_context", fake_retrieve_context)
    monkeypatch.setattr(
        runtime.execution_manager,
        "generate_response",
        lambda **kwargs: asyncio.sleep(0, result=("未检索到命中", UsagePayload(requests=1))),
    )

    turn_id = await runtime.execute_turn(
        ws_session_id=None,
        ws_public_session_id="ws-public",
        conversation=conversation_summary,
        turn_request=TurnStartRequest(
            conversation_id=conversation.id,
            request_id="req-retrieval-no-hit",
            input=MessageBundlePayload(
                role=ChatRoleEnum.user,
                blocks=[TextBlock(text="根据知识库回答这个问题")],
            ),
            resource_selection=ResourceSelection.model_validate(
                {
                    "actions": [
                        {
                            "kind": "knowledge_retrieval",
                            "priority": 20,
                            "config": {"collection_ids": [321], "top_k": 2},
                        },
                    ],
                },
            ),
        ),
        account_id=1,
        is_staff=False,
        send_event=collect_event,
    )

    await runtime.running_turns[turn_id].task

    turn = await ChatTurn.get(id=turn_id)

    assert str(turn.status) == "completed"
    assert any(
        event.event == "warning" and event.payload.code == "knowledge_retrieval_no_hit"
        for event in events
    )


@pytest.mark.asyncio
async def test_extension_intermediate_stages_create_nested_steps_and_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = ChatRepository()
    runtime = ChatRuntime(repository=repository)

    conversation = await repository.create_conversation(
        title="extension-intermediate-stages",
        user_id=1,
        resource_selection=ResourceSelection(),
    )
    conversation_summary = await repository.summarize_conversation(conversation)
    events = []

    async def fake_retrieve_context(**kwargs):
        callback = kwargs.get("extension_event_callback")
        if callback is not None:
            await callback(
                ExtensionEventStageEnum.rewriting,
                "正在改写查询",
                {"rewritten_query": "数据一致性 定义"},
            )
            await callback(
                ExtensionEventStageEnum.retrieving,
                "正在检索知识库",
                {"collection_ids": [321]},
            )
            await callback(
                ExtensionEventStageEnum.reranking,
                "正在 rerank 检索结果",
                {"candidate_count": 5},
            )
        return {
            "retrievals": [],
            "requested_collection_ids": [321],
            "searched_collection_ids": [321],
            "missing_collection_ids": [],
            "inaccessible_collection_ids": [],
            "failed_collection_ids": [],
        }

    async def collect_event(event) -> None:
        events.append(event)

    monkeypatch.setattr(runtime.execution_manager, "retrieve_context", fake_retrieve_context)
    monkeypatch.setattr(
        runtime.execution_manager,
        "generate_response",
        lambda **kwargs: asyncio.sleep(0, result=("未检索到命中", UsagePayload(requests=1))),
    )

    turn_id = await runtime.execute_turn(
        ws_session_id=None,
        ws_public_session_id="ws-public",
        conversation=conversation_summary,
        turn_request=TurnStartRequest(
            conversation_id=conversation.id,
            request_id="req-extension-intermediate-stages",
            input=MessageBundlePayload(
                role=ChatRoleEnum.user,
                blocks=[TextBlock(text="何为数据一致性")],
            ),
            resource_selection=ResourceSelection.model_validate(
                {
                    "actions": [
                        {
                            "kind": "knowledge_retrieval",
                            "priority": 20,
                            "config": {"collection_ids": [321], "top_k": 2},
                            "metadata": {
                                "capability_id": 1,
                                "capability_key": "knowledge_base_retrieval",
                                "capability_name": "知识库检索",
                                "capability_kind": "extension",
                                "capability_order": 1,
                                "emit_intermediate_events": True,
                            },
                        },
                    ],
                },
            ),
        ),
        account_id=1,
        is_staff=False,
        send_event=collect_event,
    )

    await runtime.running_turns[turn_id].task

    retrieval_step = await ChatStep.filter(
        turn_id=turn_id,
        name="knowledge_retrieval",
        deleted_at=0,
    ).first()
    assert retrieval_step is not None

    nested_steps = await ChatStep.filter(
        turn_id=turn_id,
        parent_step_id=retrieval_step.id,
        deleted_at=0,
    ).order_by("sequence")
    extension_data = await ChatData.filter(
        turn_id=turn_id,
        payload_type="extension_event",
        deleted_at=0,
    ).order_by("id")
    step_io_data = await ChatData.filter(
        turn_id=turn_id,
        payload_type="step_io",
        deleted_at=0,
    ).order_by("id")

    assert [step.name for step in nested_steps] == [
        "extension:retrieving",
        "extension:rewriting",
        "extension:retrieving",
        "extension:reranking",
        "extension:no_hit",
    ]
    assert [item.payload["stage"] for item in extension_data] == [
        ExtensionEventStageEnum.retrieving.value,
        ExtensionEventStageEnum.rewriting.value,
        ExtensionEventStageEnum.retrieving.value,
        ExtensionEventStageEnum.reranking.value,
        ExtensionEventStageEnum.no_hit.value,
    ]
    assert [item.payload["phase"] for item in step_io_data] == ["input", "output"]
    assert StepIOPayload.model_validate(step_io_data[0].payload).phase.value == "input"
    assert StepIOPayload.model_validate(step_io_data[1].payload).phase.value == "output"
    assert retrieval_step.input_data_ids
    assert retrieval_step.output_data_ids
    assert any(
        event.event == "data.created"
        and event.payload.data.payload_type == "extension_event"
        and isinstance(event.payload.data.payload, ExtensionEventPayload)
        for event in events
    )
    assert any(
        event.event == "data.created"
        and event.payload.data.payload_type == "step_io"
        and isinstance(event.payload.data.payload, StepIOPayload)
        for event in events
    )


@pytest.mark.asyncio
async def test_function_tool_planner_can_override_heuristic_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = ChatRepository()
    function_tools = FunctionToolRegistry()

    async def execute_heuristic_tool(session, spec) -> FunctionToolExecutionResult:
        payload = MessageBundlePayload(
            role=ChatRoleEnum.assistant,
            blocks=[TextBlock(text="heuristic answer")],
        )
        return FunctionToolExecutionResult(
            tool_name=spec.tool_name,
            title=spec.title,
            summary="heuristic result",
            text=payload.text,
            prefer_terminal=True,
            terminal_payload=payload,
        )

    async def execute_planned_tool(session, spec) -> FunctionToolExecutionResult:
        payload = MessageBundlePayload(
            role=ChatRoleEnum.assistant,
            blocks=[TextBlock(text="planned answer")],
        )
        return FunctionToolExecutionResult(
            tool_name=spec.tool_name,
            title=spec.title,
            summary="planned result",
            text=payload.text,
            prefer_terminal=True,
            terminal_payload=payload,
        )

    function_tools.register(
        "heuristic_tool",
        title="Heuristic Tool",
        description="启发式容易误判时会被命中。",
        matcher=lambda query, session: 1.0,
        executor=execute_heuristic_tool,
        default_result_mode=FunctionCallResultModeEnum.terminal,
    )
    function_tools.register(
        "planned_tool",
        title="Planned Tool",
        description="只有 planner 应该选中它。",
        matcher=lambda query, session: 0.0,
        executor=execute_planned_tool,
        default_result_mode=FunctionCallResultModeEnum.terminal,
    )

    runtime = ChatRuntime(repository=repository, function_tool_registry=function_tools)

    async def fake_plan_function_tools_with_llm(config, *, session_context, scored_specs):
        return SimpleNamespace(selected_tool_names=["planned_tool"], summary="planner selected")

    monkeypatch.setattr(runtime.execution_manager, "plan_function_tools_with_llm", fake_plan_function_tools_with_llm)

    conversation = await repository.create_conversation(
        title="planner-tool-selection",
        user_id=1,
        resource_selection=ResourceSelection(),
    )
    conversation_summary = await repository.summarize_conversation(conversation)

    turn_id = await runtime.execute_turn(
        ws_session_id=None,
        ws_public_session_id="ws-public",
        conversation=conversation_summary,
        turn_request=TurnStartRequest(
            conversation_id=conversation.id,
            request_id="req-planner-tool-selection",
            input=MessageBundlePayload(
                role=ChatRoleEnum.user,
                blocks=[TextBlock(text="请判断用哪个函数更合适")],
            ),
            resource_selection=ResourceSelection.model_validate(
                {
                    "actions": [
                        {
                            "kind": "function_call",
                            "priority": 10,
                            "config": {
                                "selection_mode": "hybrid",
                                "planner_model_config_id": 1,
                                "tools": [
                                    {"tool_name": "heuristic_tool"},
                                    {"tool_name": "planned_tool"},
                                ],
                            },
                        },
                    ],
                },
            ),
        ),
        account_id=1,
        is_staff=False,
        send_event=lambda event: asyncio.sleep(0),
    )

    await runtime.running_turns[turn_id].task

    turn = await ChatTurn.get(id=turn_id)
    function_data = await ChatData.filter(
        turn_id=turn_id,
        payload_type="function_result",
        deleted_at=0,
    ).first()
    output = await ChatData.get(id=turn.output_root_data_id)

    assert str(turn.status) == "completed"
    assert function_data is not None
    assert function_data.payload["tool_name"] == "planned_tool"
    assert "planned answer" in output.payload["blocks"][0]["text"]
