import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from ext.ext_tortoise.enums import (
    ChatDataKindEnum,
    ChatStepStatusEnum,
    ChatTurnStatusEnum,
    ChatTurnTriggerEnum,
)
from ext.ext_tortoise.models.knowledge_base import ChatData, ChatStep, ChatTurn, Collection
from service.chat.domain.schema import (
    ActionResultDispositionEnum,
    ChatRequestContext,
    ChatPayloadTypeEnum,
    ChatRoleEnum,
    ChatWarningCodeEnum,
    EventNameEnum,
    MessageBundlePayload,
    ResourceSelection,
    TextBlock,
    TurnStartRequest,
    UsagePayload,
)
from service.chat.runtime.engine import ChatRuntime
from service.chat.runtime.tool_executor import ToolExecutionResult, ToolExecutionWarning, ToolRegistry
from service.chat.store.repository import ChatRepository


class DummyRepository:
    pass


def _request_context(*, account_id: int = 1, is_staff: bool = False) -> ChatRequestContext:
    return ChatRequestContext(
        account=SimpleNamespace(id=account_id, is_staff=is_staff),
        session_id=uuid4(),
    )


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


@pytest.mark.asyncio
async def test_terminal_tool_short_circuits_following_llm_step() -> None:
    repository = ChatRepository()
    tool_registry = ToolRegistry()

    async def execute_terminal_tool(ctx) -> ToolExecutionResult:
        payload = MessageBundlePayload(
            role=ChatRoleEnum.assistant,
            blocks=[TextBlock(text="terminal answer")],
        )
        return ToolExecutionResult(
            tool_name=ctx.tool_name or "terminal_tool",
            title="Terminal Tool",
            disposition=ActionResultDispositionEnum.terminal,
            summary="return terminal output",
            text=payload.text,
            terminal_payload=payload,
        )

    tool_registry.register(
        "terminal_tool",
        title="Terminal Tool",
        matcher=lambda query, session: 1.0,
        executor=execute_terminal_tool,
    )
    runtime = ChatRuntime(repository=repository, tool_registry=tool_registry)

    conversation = await repository.create_conversation(
        title="terminal-short-circuit",
        user_id=1,
        resource_selection=ResourceSelection(),
    )
    conversation_summary = await repository.summarize_conversation(conversation)
    events = []

    async def collect_event(event) -> None:
        events.append(event)

    turn_id = await runtime.execute_turn(
        context=_request_context(),
        conversation=conversation_summary,
        turn_request=TurnStartRequest(
            conversation_id=conversation.id,
            request_id=uuid4(),
            input=MessageBundlePayload(
                role=ChatRoleEnum.user,
                blocks=[TextBlock(text="直接给我当前答案")],
            ),
            resource_selection=ResourceSelection.model_validate(
                {
                    "actions": [
                        {
                            "kind": "tool_call",
                            "priority": 10,
                            "config": {"tools": [{"tool_name": "terminal_tool"}]},
                        },
                    ],
                },
            ),
        ),
        send_event=collect_event,
    )

    await runtime.running_turns[turn_id].task

    turn = await ChatTurn.get(id=turn_id)
    steps = await ChatStep.filter(turn_id=turn_id, deleted_at=0).order_by("sequence")
    output = await ChatData.get(id=turn.output_root_data_id)

    assert str(turn.status) == "completed"
    assert [step.name for step in steps] == ["user_message", "terminal_tool"]
    assert output.payload["tool_name"] == "terminal_tool"
    assert output.payload["terminal"] is True
    assert any(event.event == EventNameEnum.step_started.value for event in events)
    assert any(event.event == EventNameEnum.step_completed.value for event in events)


@pytest.mark.asyncio
async def test_retrieval_no_hit_emits_warning_and_keeps_llm_response() -> None:
    repository = ChatRepository()
    tool_registry = ToolRegistry()

    async def execute_retrieval_tool(ctx, collection_ids: list[int], top_k: int = 5) -> ToolExecutionResult:
        return ToolExecutionResult(
            tool_name=ctx.tool_name or "knowledge_base_search",
            title="知识库检索",
            summary="知识库检索无命中",
            data={
                "requested_collection_ids": list(collection_ids),
                "searched_collection_ids": list(collection_ids),
                "missing_collection_ids": [],
                "inaccessible_collection_ids": [],
                "failed_collection_ids": [],
                "top_k": top_k,
            },
            warnings=[
                ToolExecutionWarning(
                    message="知识库检索无命中",
                    code=ChatWarningCodeEnum.knowledge_retrieval_no_hit,
                ),
            ],
        )

    tool_registry.register(
        "knowledge_base_search",
        title="知识库检索",
        executor=execute_retrieval_tool,
        matcher=lambda query, session: 1.0,
    )
    runtime = ChatRuntime(repository=repository, tool_registry=tool_registry)

    conversation = await repository.create_conversation(
        title="retrieval-no-hit-warning",
        user_id=1,
        resource_selection=ResourceSelection(),
    )
    conversation_summary = await repository.summarize_conversation(conversation)
    events = []

    async def fake_generate_response(**kwargs):
        return "未检索到命中", UsagePayload(requests=1)

    async def collect_event(event) -> None:
        events.append(event)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(runtime.execution_manager, "generate_response", fake_generate_response)

    turn_id = await runtime.execute_turn(
        context=_request_context(),
        conversation=conversation_summary,
        turn_request=TurnStartRequest(
            conversation_id=conversation.id,
            request_id=uuid4(),
            input=MessageBundlePayload(
                role=ChatRoleEnum.user,
                blocks=[TextBlock(text="根据知识库回答这个问题")],
            ),
            resource_selection=ResourceSelection.model_validate(
                {
                    "actions": [
                        {
                            "kind": "tool_call",
                            "priority": 20,
                            "config": {
                                "tools": [
                                    {
                                        "tool_name": "knowledge_base_search",
                                        "args": {"collection_ids": [321], "top_k": 2},
                                    },
                                ],
                            },
                        },
                    ],
                },
            ),
        ),
        send_event=collect_event,
    )

    await runtime.running_turns[turn_id].task
    monkeypatch.undo()

    turn = await ChatTurn.get(id=turn_id)
    steps = await ChatStep.filter(turn_id=turn_id, deleted_at=0).order_by("sequence")
    output = await ChatData.get(id=turn.output_root_data_id)

    assert str(turn.status) == "completed"
    assert [step.name for step in steps] == ["user_message", "knowledge_base_search", "llm_response"]
    assert output.payload["blocks"][0]["text"] == "未检索到命中"
    assert any(
        event.event == EventNameEnum.warning.value
        and event.payload.code == "knowledge_retrieval_no_hit"
        for event in events
    )


@pytest.mark.asyncio
async def test_tool_planner_can_override_heuristic_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = ChatRepository()
    tool_registry = ToolRegistry()

    async def execute_heuristic_tool(ctx) -> ToolExecutionResult:
        payload = MessageBundlePayload(
            role=ChatRoleEnum.assistant,
            blocks=[TextBlock(text="heuristic answer")],
        )
        return ToolExecutionResult(
            tool_name=ctx.tool_name or "heuristic_tool",
            title="Heuristic Tool",
            disposition=ActionResultDispositionEnum.terminal,
            summary="heuristic result",
            text=payload.text,
            terminal_payload=payload,
        )

    async def execute_planned_tool(ctx) -> ToolExecutionResult:
        payload = MessageBundlePayload(
            role=ChatRoleEnum.assistant,
            blocks=[TextBlock(text="planned answer")],
        )
        return ToolExecutionResult(
            tool_name=ctx.tool_name or "planned_tool",
            title="Planned Tool",
            disposition=ActionResultDispositionEnum.terminal,
            summary="planned result",
            text=payload.text,
            terminal_payload=payload,
        )

    tool_registry.register(
        "heuristic_tool",
        title="Heuristic Tool",
        description="启发式容易误判时会被命中。",
        matcher=lambda query, session: 1.0,
        executor=execute_heuristic_tool,
    )
    tool_registry.register(
        "planned_tool",
        title="Planned Tool",
        description="只有 planner 应该选中它。",
        matcher=lambda query, session: 0.0,
        executor=execute_planned_tool,
    )

    runtime = ChatRuntime(repository=repository, tool_registry=tool_registry)

    async def fake_plan_tools_with_llm(config, *, session_context, scored_specs):
        return SimpleNamespace(selected_tool_names=["planned_tool"], summary="planner selected")

    monkeypatch.setattr(runtime.execution_manager, "plan_tools_with_llm", fake_plan_tools_with_llm)

    conversation = await repository.create_conversation(
        title="planner-tool-selection",
        user_id=1,
        resource_selection=ResourceSelection(),
    )
    conversation_summary = await repository.summarize_conversation(conversation)

    turn_id = await runtime.execute_turn(
        context=_request_context(),
        conversation=conversation_summary,
        turn_request=TurnStartRequest(
            conversation_id=conversation.id,
            request_id=uuid4(),
            input=MessageBundlePayload(
                role=ChatRoleEnum.user,
                blocks=[TextBlock(text="请判断用哪个函数更合适")],
            ),
            resource_selection=ResourceSelection.model_validate(
                {
                    "actions": [
                        {
                            "kind": "tool_call",
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
        send_event=lambda event: asyncio.sleep(0),
    )

    await runtime.running_turns[turn_id].task

    turn = await ChatTurn.get(id=turn_id)
    output = await ChatData.get(id=turn.output_root_data_id)

    assert str(turn.status) == "completed"
    assert output.payload["tool_name"] == "planned_tool"


@pytest.mark.asyncio
async def test_tool_call_with_missing_registered_tool_fails() -> None:
    repository = ChatRepository()
    runtime = ChatRuntime(repository=repository)

    conversation = await repository.create_conversation(
        title="tool-call-required-missing",
        user_id=1,
        resource_selection=ResourceSelection(),
    )
    conversation_summary = await repository.summarize_conversation(conversation)

    turn_id = await runtime.execute_turn(
        context=_request_context(),
        conversation=conversation_summary,
        turn_request=TurnStartRequest(
            conversation_id=conversation.id,
            request_id=uuid4(),
            input=MessageBundlePayload(
                role=ChatRoleEnum.user,
                blocks=[TextBlock(text="请执行必须存在的工具")],
            ),
            resource_selection=ResourceSelection.model_validate(
                {
                    "actions": [
                        {
                            "kind": "tool_call",
                            "priority": 10,
                            "config": {"tools": [{"tool_name": "missing_tool"}]},
                        },
                    ],
                },
            ),
        ),
        send_event=lambda event: asyncio.sleep(0),
    )

    await runtime.running_turns[turn_id].task

    turn = await ChatTurn.get(id=turn_id)
    output = await ChatData.get(id=turn.output_root_data_id)

    assert str(turn.status) == "failed"
    assert output.payload["type"] == "error"
    assert "missing_tool" in output.payload["message"]


@pytest.mark.asyncio
async def test_cancel_turn_marks_active_steps_canceled() -> None:
    repository = ChatRepository()
    tool_registry = ToolRegistry()
    tool_started = asyncio.Event()

    async def execute_slow_tool(ctx) -> ToolExecutionResult:
        tool_started.set()
        await asyncio.Future()
        raise AssertionError("unreachable")

    tool_registry.register(
        "slow_tool",
        title="Slow Tool",
        matcher=lambda query, session: 1.0,
        executor=execute_slow_tool,
    )
    runtime = ChatRuntime(repository=repository, tool_registry=tool_registry)

    conversation = await repository.create_conversation(
        title="cancel-active-step",
        user_id=1,
        resource_selection=ResourceSelection(),
    )
    conversation_summary = await repository.summarize_conversation(conversation)

    turn_id = await runtime.execute_turn(
        context=_request_context(),
        conversation=conversation_summary,
        turn_request=TurnStartRequest(
            conversation_id=conversation.id,
            request_id=uuid4(),
            input=MessageBundlePayload(
                role=ChatRoleEnum.user,
                blocks=[TextBlock(text="执行一个很慢的工具")],
            ),
            resource_selection=ResourceSelection.model_validate(
                {
                    "actions": [
                        {
                            "kind": "tool_call",
                            "priority": 10,
                            "config": {"tools": [{"tool_name": "slow_tool"}]},
                        },
                    ],
                },
            ),
        ),
        send_event=lambda event: asyncio.sleep(0),
    )

    await asyncio.wait_for(tool_started.wait(), timeout=1)
    assert await runtime.cancel_turn(turn_id) is True
    await runtime.running_turns[turn_id].task

    turn = await ChatTurn.get(id=turn_id)
    steps = await ChatStep.filter(turn_id=turn_id, deleted_at=0).order_by("sequence")

    assert str(turn.status) == ChatTurnStatusEnum.canceled.value
    assert [step.name for step in steps] == ["user_message", "slow_tool"]
    assert str(steps[0].status) == ChatStepStatusEnum.completed.value
    assert str(steps[1].status) == ChatStepStatusEnum.canceled.value


@pytest.mark.asyncio
async def test_action_error_is_recorded_on_action_step_without_generic_turn_error() -> None:
    repository = ChatRepository()
    tool_registry = ToolRegistry()

    async def execute_broken_tool(ctx) -> ToolExecutionResult:
        raise RuntimeError("broken tool")

    tool_registry.register(
        "broken_tool",
        title="Broken Tool",
        matcher=lambda query, session: 1.0,
        executor=execute_broken_tool,
    )
    runtime = ChatRuntime(repository=repository, tool_registry=tool_registry)

    conversation = await repository.create_conversation(
        title="failed-active-step",
        user_id=1,
        resource_selection=ResourceSelection(),
    )
    conversation_summary = await repository.summarize_conversation(conversation)

    turn_id = await runtime.execute_turn(
        context=_request_context(),
        conversation=conversation_summary,
        turn_request=TurnStartRequest(
            conversation_id=conversation.id,
            request_id=uuid4(),
            input=MessageBundlePayload(
                role=ChatRoleEnum.user,
                blocks=[TextBlock(text="执行一个会失败的工具")],
            ),
            resource_selection=ResourceSelection.model_validate(
                {
                    "actions": [
                        {
                            "kind": "tool_call",
                            "priority": 10,
                            "config": {"tools": [{"tool_name": "broken_tool"}]},
                        },
                    ],
                },
            ),
        ),
        send_event=lambda event: asyncio.sleep(0),
    )

    await runtime.running_turns[turn_id].task

    turn = await ChatTurn.get(id=turn_id)
    steps = await ChatStep.filter(turn_id=turn_id, deleted_at=0).order_by("sequence")
    output = await ChatData.get(id=turn.output_root_data_id)

    assert str(turn.status) == ChatTurnStatusEnum.failed.value
    assert [step.name for step in steps] == ["user_message", "broken_tool"]
    assert str(steps[0].status) == ChatStepStatusEnum.completed.value
    assert str(steps[1].status) == ChatStepStatusEnum.failed.value
    assert output.payload["type"] == "error"
    assert output.payload["message"] == "broken tool"


@pytest.mark.asyncio
async def test_llm_failure_is_recorded_on_llm_step_without_generic_turn_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = ChatRepository()
    runtime = ChatRuntime(repository=repository)

    async def fake_generate_response(**kwargs):
        raise RuntimeError("llm exploded")

    monkeypatch.setattr(runtime.execution_manager, "generate_response", fake_generate_response)

    conversation = await repository.create_conversation(
        title="failed-llm-step",
        user_id=1,
        resource_selection=ResourceSelection(),
    )
    conversation_summary = await repository.summarize_conversation(conversation)

    turn_id = await runtime.execute_turn(
        context=_request_context(),
        conversation=conversation_summary,
        turn_request=TurnStartRequest(
            conversation_id=conversation.id,
            request_id=uuid4(),
            input=MessageBundlePayload(
                role=ChatRoleEnum.user,
                blocks=[TextBlock(text="请给我一个答案")],
            ),
            resource_selection=ResourceSelection(),
        ),
        send_event=lambda event: asyncio.sleep(0),
    )

    await runtime.running_turns[turn_id].task

    turn = await ChatTurn.get(id=turn_id)
    steps = await ChatStep.filter(turn_id=turn_id, deleted_at=0).order_by("sequence")
    output = await ChatData.get(id=turn.output_root_data_id)

    assert str(turn.status) == ChatTurnStatusEnum.failed.value
    assert [step.name for step in steps] == ["user_message", "llm_response"]
    assert str(steps[0].status) == ChatStepStatusEnum.completed.value
    assert str(steps[1].status) == ChatStepStatusEnum.failed.value
    assert output.payload["type"] == "error"
    assert output.payload["message"] == "llm exploded"


@pytest.mark.asyncio
async def test_mcp_call_executes_platform_session_state_and_persists_terminal_output() -> None:
    repository = ChatRepository()
    runtime = ChatRuntime(repository=repository)

    conversation = await repository.create_conversation(
        title="mcp-call-platform-session-state",
        user_id=1,
        resource_selection=ResourceSelection(),
    )
    conversation_summary = await repository.summarize_conversation(conversation)

    turn_id = await runtime.execute_turn(
        context=_request_context(),
        conversation=conversation_summary,
        turn_request=TurnStartRequest(
            conversation_id=conversation.id,
            request_id=uuid4(),
            input=MessageBundlePayload(
                role=ChatRoleEnum.user,
                blocks=[TextBlock(text="请通过 MCP 返回当前会话状态")],
            ),
            resource_selection=ResourceSelection.model_validate(
                {
                    "actions": [
                        {
                            "kind": "mcp_call",
                            "priority": 10,
                            "config": {"server_name": "platform", "tool_names": ["session_state"]},
                        },
                    ],
                },
            ),
        ),
        send_event=lambda event: asyncio.sleep(0),
    )

    await runtime.running_turns[turn_id].task

    turn = await ChatTurn.get(id=turn_id)
    steps = await ChatStep.filter(turn_id=turn_id, deleted_at=0).order_by("sequence")
    output = await ChatData.get(id=turn.output_root_data_id)

    assert str(turn.status) == "completed"
    assert [step.name for step in steps] == ["user_message", "platform.session_state"]
    assert output.payload["server_name"] == "platform"
    assert output.payload["tool_name"] == "session_state"
    assert output.payload["terminal"] is True


@pytest.mark.asyncio
async def test_mcp_call_context_result_can_feed_following_llm_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = ChatRepository()
    runtime = ChatRuntime(repository=repository)

    await Collection.create(name="公开集合A", is_public=True)
    await Collection.create(name="私有集合B", is_public=False, user_id=uuid4())

    conversation = await repository.create_conversation(
        title="mcp-call-collection-catalog",
        user_id=1,
        resource_selection=ResourceSelection(),
    )
    conversation_summary = await repository.summarize_conversation(conversation)

    async def fake_generate_response(**kwargs):
        return "已根据 MCP 上下文回答", UsagePayload(requests=1)

    monkeypatch.setattr(runtime.execution_manager, "generate_response", fake_generate_response)

    turn_id = await runtime.execute_turn(
        context=_request_context(),
        conversation=conversation_summary,
        turn_request=TurnStartRequest(
            conversation_id=conversation.id,
            request_id=uuid4(),
            input=MessageBundlePayload(
                role=ChatRoleEnum.user,
                blocks=[TextBlock(text="请列出可访问集合并总结")],
            ),
            resource_selection=ResourceSelection.model_validate(
                {
                    "actions": [
                        {
                            "kind": "mcp_call",
                            "priority": 10,
                            "config": {
                                "server_name": "knowledge_base",
                                "tool_names": ["collection_catalog"],
                            },
                        },
                        {
                            "kind": "llm_response",
                            "priority": 20,
                            "config": {},
                        },
                    ],
                },
            ),
        ),
        send_event=lambda event: asyncio.sleep(0),
    )

    await runtime.running_turns[turn_id].task

    turn = await ChatTurn.get(id=turn_id)
    mcp_data = await ChatData.filter(turn_id=turn_id, payload_type="mcp_result", deleted_at=0).first()
    output = await ChatData.get(id=turn.output_root_data_id)

    assert str(turn.status) == "completed"
    assert mcp_data is not None
    assert mcp_data.payload["server_name"] == "knowledge_base"
    assert mcp_data.payload["terminal"] is False
    items = mcp_data.payload["content"]["items"]
    assert len(items) == 1
    assert items[0]["name"] == "公开集合A"
    assert items[0]["is_public"] is True
    assert output.payload["blocks"][0]["text"] == "已根据 MCP 上下文回答"


@pytest.mark.asyncio
async def test_mcp_failure_is_recorded_on_mcp_step_without_generic_turn_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = ChatRepository()
    runtime = ChatRuntime(repository=repository)

    async def explode_execute_many(*args, **kwargs):
        raise RuntimeError("mcp exploded")

    monkeypatch.setattr(runtime.mcp_registry, "execute_many", explode_execute_many)

    conversation = await repository.create_conversation(
        title="failed-mcp-step",
        user_id=1,
        resource_selection=ResourceSelection(),
    )
    conversation_summary = await repository.summarize_conversation(conversation)

    turn_id = await runtime.execute_turn(
        context=_request_context(),
        conversation=conversation_summary,
        turn_request=TurnStartRequest(
            conversation_id=conversation.id,
            request_id=uuid4(),
            input=MessageBundlePayload(
                role=ChatRoleEnum.user,
                blocks=[TextBlock(text="请执行一个失败的 MCP 调用")],
            ),
            resource_selection=ResourceSelection.model_validate(
                {
                    "actions": [
                        {
                            "kind": "mcp_call",
                            "priority": 10,
                            "config": {"server_name": "platform", "tool_names": ["session_state"]},
                        },
                    ],
                },
            ),
        ),
        send_event=lambda event: asyncio.sleep(0),
    )

    await runtime.running_turns[turn_id].task

    turn = await ChatTurn.get(id=turn_id)
    steps = await ChatStep.filter(turn_id=turn_id, deleted_at=0).order_by("sequence")
    output = await ChatData.get(id=turn.output_root_data_id)

    assert str(turn.status) == ChatTurnStatusEnum.failed.value
    assert [step.name for step in steps] == ["user_message", "mcp_call"]
    assert str(steps[0].status) == ChatStepStatusEnum.completed.value
    assert str(steps[1].status) == ChatStepStatusEnum.failed.value
    assert output.payload["type"] == "error"
    assert output.payload["message"] == "mcp exploded"


@pytest.mark.asyncio
async def test_sub_agent_call_executes_nested_actions_and_returns_terminal_output() -> None:
    repository = ChatRepository()
    runtime = ChatRuntime(repository=repository)

    conversation = await repository.create_conversation(
        title="sub-agent-delegate",
        user_id=1,
        resource_selection=ResourceSelection(),
    )
    conversation_summary = await repository.summarize_conversation(conversation)

    turn_id = await runtime.execute_turn(
        context=_request_context(),
        conversation=conversation_summary,
        turn_request=TurnStartRequest(
            conversation_id=conversation.id,
            request_id=uuid4(),
            input=MessageBundlePayload(
                role=ChatRoleEnum.user,
                blocks=[TextBlock(text="请计算 2 * (3 + 4)")],
            ),
            resource_selection=ResourceSelection.model_validate(
                {
                    "actions": [
                        {
                            "kind": "sub_agent_call",
                            "priority": 10,
                            "name": "math_delegate",
                            "metadata": {
                                "delegation": {
                                    "output_contract": "terminal_text",
                                },
                            },
                            "config": {
                                "system_prompt": "你是数学专家代理。",
                                "actions": [
                                    {
                                        "kind": "tool_call",
                                        "priority": 10,
                                        "config": {"tools": [{"tool_name": "calculate_expression"}]},
                                    },
                                ],
                            },
                        },
                    ],
                },
            ),
        ),
        send_event=lambda event: asyncio.sleep(0),
    )

    await runtime.running_turns[turn_id].task

    turn = await ChatTurn.get(id=turn_id)
    steps = await ChatStep.filter(turn_id=turn_id, deleted_at=0).order_by("sequence")
    sub_agent_data = await ChatData.filter(turn_id=turn_id, payload_type="sub_agent_result", deleted_at=0).first()
    output = await ChatData.get(id=turn.output_root_data_id)

    assert str(turn.status) == "completed"
    delegate_step = next(step for step in steps if step.name == "math_delegate")
    assert any(step.name == "calculate_expression" and step.parent_step_id == delegate_step.id for step in steps)
    assert sub_agent_data is not None
    assert sub_agent_data.payload["terminal"] is True
    assert sub_agent_data.payload["content_text"] == "计算结果：2 * (3 + 4) = 14"
    assert output.payload["content_text"] == "计算结果：2 * (3 + 4) = 14"


@pytest.mark.asyncio
async def test_sub_agent_llm_uses_delegate_prompt_and_history_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = ChatRepository()
    runtime = ChatRuntime(repository=repository)

    conversation = await repository.create_conversation(
        title="sub-agent-config",
        user_id=1,
        resource_selection=ResourceSelection(),
    )
    seed_turn = await repository.create_turn(
        conversation_id=conversation.id,
        request_id=uuid4(),
        trigger=ChatTurnTriggerEnum.user,
        resource_selection=ResourceSelection(),
    )
    seed_input_step = await repository.create_step(
        conversation_id=conversation.id,
        turn_id=seed_turn.id,
        name="user_message",
        kind="system",
        sequence=10,
    )
    seed_input = await repository.create_data(
        conversation_id=conversation.id,
        turn_id=seed_turn.id,
        step_id=seed_input_step.id,
        kind=ChatDataKindEnum.output,
        payload_type=ChatPayloadTypeEnum.message_bundle,
        payload=MessageBundlePayload(
            role=ChatRoleEnum.user,
            blocks=[TextBlock(text="历史问题")],
        ),
    )
    await repository.update_step(seed_input_step, status=ChatStepStatusEnum.completed)
    await repository.update_turn(seed_turn, input_root_data_id=seed_input.id)
    seed_output_step = await repository.create_step(
        conversation_id=conversation.id,
        turn_id=seed_turn.id,
        name="llm_response",
        kind="llm",
        sequence=20,
    )
    seed_output = await repository.create_data(
        conversation_id=conversation.id,
        turn_id=seed_turn.id,
        step_id=seed_output_step.id,
        kind=ChatDataKindEnum.output,
        payload_type=ChatPayloadTypeEnum.message_bundle,
        payload=MessageBundlePayload(
            role=ChatRoleEnum.assistant,
            blocks=[TextBlock(text="历史回答")],
        ),
    )
    await repository.update_step(seed_output_step, status=ChatStepStatusEnum.completed)
    await repository.finalize_turn(
        seed_turn,
        status=ChatTurnStatusEnum.completed,
        output_root_data_id=seed_output.id,
        finished_at=seed_turn.started_at,
    )

    captured: dict[str, object] = {}

    async def fake_generate_response(**kwargs):
        captured["system_prompt_prefix"] = kwargs["system_prompt_prefix"]
        captured["extra_instructions"] = kwargs["extra_instructions"]
        captured["history"] = kwargs["context"].history
        captured["agent_key"] = kwargs["session_context"].turn_request.agent_key
        return "delegated answer", UsagePayload(requests=1)

    monkeypatch.setattr(runtime.execution_manager, "generate_response", fake_generate_response)
    conversation_summary = await repository.summarize_conversation(conversation)

    turn_id = await runtime.execute_turn(
        context=_request_context(),
        conversation=conversation_summary,
        turn_request=TurnStartRequest(
            conversation_id=conversation.id,
            request_id=uuid4(),
            input=MessageBundlePayload(
                role=ChatRoleEnum.user,
                blocks=[TextBlock(text="请处理这个委派任务")],
            ),
            resource_selection=ResourceSelection.model_validate(
                {
                    "actions": [
                        {
                            "kind": "sub_agent_call",
                            "priority": 10,
                            "name": "delegate.writer",
                            "metadata": {
                                "delegation": {
                                    "mounted_agent_key": "agent.writer",
                                    "mounted_agent_name": "Writer Agent",
                                    "pass_message_history": False,
                                    "output_contract": "terminal_text",
                                },
                            },
                            "config": {
                                "system_prompt": "你是一个只负责委派结果整理的专家代理。",
                                "instructions": ["只输出结论，不要展开过程。"],
                                "actions": [
                                    {
                                        "kind": "llm_response",
                                        "priority": 10,
                                        "config": {},
                                    },
                                ],
                            },
                        },
                    ],
                },
            ),
        ),
        send_event=lambda event: asyncio.sleep(0),
    )

    await runtime.running_turns[turn_id].task

    turn = await ChatTurn.get(id=turn_id)
    output = await ChatData.get(id=turn.output_root_data_id)

    assert str(turn.status) == ChatTurnStatusEnum.completed.value
    assert output.payload["content_text"] == "delegated answer"
    assert captured["system_prompt_prefix"] == "你是一个只负责委派结果整理的专家代理。"
    assert captured["extra_instructions"] == ["只输出结论，不要展开过程。"]
    assert captured["history"] == []
    assert captured["agent_key"] == "agent.writer"


@pytest.mark.asyncio
async def test_sub_agent_inherits_parent_prompt_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = ChatRepository()
    runtime = ChatRuntime(repository=repository)

    captured: dict[str, object] = {}

    async def fake_generate_response(**kwargs):
        prompt_context = kwargs["context"].prompt_context
        captured["prompt_instructions"] = [] if prompt_context is None else list(prompt_context.instructions)
        captured["applied_action_ids"] = [] if prompt_context is None else list(prompt_context.applied_action_ids)
        return "delegated answer", UsagePayload(requests=1)

    monkeypatch.setattr(runtime.execution_manager, "generate_response", fake_generate_response)

    conversation = await repository.create_conversation(
        title="sub-agent-inherit-prompt-state",
        user_id=1,
        resource_selection=ResourceSelection(),
    )
    conversation_summary = await repository.summarize_conversation(conversation)

    turn_id = await runtime.execute_turn(
        context=_request_context(),
        conversation=conversation_summary,
        turn_request=TurnStartRequest(
            conversation_id=conversation.id,
            request_id=uuid4(),
            input=MessageBundlePayload(
                role=ChatRoleEnum.user,
                blocks=[TextBlock(text="请处理这个委派任务")],
            ),
            resource_selection=ResourceSelection.model_validate(
                {
                    "actions": [
                        {
                            "kind": "system_prompt",
                            "priority": 5,
                            "config": {
                                "instructions": ["父级系统提示约束"],
                            },
                        },
                        {
                            "kind": "sub_agent_call",
                            "priority": 10,
                            "name": "delegate.writer",
                            "metadata": {
                                "delegation": {
                                    "mounted_agent_key": "agent.writer",
                                    "output_contract": "terminal_text",
                                },
                            },
                            "config": {
                                "system_prompt": "你是一个接收父级提示词上下文的代理。",
                                "actions": [
                                    {
                                        "kind": "llm_response",
                                        "priority": 10,
                                        "config": {},
                                    },
                                ],
                            },
                        },
                    ],
                },
            ),
        ),
        send_event=lambda event: asyncio.sleep(0),
    )

    await runtime.running_turns[turn_id].task

    assert captured["prompt_instructions"] == ["父级系统提示约束"]
    assert captured["applied_action_ids"] == ["anonymous:system_prompt"]


@pytest.mark.asyncio
async def test_sub_agent_pass_deps_fields_copies_selected_session_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = ChatRepository()
    tool_registry = ToolRegistry()
    runtime = ChatRuntime(repository=repository, tool_registry=tool_registry)

    async def stash_state_tool(ctx) -> ToolExecutionResult:
        ctx.deps.set_state("shared_ticket_id", "T-100")
        ctx.deps.set_state("private_note", "do-not-pass")
        return ToolExecutionResult(
            tool_name=ctx.tool_name or "stash_state",
            title="Stash State",
            summary="stored state",
            text="stored state",
        )

    tool_registry.register(
        "stash_state",
        title="Stash State",
        matcher=lambda query, session: 1.0,
        executor=stash_state_tool,
    )

    captured: dict[str, object] = {}

    async def fake_generate_response(**kwargs):
        session = kwargs["session_context"]
        captured["shared_ticket_id"] = session.get_state("shared_ticket_id")
        captured["private_note"] = session.get_state("private_note")
        return "delegated answer", UsagePayload(requests=1)

    monkeypatch.setattr(runtime.execution_manager, "generate_response", fake_generate_response)

    conversation = await repository.create_conversation(
        title="sub-agent-pass-deps",
        user_id=1,
        resource_selection=ResourceSelection(),
    )
    conversation_summary = await repository.summarize_conversation(conversation)

    turn_id = await runtime.execute_turn(
        context=_request_context(),
        conversation=conversation_summary,
        turn_request=TurnStartRequest(
            conversation_id=conversation.id,
            request_id=uuid4(),
            input=MessageBundlePayload(
                role=ChatRoleEnum.user,
                blocks=[TextBlock(text="请处理这个委派任务")],
            ),
            resource_selection=ResourceSelection.model_validate(
                {
                    "actions": [
                        {
                            "kind": "tool_call",
                            "priority": 5,
                            "config": {"tools": [{"tool_name": "stash_state"}]},
                        },
                        {
                            "kind": "sub_agent_call",
                            "priority": 10,
                            "name": "delegate.with.deps",
                            "metadata": {
                                "delegation": {
                                    "mounted_agent_key": "agent.writer",
                                    "pass_message_history": False,
                                    "pass_deps_fields": ["shared_ticket_id"],
                                    "output_contract": "terminal_text",
                                },
                            },
                            "config": {
                                "system_prompt": "你是一个接收委派依赖字段的代理。",
                                "actions": [
                                    {
                                        "kind": "llm_response",
                                        "priority": 10,
                                        "config": {},
                                    },
                                ],
                            },
                        },
                    ],
                },
            ),
        ),
        send_event=lambda event: asyncio.sleep(0),
    )

    await runtime.running_turns[turn_id].task

    assert captured["shared_ticket_id"] == "T-100"
    assert captured["private_note"] is None


@pytest.mark.asyncio
async def test_runtime_supports_concurrent_turns_across_users_and_conversations() -> None:
    repository = ChatRepository()
    tool_registry = ToolRegistry()
    runtime = ChatRuntime(repository=repository, tool_registry=tool_registry)

    started_conversations: list[int] = []
    both_started = asyncio.Event()
    release = asyncio.Event()

    async def execute_blocking_terminal_tool(ctx) -> ToolExecutionResult:
        started_conversations.append(ctx.deps.conversation_id)
        if len(started_conversations) >= 2:
            both_started.set()
        await asyncio.wait_for(release.wait(), timeout=1)
        payload = MessageBundlePayload(
            role=ChatRoleEnum.assistant,
            blocks=[TextBlock(text=f"done:{ctx.deps.conversation_id}")],
        )
        return ToolExecutionResult(
            tool_name=ctx.tool_name or "blocking_terminal_tool",
            title="Blocking Terminal Tool",
            disposition=ActionResultDispositionEnum.terminal,
            summary=payload.text,
            text=payload.text,
            terminal_payload=payload,
        )

    tool_registry.register(
        "blocking_terminal_tool",
        title="Blocking Terminal Tool",
        matcher=lambda query, session: 1.0,
        executor=execute_blocking_terminal_tool,
    )

    conversation_a = await repository.create_conversation(
        title="concurrent-a",
        user_id=1,
        resource_selection=ResourceSelection(),
    )
    conversation_b = await repository.create_conversation(
        title="concurrent-b",
        user_id=2,
        resource_selection=ResourceSelection(),
    )
    summary_a = await repository.summarize_conversation(conversation_a)
    summary_b = await repository.summarize_conversation(conversation_b)

    async def run_turn(conversation_summary, *, account_id: int, expression: str) -> int:
        turn_id = await runtime.execute_turn(
            context=_request_context(account_id=account_id),
            conversation=conversation_summary,
            turn_request=TurnStartRequest(
                conversation_id=conversation_summary.id,
                request_id=uuid4(),
                input=MessageBundlePayload(
                    role=ChatRoleEnum.user,
                    blocks=[TextBlock(text=expression)],
                ),
                resource_selection=ResourceSelection.model_validate(
                    {
                        "actions": [
                            {
                                "kind": "tool_call",
                                "priority": 10,
                                "config": {"tools": [{"tool_name": "blocking_terminal_tool"}]},
                            },
                        ],
                    },
                ),
            ),
            send_event=lambda event: asyncio.sleep(0),
        )
        await runtime.running_turns[turn_id].task
        return turn_id

    task_a = asyncio.create_task(run_turn(summary_a, account_id=1, expression="用户A"))
    task_b = asyncio.create_task(run_turn(summary_b, account_id=2, expression="用户B"))

    await asyncio.wait_for(both_started.wait(), timeout=1)
    assert len(runtime.running_turns) == 2
    assert set(started_conversations) == {conversation_a.id, conversation_b.id}

    release.set()
    turn_id_a, turn_id_b = await asyncio.gather(task_a, task_b)

    turn_a = await ChatTurn.get(id=turn_id_a)
    turn_b = await ChatTurn.get(id=turn_id_b)
    output_a = await ChatData.get(id=turn_a.output_root_data_id)
    output_b = await ChatData.get(id=turn_b.output_root_data_id)

    assert str(turn_a.status) == ChatTurnStatusEnum.completed.value
    assert str(turn_b.status) == ChatTurnStatusEnum.completed.value
    assert output_a.payload["content_text"] == f"done:{conversation_a.id}"
    assert output_b.payload["content_text"] == f"done:{conversation_b.id}"


@pytest.mark.asyncio
async def test_repository_build_history_supports_non_message_terminal_outputs() -> None:
    repository = ChatRepository()
    conversation = await repository.create_conversation(
        title="history-with-tool-result",
        user_id=1,
        resource_selection=ResourceSelection(),
    )
    turn = await repository.create_turn(
        conversation_id=conversation.id,
        request_id=uuid4(),
        trigger=ChatTurnTriggerEnum.user,
        resource_selection=ResourceSelection(),
    )
    user_step = await repository.create_step(
        conversation_id=conversation.id,
        turn_id=turn.id,
        name="user_message",
        kind="system",
        sequence=10,
    )
    user_data = await repository.create_data(
        conversation_id=conversation.id,
        turn_id=turn.id,
        step_id=user_step.id,
        kind=ChatDataKindEnum.output,
        payload_type="message_bundle",
        payload=MessageBundlePayload(
            role=ChatRoleEnum.user,
            blocks=[TextBlock(text="请计算 6 * 7")],
        ),
    )
    terminal_step = await repository.create_step(
        conversation_id=conversation.id,
        turn_id=turn.id,
        name="calculate_expression",
        kind="tool",
        sequence=20,
    )
    terminal_data = await repository.create_data(
        conversation_id=conversation.id,
        turn_id=turn.id,
        step_id=terminal_step.id,
        kind=ChatDataKindEnum.output,
        payload_type="tool_result",
        payload={
            "type": "tool_result",
            "tool_name": "calculate_expression",
            "title": "表达式计算",
            "disposition": "terminal",
            "summary": "计算结果：6 * 7 = 42",
            "content_text": "计算结果：6 * 7 = 42",
            "content": {"expression": "6 * 7", "result": 42},
            "terminal": True,
        },
    )
    await repository.update_turn(turn, input_root_data_id=user_data.id)
    await repository.finalize_turn(
        turn,
        status=ChatTurnStatusEnum.completed,
        finished_at=turn.started_at,
        output_root_data_id=terminal_data.id,
    )

    history = await repository.build_history(conversation.id)

    assert len(history) == 1
    assert history[0].user_text == "请计算 6 * 7"
    assert history[0].assistant_text == "计算结果：6 * 7 = 42"
