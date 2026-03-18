import asyncio

import pytest

from service.chat.runtime.engine import ChatRuntime
from service.chat.store.repository import ChatRepository
from service.chat.domain.schema import (
    ChatRoleEnum,
    FunctionCallResultModeEnum,
    FunctionToolSpec,
    MessageBundlePayload,
    ResourceSelection,
    TextBlock,
    TurnStartRequest,
    UsagePayload,
)
from service.chat.runtime.function_tools import FunctionToolExecutionResult, FunctionToolRegistry
from ext.ext_tortoise.models.knowledge_base import ChatStep, ChatTurn


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
        raise AssertionError("retrieval capability should be skipped after terminal output")

    monkeypatch.setattr(runtime, "retrieve_context", fail_retrieval)

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
                    "capabilities": [
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
