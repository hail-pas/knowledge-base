from types import SimpleNamespace
import pytest
from uuid import uuid4

from service.chat.execution.registry import ExecutionAction
from service.chat.domain.schema import (
    ChatActionKindEnum,
    ChatRequestContext,
    ChatRoleEnum,
    ConversationSummary,
    MessageBundlePayload,
    ResourceSelection,
    TextBlock,
    ToolCallConfig,
    ToolResultModeEnum,
    ToolSpec,
    TurnStartRequest,
)
from service.chat.runtime.tool_executor import create_default_tool_registry
from service.chat.runtime.session import ChatSessionContext
from ext.ext_tortoise.enums import ChatStepKindEnum


def _request_context(*, account_id: int = 1, is_staff: bool = False) -> ChatRequestContext:
    return ChatRequestContext(
        account=SimpleNamespace(id=account_id, is_staff=is_staff),
        session_id=uuid4(),
    )


def _session(query: str) -> ChatSessionContext:
    return ChatSessionContext(
        request_context=_request_context(),
        conversation=ConversationSummary(
            id=3,
            title="函数测试会话",
            agent_key="orchestrator.default",
            user_id=1,
            default_resource_selection=ResourceSelection(),
        ),
        turn_request=TurnStartRequest(
            conversation_id=3,
            request_id=uuid4(),
            input=MessageBundlePayload(
                role=ChatRoleEnum.user,
                blocks=[TextBlock(text=query)],
            ),
        ),
        resolved_selection=ResourceSelection(),
        resolved_actions=[
            ExecutionAction(
                action_id="builtin:tool",
                kind=ChatActionKindEnum.tool_call,
                step_kind=ChatStepKindEnum.tool,
                name="tool_router",
                config=ToolCallConfig(tools=[ToolSpec(tool_name="session_context")]),
                priority=10,
                source="builtin",
            ),
        ],
    )


@pytest.mark.asyncio
async def test_function_tool_registry_executes_math_expression() -> None:
    registry = create_default_tool_registry()
    session = _session("请计算 12 + 7 * 2")
    execution = await registry.execute(
        "calculate_expression",
        session=session,
        force=False,
    )

    assert execution is not None
    definition, result = execution
    assert definition.name == "calculate_expression"
    assert registry.resolve_result_mode(
        ToolSpec(tool_name="calculate_expression"),
        result,
    ) == ToolResultModeEnum.terminal
    assert result.text is not None
    assert "26" in result.text


@pytest.mark.asyncio
async def test_function_tool_registry_executes_session_context() -> None:
    registry = create_default_tool_registry()
    session = _session("请输出当前会话上下文")
    execution = await registry.execute(
        "session_context",
        session=session,
        force=False,
    )

    assert execution is not None
    definition, result = execution
    assert definition.name == "session_context"
    assert (
        registry.resolve_result_mode(
            ToolSpec(tool_name="session_context", result_mode=ToolResultModeEnum.context),
            result,
        )
        == ToolResultModeEnum.context
    )
    assert result.data is not None
    assert result.data["conversation_id"] == 3
    assert result.data["request_id"] == str(session.turn_request.request_id)
