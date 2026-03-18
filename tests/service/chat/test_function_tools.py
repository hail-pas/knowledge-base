import pytest

from service.chat.capability.registry import CapabilityDescriptor
from service.chat.domain.schema import (
    ChatCapabilityKindEnum,
    ChatRoleEnum,
    ConversationSummary,
    FunctionCallConfig,
    FunctionCallResultModeEnum,
    FunctionToolSpec,
    MessageBundlePayload,
    ResourceSelection,
    TextBlock,
    TurnStartRequest,
)
from service.chat.runtime.function_tools import create_default_function_tool_registry
from service.chat.runtime.session import ChatSessionContext
from ext.ext_tortoise.enums import ChatStepKindEnum


def _session(query: str) -> ChatSessionContext:
    return ChatSessionContext(
        account_id=1,
        is_staff=False,
        ws_session_id=10,
        ws_public_session_id="session-public",
        conversation=ConversationSummary(
            id=3,
            title="函数测试会话",
            status="active",
            user_id=1,
            default_resource_selection=ResourceSelection(),
        ),
        turn_request=TurnStartRequest(
            conversation_id=3,
            request_id="req-function-test",
            input=MessageBundlePayload(
                role=ChatRoleEnum.user,
                blocks=[TextBlock(text=query)],
            ),
        ),
        resolved_selection=ResourceSelection(),
        resolved_capabilities=[
            CapabilityDescriptor(
                capability_id="builtin:function",
                kind=ChatCapabilityKindEnum.function_call,
                step_kind=ChatStepKindEnum.tool,
                name="function_router",
                config=FunctionCallConfig(tools=[FunctionToolSpec(tool_name="session_context")]),
                priority=10,
                source="builtin",
            ),
        ],
    )


@pytest.mark.asyncio
async def test_function_tool_registry_executes_math_expression() -> None:
    registry = create_default_function_tool_registry()
    session = _session("请计算 12 + 7 * 2")
    execution = await registry.execute(
        FunctionToolSpec(tool_name="calculate_expression"),
        session=session,
    )

    assert execution is not None
    definition, result = execution
    assert definition.name == "calculate_expression"
    assert registry.resolve_result_mode(
        definition,
        FunctionToolSpec(tool_name="calculate_expression"),
        result,
    ) == FunctionCallResultModeEnum.terminal
    assert result.text is not None
    assert "26" in result.text


@pytest.mark.asyncio
async def test_function_tool_registry_executes_session_context() -> None:
    registry = create_default_function_tool_registry()
    session = _session("请输出当前会话上下文")
    execution = await registry.execute(
        FunctionToolSpec(tool_name="session_context", result_mode=FunctionCallResultModeEnum.context),
        session=session,
    )

    assert execution is not None
    definition, result = execution
    assert definition.name == "session_context"
    assert result.data is not None
    assert result.data["conversation_id"] == 3
    assert result.data["request_id"] == "req-function-test"
