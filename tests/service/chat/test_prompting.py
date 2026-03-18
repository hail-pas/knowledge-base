from typing import Any, cast

from ext.ext_tortoise.enums import ChatStepKindEnum
from service.chat.capability.registry import CapabilityDescriptor
from service.chat.domain.schema import (
    ChatCapabilityKindEnum,
    ChatRoleEnum,
    ConversationSummary,
    FunctionCallConfig,
    FunctionCallResultModeEnum,
    FunctionExecutionSummary,
    FunctionToolSpec,
    IntentRecognitionResult,
    KnowledgeRetrievalConfig,
    MessageBundlePayload,
    ResourceSelection,
    RetrievalBlock,
    SystemPromptConfig,
    SystemPromptPlaceholderEnum,
    TextBlock,
    ToolCallConfig,
    ToolExecutionPolicyEnum,
    TurnStartRequest,
)
from service.chat.runtime.context import TurnArtifacts
from service.chat.runtime.prompting import ChatPromptBuilder
from service.chat.runtime.session import ChatSessionContext


def _descriptor(
    *,
    capability_id: str,
    kind: str,
    name: str,
    priority: int,
    source: str,
    step_kind: str,
    config: Any,
) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        capability_id=capability_id,
        kind=ChatCapabilityKindEnum(kind),
        step_kind=ChatStepKindEnum(step_kind),
        name=name,
        config=config,
        priority=priority,
        source=source,
    )


def test_turn_artifacts_build_context_with_structured_capability_results() -> None:
    artifacts = TurnArtifacts()
    retrieval_descriptor = _descriptor(
        capability_id="cap:retrieval",
        kind="knowledge_retrieval",
        name="kb_search",
        priority=10,
        source="profile",
        step_kind="retrieval",
        config=KnowledgeRetrievalConfig(collection_ids=[1], top_k=3),
    )
    tool_descriptor = _descriptor(
        capability_id="cap:tool",
        kind="tool_call",
        name="weather_tool",
        priority=20,
        source="binding:conversation",
        step_kind="tool",
        config=ToolCallConfig(policy=ToolExecutionPolicyEnum.optional, tool_names=["weather"]),
    )

    artifacts.add_instruction("回答时优先使用最新 capability 结果。")
    artifacts.add_retrieval_context(
        retrieval_descriptor,
        retrievals=[
            RetrievalBlock(
                source_id="chunk-1",
                collection_id=1,
                score=0.9,
                snippet="这是来自知识库的检索命中。",
            ),
        ],
        title="知识库命中",
    )
    artifacts.add_json_context(
        tool_descriptor,
        data={"city": "Shanghai", "weather": "sunny"},
        title="天气工具结果",
    )
    artifacts.set_terminal_output(
        tool_descriptor,
        payload=MessageBundlePayload(
            role=ChatRoleEnum.assistant,
            blocks=[TextBlock(text="直接返回最终结果。")],
        ),
    )

    context = artifacts.build_context([])

    assert context.instructions == ["回答时优先使用最新 capability 结果。"]
    assert len(context.references) == 1
    assert [item.capability_id for item in context.ordered_context_items()] == [
        "cap:retrieval",
        "cap:tool",
    ]
    assert context.terminal_output is not None
    assert context.terminal_output.payload.text == "直接返回最终结果。"


def test_chat_prompt_builder_renders_context_sections() -> None:
    artifacts = TurnArtifacts()
    retrieval_descriptor = _descriptor(
        capability_id="cap:retrieval",
        kind="knowledge_retrieval",
        name="kb_search",
        priority=10,
        source="profile",
        step_kind="retrieval",
        config=KnowledgeRetrievalConfig(collection_ids=[1], top_k=3),
    )
    tool_descriptor = _descriptor(
        capability_id="cap:tool",
        kind="tool_call",
        name="weather_tool",
        priority=20,
        source="binding:conversation",
        step_kind="tool",
        config=ToolCallConfig(policy=ToolExecutionPolicyEnum.optional, tool_names=["weather"]),
    )
    artifacts.add_instruction("优先使用结构化工具结果。")
    artifacts.add_retrieval_context(
        retrieval_descriptor,
        retrievals=[
            RetrievalBlock(
                source_id="chunk-1",
                collection_id=1,
                score=0.91,
                snippet="知识库中提到上海今天晴朗。",
            ),
        ],
        title="知识库命中",
    )
    artifacts.add_json_context(
        tool_descriptor,
        data={"city": "Shanghai", "weather": "sunny"},
        title="天气工具结果",
    )
    prompt = ChatPromptBuilder().build(
        query="上海今天天气怎么样？",
        context=artifacts.build_context([]),
    )

    assert "额外执行约束" in prompt.system_prompt
    assert "Capability 上下文" in prompt.user_prompt
    assert "知识库命中" in prompt.user_prompt
    assert "天气工具结果" in prompt.user_prompt
    assert '"weather": "sunny"' in prompt.user_prompt


def test_chat_prompt_builder_supports_dynamic_system_prompt_placeholders() -> None:
    artifacts = TurnArtifacts()
    system_prompt_descriptor = _descriptor(
        capability_id="cap:system_prompt",
        kind="system_prompt",
        name="dynamic_prompt",
        priority=5,
        source="profile",
        step_kind="system",
        config=SystemPromptConfig(
            include_placeholders=[
                SystemPromptPlaceholderEnum.intent_summary,
                SystemPromptPlaceholderEnum.function_summary,
                SystemPromptPlaceholderEnum.instructions_summary,
            ],
            instructions=["优先引用函数结果。"],
            variable_overrides={"assistant_identity": "你是一个受控的工作流助手。"},
        ),
    )
    tool_descriptor = _descriptor(
        capability_id="cap:function",
        kind="function_call",
        name="function_router",
        priority=20,
        source="binding:conversation",
        step_kind="tool",
        config=FunctionCallConfig(tools=[FunctionToolSpec(tool_name="session_context")]),
    )
    session = ChatSessionContext(
        account_id=1,
        is_staff=False,
        ws_session_id=10,
        ws_public_session_id="ws-public",
        conversation=ConversationSummary(
            id=9,
            title="调试会话",
            status="active",
            user_id=1,
            default_resource_selection=ResourceSelection(),
        ),
        turn_request=TurnStartRequest(
            conversation_id=9,
            input=MessageBundlePayload(
                role=ChatRoleEnum.user,
                blocks=[TextBlock(text="请基于当前会话上下文回答。")],
            ),
        ),
        resolved_selection=ResourceSelection(),
        resolved_capabilities=[system_prompt_descriptor, tool_descriptor],
        artifacts=artifacts,
    )
    session.apply_system_prompt_config(
        system_prompt_descriptor,
        cast(SystemPromptConfig, system_prompt_descriptor.config),
    )
    artifacts.add_instruction("只输出关键结论。")
    artifacts.set_intent_result(
        IntentRecognitionResult(
            intent="session_context",
            confidence=0.9,
            matched_keywords=["会话上下文"],
            description="用户正在询问当前会话状态",
        ),
    )
    artifacts.add_function_execution(
        FunctionExecutionSummary(
            tool_name="session_context",
            title="会话上下文",
            result_mode=FunctionCallResultModeEnum.context,
            summary="已提取当前会话上下文摘要",
        ),
    )

    prompt = ChatPromptBuilder().build(
        query="请基于当前会话上下文回答。",
        context=artifacts.build_context([]),
        session=session,
    )

    assert "你是一个受控的工作流助手。" in prompt.system_prompt
    assert "意图识别结果" in prompt.system_prompt
    assert "已执行函数" in prompt.system_prompt
    assert "优先引用函数结果。" in prompt.system_prompt
    assert "当前 capability 计划" not in prompt.system_prompt
