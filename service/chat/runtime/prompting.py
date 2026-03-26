from __future__ import annotations

import json
from dataclasses import dataclass

from service.chat.domain.schema import (
    DEFAULT_SYSTEM_PROMPT_PLACEHOLDERS,
    ActionContextItem,
    ChatActionKindEnum,
    ChatContextEnvelope,
    JsonContextItem,
    RetrievalContextItem,
    SystemPromptPlaceholderEnum,
    TextContextItem,
)
from service.chat.runtime.session import ChatSessionContext


@dataclass(slots=True)
class ChatPromptBundle:
    system_prompt: str
    user_prompt: str


class ChatPromptBuilder:
    def build(
        self,
        *,
        query: str,
        context: ChatContextEnvelope,
        session: ChatSessionContext | None = None,
    ) -> ChatPromptBundle:
        system_sections = self.build_system_sections(context=context, session=session)

        user_sections = [f"用户问题：{query}"]
        if context.history:
            user_sections.append(
                "历史对话：\n"
                + "\n".join(f"用户：{item.user_text}\n助手：{item.assistant_text}" for item in context.history[-6:]),
            )
        ordered_context = context.ordered_context_items()
        if ordered_context:
            user_sections.append(
                "执行上下文：\n"
                + "\n\n".join(
                    self.render_context_item(index=index, item=item)
                    for index, item in enumerate(ordered_context, start=1)
                ),
            )
        user_sections.append("输出要求：优先吸收执行上下文形成答案；无法确认时直接说明不确定。")
        return ChatPromptBundle(
            system_prompt="\n\n".join(section for section in system_sections if section.strip()),
            user_prompt="\n\n".join(section for section in user_sections if section.strip()),
        )

    def build_system_sections(
        self,
        *,
        context: ChatContextEnvelope,
        session: ChatSessionContext | None,
    ) -> list[str]:
        prompt_context = context.prompt_context
        overrides = prompt_context.variable_overrides if prompt_context else {}
        sections = [
            overrides.get("assistant_identity", "你是当前知识库系统中的聊天助手。"),
            overrides.get(
                "response_policy",
                "回答要求：准确、简洁、结构清晰；如果上下文不足，需要明确说明。",
            ),
        ]
        agent_overlay = self.render_agent_overlay(session=session)
        capability_overlay = self.render_capability_overlay(session=session)
        if agent_overlay:
            sections.append(agent_overlay)
        if capability_overlay:
            sections.append(capability_overlay)
        placeholders = list(prompt_context.placeholders) if prompt_context else list(DEFAULT_SYSTEM_PROMPT_PLACEHOLDERS)
        for placeholder in placeholders:
            content = self.render_placeholder(
                placeholder=placeholder,
                context=context,
                session=session,
                overrides=overrides,
            )
            if content:
                sections.append(content)
        return sections

    def render_agent_overlay(self, *, session: ChatSessionContext | None) -> str:
        if session is None or session.agent is None:
            return ""
        return session.agent.system_prompt.strip()

    def render_capability_overlay(self, *, session: ChatSessionContext | None) -> str:
        if session is None or not session.selected_capabilities:
            return ""
        lines: list[str] = []
        for item in session.selected_capabilities:
            summary = item.description or item.name
            constraints = f"；约束：{', '.join(item.constraints)}" if item.constraints else ""
            instructions = f"；指令：{', '.join(item.instructions)}" if item.instructions else ""
            lines.append(f"- {item.name} [{item.runtime_kind.value}] {summary}{constraints}{instructions}")
        return "已选能力 Overlay：\n" + "\n".join(lines)

    def render_placeholder(
        self,
        *,
        placeholder: SystemPromptPlaceholderEnum,
        context: ChatContextEnvelope,
        session: ChatSessionContext | None,
        overrides: dict[str, str],
    ) -> str:
        if placeholder.value in overrides:
            return overrides[placeholder.value]
        if placeholder == SystemPromptPlaceholderEnum.action_summary:
            return self.render_action_summary(context=context, session=session)
        if placeholder == SystemPromptPlaceholderEnum.tool_summary:
            return self.render_tool_summary(context=context)
        if placeholder == SystemPromptPlaceholderEnum.conversation_summary:
            return self.render_conversation_summary(context=context, session=session)
        if placeholder == SystemPromptPlaceholderEnum.instructions_summary:
            return self.render_instruction_summary(context=context)
        if placeholder == SystemPromptPlaceholderEnum.context_policy:
            return (
                "上下文策略：优先使用执行步骤输出；检索结果是证据片段；"
                "工具/MCP 结果是结构化事实；历史对话只用于补充语境，不覆盖最新执行结果。"
            )
        return ""

    def render_action_summary(
        self,
        *,
        context: ChatContextEnvelope,
        session: ChatSessionContext | None,
    ) -> str:
        if session is None:
            capabilities = [
                (
                    item.action_name,
                    item.action_kind,
                    item.source,
                )
                for item in context.ordered_context_items()
            ]
        else:
            if session.selected_capabilities:
                return "当前能力计划：\n" + "\n".join(
                    f"- {item.name} ({item.capability_key} / {item.runtime_kind.value})"
                    for item in session.selected_capabilities
                )
            highlighted = set(session.prompt_state.highlight_actions)
            raw_items = session.resolved_actions
            if highlighted:
                filtered = [item for item in raw_items if item.kind in highlighted]
                raw_items = filtered or raw_items
            capabilities = [(item.name, item.kind, item.source) for item in raw_items]
        if not capabilities:
            return ""
        lines = [
            f"- {name} ({kind.value} / {source})"
            for name, kind, source in capabilities
            if isinstance(kind, ChatActionKindEnum)
        ]
        return "当前执行计划：\n" + "\n".join(lines)

    def render_tool_summary(self, *, context: ChatContextEnvelope) -> str:
        if not context.executed_tools:
            return ""
        lines = []
        for item in context.executed_tools:
            summary = item.summary or "无摘要"
            lines.append(f"- {item.tool_name} [{item.disposition.value}] {summary}")
        return "已执行工具：\n" + "\n".join(lines)

    def render_conversation_summary(
        self,
        *,
        context: ChatContextEnvelope,
        session: ChatSessionContext | None,
    ) -> str:
        if session is None:
            return ""
        return (
            "会话摘要：\n"
            f"- conversation_id: {session.conversation.id}\n"
            f"- title: {session.conversation.title}\n"
            f"- agent_key: {session.conversation.agent_key}\n"
            f"- history_turns: {len(context.history)}"
        )

    def render_instruction_summary(self, *, context: ChatContextEnvelope) -> str:
        instructions = [item.strip() for item in context.instructions if item.strip()]
        prompt_instructions = context.prompt_context.instructions if context.prompt_context else []
        instructions.extend(item.strip() for item in prompt_instructions if item.strip())
        deduplicated: list[str] = []
        for item in instructions:
            if item and item not in deduplicated:
                deduplicated.append(item)
        if not deduplicated:
            return ""
        return "额外执行约束：\n" + "\n".join(f"- {item}" for item in deduplicated)

    def render_context_item(self, *, index: int, item: ActionContextItem) -> str:
        header = f"[{index}] {item.title or item.action_name} ({item.action_kind.value} / {item.source})"
        if isinstance(item, TextContextItem):
            body = item.text
        elif isinstance(item, JsonContextItem):
            body = json.dumps(item.data, ensure_ascii=False, indent=2)
        else:
            assert isinstance(item, RetrievalContextItem)
            body = "\n\n".join(
                f"[{entry_index}] 集合{retrieval.collection_id} 分数{retrieval.score:.3f}\n{retrieval.snippet}"
                for entry_index, retrieval in enumerate(item.retrievals, start=1)
            )
        return f"{header}\n{body}".strip()
