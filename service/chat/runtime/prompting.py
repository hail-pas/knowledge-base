from __future__ import annotations

import json
from dataclasses import dataclass

from service.chat.domain.schema import (
    DEFAULT_SYSTEM_PROMPT_PLACEHOLDERS,
    ChatContextEnvelope,
    CapabilityContextItem,
    ChatCapabilityKindEnum,
    ChatContextItemTypeEnum,
    SystemPromptPlaceholderEnum,
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
                "Capability 上下文：\n"
                + "\n\n".join(
                    self.render_context_item(index=index, item=item)
                    for index, item in enumerate(ordered_context, start=1)
                ),
            )
        if context.references and not ordered_context:
            user_sections.append(
                "知识库参考：\n"
                + "\n\n".join(
                    f"[{idx}] 集合{item.collection_id} 分数{item.score:.3f}\n{item.snippet}"
                    for idx, item in enumerate(context.references, start=1)
                ),
            )
        user_sections.append("输出要求：优先吸收 capability 上下文形成答案；无法确认时直接说明不确定。")
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
        if placeholder == SystemPromptPlaceholderEnum.capability_summary:
            return self.render_capability_summary(context=context, session=session)
        if placeholder == SystemPromptPlaceholderEnum.intent_summary:
            return self.render_intent_summary(context=context)
        if placeholder == SystemPromptPlaceholderEnum.function_summary:
            return self.render_function_summary(context=context)
        if placeholder == SystemPromptPlaceholderEnum.conversation_summary:
            return self.render_conversation_summary(context=context, session=session)
        if placeholder == SystemPromptPlaceholderEnum.instructions_summary:
            return self.render_instruction_summary(context=context)
        if placeholder == SystemPromptPlaceholderEnum.context_policy:
            return (
                "上下文策略：优先使用 capability 输出；检索结果是证据片段；"
                "函数/工具结果是结构化事实；历史对话只用于补充语境，不覆盖最新 capability 结果。"
            )
        return ""

    def render_capability_summary(
        self,
        *,
        context: ChatContextEnvelope,
        session: ChatSessionContext | None,
    ) -> str:
        if session is None:
            capabilities = [
                (
                    item.capability_name,
                    item.capability_kind,
                    item.source,
                )
                for item in context.ordered_context_items()
            ]
        else:
            highlighted = set(session.prompt_state.highlight_capabilities)
            raw_items = session.resolved_capabilities
            if highlighted:
                filtered = [item for item in raw_items if item.kind in highlighted]
                raw_items = filtered or raw_items
            capabilities = [(item.name, item.kind, item.source) for item in raw_items]
        if not capabilities:
            return ""
        lines = [
            f"- {name} ({kind.value} / {source})"
            for name, kind, source in capabilities
            if isinstance(kind, ChatCapabilityKindEnum)
        ]
        return "当前 capability 计划：\n" + "\n".join(lines)

    def render_intent_summary(self, *, context: ChatContextEnvelope) -> str:
        if context.intent_result is None:
            return ""
        matched_keywords = ", ".join(context.intent_result.matched_keywords) or "无"
        description = context.intent_result.description or "未提供描述"
        return (
            "意图识别结果：\n"
            f"- intent: {context.intent_result.intent}\n"
            f"- confidence: {context.intent_result.confidence:.2f}\n"
            f"- matched_keywords: {matched_keywords}\n"
            f"- description: {description}"
        )

    def render_function_summary(self, *, context: ChatContextEnvelope) -> str:
        if not context.executed_functions:
            return ""
        lines = []
        for item in context.executed_functions:
            status = "matched" if item.matched else "not_matched"
            summary = item.summary or "无摘要"
            lines.append(f"- {item.tool_name} [{status} / {item.result_mode.value}] {summary}")
        return "已执行函数：\n" + "\n".join(lines)

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
            f"- status: {session.conversation.status}\n"
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

    def render_context_item(self, *, index: int, item: CapabilityContextItem) -> str:
        header = f"[{index}] {item.title or item.capability_name} ({item.capability_kind.value} / {item.source})"
        if item.item_type == ChatContextItemTypeEnum.text:
            body = item.text or ""
        elif item.item_type == ChatContextItemTypeEnum.json:
            body = json.dumps(item.data or {}, ensure_ascii=False, indent=2)
        else:
            body = "\n\n".join(
                f"[{entry_index}] 集合{retrieval.collection_id} 分数{retrieval.score:.3f}\n{retrieval.snippet}"
                for entry_index, retrieval in enumerate(item.retrievals, start=1)
            )
        return f"{header}\n{body}".strip()
