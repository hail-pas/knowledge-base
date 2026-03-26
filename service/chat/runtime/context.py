from __future__ import annotations

from dataclasses import field, dataclass

from pydantic import JsonValue

from service.chat.domain.schema import (
    ActionContextItem,
    ChatContextEnvelope,
    ActionTerminalOutput,
    ChatHistoryItem,
    JsonContextItem,
    MessageBundlePayload,
    PromptContextPayload,
    RetrievalBlock,
    RetrievalContextItem,
    TextContextItem,
    UsagePayload,
    ToolExecutionSummary,
)
from service.chat.execution.registry import ExecutionAction


@dataclass(slots=True)
class TurnArtifacts:
    instructions: list[str] = field(default_factory=list)
    context_items: list[ActionContextItem] = field(default_factory=list)
    prompt_context: PromptContextPayload | None = None
    executed_tools: list[ToolExecutionSummary] = field(default_factory=list)
    terminal_output: ActionTerminalOutput | None = None
    usage: UsagePayload | None = None

    def add_instruction(self, instruction: str) -> None:
        instruction = instruction.strip()
        if instruction:
            self.instructions.append(instruction)

    def add_text_context(
        self,
        action: ExecutionAction,
        *,
        text: str,
        title: str | None = None,
    ) -> None:
        text = text.strip()
        if not text:
            return
        self.context_items.append(
            TextContextItem(
                action_id=action.action_id,
                action_kind=action.kind,
                action_name=action.name,
                source=action.source,
                title=title or action.name,
                priority=action.priority,
                text=text,
            ),
        )

    def add_json_context(
        self,
        action: ExecutionAction,
        *,
        data: JsonValue,
        title: str | None = None,
    ) -> None:
        self.context_items.append(
            JsonContextItem[JsonValue](
                action_id=action.action_id,
                action_kind=action.kind,
                action_name=action.name,
                source=action.source,
                title=title or action.name,
                priority=action.priority,
                data=data,
            ),
        )

    def add_retrieval_context(
        self,
        action: ExecutionAction,
        *,
        retrievals: list[RetrievalBlock],
        title: str | None = None,
    ) -> None:
        if not retrievals:
            return
        self.context_items.append(
            RetrievalContextItem(
                action_id=action.action_id,
                action_kind=action.kind,
                action_name=action.name,
                source=action.source,
                title=title or action.name,
                priority=action.priority,
                retrievals=retrievals,
            ),
        )

    def set_prompt_context(self, prompt_context: PromptContextPayload) -> None:
        self.prompt_context = prompt_context

    def add_tool_execution(self, execution: ToolExecutionSummary) -> None:
        self.executed_tools.append(execution)

    def set_terminal_output(
        self,
        action: ExecutionAction,
        *,
        payload: MessageBundlePayload,
        output_data_id: int | None = None,
    ) -> None:
        self.terminal_output = ActionTerminalOutput(
            action_id=action.action_id,
            action_kind=action.kind,
            action_name=action.name,
            source=action.source,
            data_id=output_data_id,
            payload=payload,
        )

    def set_usage(self, usage: UsagePayload) -> None:
        self.usage = usage

    def build_context(self, history: list[ChatHistoryItem]) -> ChatContextEnvelope:
        return ChatContextEnvelope(
            history=history,
            instructions=list(self.instructions),
            context_items=list(self.context_items),
            prompt_context=self.prompt_context,
            executed_tools=list(self.executed_tools),
            terminal_output=self.terminal_output,
        )
