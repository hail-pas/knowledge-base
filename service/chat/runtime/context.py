from __future__ import annotations

from typing import Any
from dataclasses import field, dataclass

from service.chat.domain.schema import (
    UsagePayload,
    RetrievalBlock,
    ChatHistoryItem,
    ActionContextItem,
    ChatContextEnvelope,
    ActionTerminalOutput,
    MessageBundlePayload,
    PromptContextPayload,
    ChatContextItemTypeEnum,
    IntentRecognitionResult,
    FunctionExecutionSummary,
)
from service.chat.execution.registry import ExecutionAction


@dataclass(slots=True)
class TurnArtifacts:
    instructions: list[str] = field(default_factory=list)
    context_items: list[ActionContextItem] = field(default_factory=list)
    prompt_context: PromptContextPayload | None = None
    intent_result: IntentRecognitionResult | None = None
    executed_functions: list[FunctionExecutionSummary] = field(default_factory=list)
    terminal_output: ActionTerminalOutput | None = None
    output_payload: MessageBundlePayload | None = None
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
        metadata: dict[str, Any] | None = None,
    ) -> None:
        text = text.strip()
        if not text:
            return
        self.context_items.append(
            ActionContextItem(
                action_id=action.action_id,
                action_kind=action.kind,
                action_name=action.name,
                source=action.source,
                item_type=ChatContextItemTypeEnum.text,
                title=title or action.name,
                priority=action.priority,
                text=text,
                metadata=metadata or {},
            ),
        )

    def add_json_context(
        self,
        action: ExecutionAction,
        *,
        data: dict[str, Any],
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.context_items.append(
            ActionContextItem(
                action_id=action.action_id,
                action_kind=action.kind,
                action_name=action.name,
                source=action.source,
                item_type=ChatContextItemTypeEnum.json,
                title=title or action.name,
                priority=action.priority,
                data=data,
                metadata=metadata or {},
            ),
        )

    def add_retrieval_context(
        self,
        action: ExecutionAction,
        *,
        retrievals: list[RetrievalBlock],
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not retrievals:
            return
        self.context_items.append(
            ActionContextItem(
                action_id=action.action_id,
                action_kind=action.kind,
                action_name=action.name,
                source=action.source,
                item_type=ChatContextItemTypeEnum.retrieval,
                title=title or action.name,
                priority=action.priority,
                retrievals=retrievals,
                metadata=metadata or {},
            ),
        )

    def set_prompt_context(self, prompt_context: PromptContextPayload) -> None:
        self.prompt_context = prompt_context

    def set_intent_result(self, result: IntentRecognitionResult) -> None:
        self.intent_result = result

    def add_function_execution(self, execution: FunctionExecutionSummary) -> None:
        self.executed_functions.append(execution)

    def set_terminal_output(
        self,
        action: ExecutionAction,
        *,
        payload: MessageBundlePayload,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.terminal_output = ActionTerminalOutput(
            action_id=action.action_id,
            action_kind=action.kind,
            action_name=action.name,
            source=action.source,
            payload=payload,
            metadata=metadata or {},
        )
        self.output_payload = payload

    def set_usage(self, usage: UsagePayload) -> None:
        self.usage = usage

    def build_context(self, history: list[ChatHistoryItem]) -> ChatContextEnvelope:
        return ChatContextEnvelope(
            history=history,
            instructions=list(self.instructions),
            context_items=list(self.context_items),
            prompt_context=self.prompt_context,
            intent_result=self.intent_result,
            executed_functions=list(self.executed_functions),
            terminal_output=self.terminal_output,
        )
