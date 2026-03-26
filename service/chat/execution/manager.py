from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable
from datetime import UTC, datetime

from pydantic import JsonValue

from ext.ext_tortoise.enums import ChatDataKindEnum, ChatStepStatusEnum, ChatStepKindEnum
from service.chat.domain.schema import (
    ActionResultDispositionEnum,
    ChatActionKindEnum,
    ChatContextEnvelope,
    ChatPayloadTypeEnum,
    ChatRoleEnum,
    ChatWarningCodeEnum,
    EventNameEnum,
    MessageBundlePayload,
    PromptContextPayload,
    ResourceSelection,
    SystemPromptConfig,
    TextBlock,
    ToolSpec,
    UsagePayload,
    WarningPayload,
)
from service.chat.domain.errors import ChatCancelledError
from service.chat.execution.registry import (
    ExecutionAction,
    ExecutionActionRegistry,
    create_default_action_registry,
)
from service.chat.execution.agents import ChatAgentActionExecutor
from service.chat.execution.steps import ChatExecutionStepManager
from service.chat.execution.tooling import ChatToolActionExecutor
from service.chat.runtime.mcp_executor import MCPServerRegistry, create_default_mcp_registry
from service.chat.runtime.prompting import ChatPromptBuilder
from service.chat.runtime.session import ChatSessionContext
from service.chat.runtime.tool_executor import (
    ToolRegistry,
    create_default_tool_registry,
)
from service.chat.store.repository import ChatRepository

LiveEmitter = Callable[[str, Any], Awaitable[None]]
ActionCancellationChecker = Callable[[asyncio.Event], Awaitable[None]]


class _ActionExecutionRequest:
    __slots__ = (
        "action",
        "turn",
        "session_context",
        "cancel_event",
        "emit",
        "ensure_not_canceled",
        "parent_step_id",
    )

    action: ExecutionAction
    turn: Any
    session_context: ChatSessionContext
    cancel_event: asyncio.Event
    emit: LiveEmitter
    ensure_not_canceled: ActionCancellationChecker
    parent_step_id: int | None

    def __init__(
        self,
        *,
        action: ExecutionAction,
        turn: Any,
        session_context: ChatSessionContext,
        cancel_event: asyncio.Event,
        emit: LiveEmitter,
        ensure_not_canceled: ActionCancellationChecker,
        parent_step_id: int | None = None,
    ) -> None:
        self.action = action
        self.turn = turn
        self.session_context = session_context
        self.cancel_event = cancel_event
        self.emit = emit
        self.ensure_not_canceled = ensure_not_canceled
        self.parent_step_id = parent_step_id


class ChatExecutionManager:
    def __init__(
        self,
        repository: ChatRepository,
        action_registry: ExecutionActionRegistry | None = None,
        tool_registry: ToolRegistry | None = None,
        mcp_registry: MCPServerRegistry | None = None,
        prompt_builder: ChatPromptBuilder | None = None,
    ) -> None:
        self.repository = repository
        self.action_registry = action_registry or create_default_action_registry()
        self.tool_registry = tool_registry or create_default_tool_registry()
        self.mcp_registry = mcp_registry or create_default_mcp_registry()
        self.prompt_builder = prompt_builder or ChatPromptBuilder()
        self.step_manager = ChatExecutionStepManager(
            repository=self.repository,
            apply_result=self.apply_result_to_session_artifacts,
            exception_message=self.exception_message,
            is_cancellation_exception=self.is_cancellation_exception,
        )
        self.tool_action_executor = ChatToolActionExecutor(
            tool_registry=self.tool_registry,
            mcp_registry=self.mcp_registry,
            step_manager=self.step_manager,
            emit_warning=self._emit_warning_adapter,
        )
        self.agent_action_executor = ChatAgentActionExecutor(
            repository=self.repository,
            prompt_builder=self.prompt_builder,
            step_manager=self.step_manager,
        )
        self._action_handlers: dict[ChatActionKindEnum, Callable[[_ActionExecutionRequest], Awaitable[None]]] = {
            ChatActionKindEnum.system_prompt: self._run_system_prompt_action,
            ChatActionKindEnum.tool_call: self._run_tool_action,
            ChatActionKindEnum.mcp_call: self._run_mcp_action,
            ChatActionKindEnum.sub_agent_call: self._run_sub_agent_action,
            ChatActionKindEnum.llm_response: self._run_llm_action,
        }

    def is_required_action(self, action: ExecutionAction) -> bool:
        return bool(
            action.metadata.capability is not None and action.metadata.capability.capability_required,
        )

    def should_stop_after_terminal(self, action: ExecutionAction) -> bool:
        config = action.config
        if isinstance(config, ToolCallConfig):
            return config.stop_after_terminal
        return True

    def build_terminal_payload(
        self,
        *,
        text: str | None,
        summary: str | None,
        fallback: str,
    ) -> MessageBundlePayload:
        return MessageBundlePayload(
            role=ChatRoleEnum.assistant,
            blocks=[TextBlock(text=text or summary or fallback)],
        )

    def exception_message(self, exc: Exception, *, fallback: str) -> str:
        return str(exc).strip() or fallback

    def is_cancellation_exception(self, exc: BaseException) -> bool:
        return isinstance(exc, (asyncio.CancelledError, ChatCancelledError))

    def apply_result_to_session_artifacts(
        self,
        action: ExecutionAction,
        *,
        session_context: ChatSessionContext,
        title: str | None,
        disposition: ActionResultDispositionEnum,
        text: str | None,
        summary: str | None,
        data: JsonValue | None,
        terminal_payload: MessageBundlePayload | None,
        fallback: str,
        output_data_id: int | None = None,
    ) -> None:
        if disposition == ActionResultDispositionEnum.context:
            if data is not None:
                session_context.artifacts.add_json_context(
                    action,
                    data=data,
                    title=title or action.name,
                )
            elif text:
                session_context.artifacts.add_text_context(
                    action,
                    text=text,
                    title=title or action.name,
                )
            return
        session_context.artifacts.set_terminal_output(
            action,
            payload=terminal_payload or self.build_terminal_payload(text=text, summary=summary, fallback=fallback),
            output_data_id=output_data_id,
        )

    def build_execution_actions_from_resource_selection(
        self,
        *,
        actions: list[Any],
        source: str,
        prefix: str,
    ) -> tuple[ResourceSelection, list[ExecutionAction]]:
        selection = self.action_registry.normalize_inline_selection(
            ResourceSelection(actions=list(actions)),
            source=source,
            prefix=prefix,
        )
        return selection, self.action_registry.build_actions(selection)

    async def emit_warning(self, emit: LiveEmitter, *, message: str, code: ChatWarningCodeEnum) -> None:
        await emit(EventNameEnum.warning.value, WarningPayload(message=message, code=code))

    async def _emit_warning_adapter(
        self,
        emit: LiveEmitter,
        *,
        message: str,
        code: ChatWarningCodeEnum,
    ) -> None:
        await self.emit_warning(emit, message=message, code=code)

    def next_step_sequence(self, session_context: ChatSessionContext) -> int:
        return self.step_manager.next_step_sequence(session_context)

    async def finalize_active_steps(
        self,
        *,
        session_context: ChatSessionContext,
        status: ChatStepStatusEnum,
        finished_at: datetime,
    ) -> None:
        await self.step_manager.finalize_active_steps(
            session_context=session_context,
            status=status,
            finished_at=finished_at,
        )

    async def record_turn_input(
        self,
        *,
        turn,
        session_context: ChatSessionContext,
        emit: LiveEmitter,
    ) -> int:
        step = await self.step_manager.start_step(
            turn=turn,
            session_context=session_context,
            name="user_message",
            kind=ChatStepKindEnum.system,
            emit=emit,
        )
        output_row = await self.step_manager.complete_step(
            step=step,
            turn=turn,
            session_context=session_context,
            emit=emit,
            output_payload=session_context.turn_request.input,
            output_payload_type=ChatPayloadTypeEnum.message_bundle,
        )
        await self.repository.update_turn(turn, input_root_data_id=output_row.id)
        return output_row.id

    async def execute_actions(
        self,
        *,
        turn,
        session_context: ChatSessionContext,
        cancel_event: asyncio.Event,
        emit: LiveEmitter,
        ensure_not_canceled: Callable[[asyncio.Event], Awaitable[None]],
        parent_step_id: int | None = None,
    ) -> None:
        for action in session_context.resolved_actions:
            await ensure_not_canceled(cancel_event)
            if session_context.artifacts.terminal_output is not None and not self.is_required_action(action):
                break
            await self.execute_action(
                _ActionExecutionRequest(
                    action=action,
                    turn=turn,
                    session_context=session_context,
                    cancel_event=cancel_event,
                    emit=emit,
                    ensure_not_canceled=ensure_not_canceled,
                    parent_step_id=parent_step_id,
                ),
            )
            if (
                session_context.artifacts.terminal_output is not None
                and not self.is_required_action(action)
                and self.should_stop_after_terminal(action)
            ):
                break

    async def execute_action(
        self,
        request: _ActionExecutionRequest,
    ) -> None:
        handler = self._action_handlers.get(request.action.kind)
        if handler is None:
            raise ValueError(f"unsupported action kind: {request.action.kind}")
        await handler(request)

    async def _run_system_prompt_action(self, request: _ActionExecutionRequest) -> None:
        await self.execute_system_prompt_action(
            request.action,
            turn=request.turn,
            session_context=request.session_context,
            emit=request.emit,
            parent_step_id=request.parent_step_id,
        )

    async def _run_tool_action(self, request: _ActionExecutionRequest) -> None:
        await self.execute_tool_action(
            request.action,
            turn=request.turn,
            session_context=request.session_context,
            cancel_event=request.cancel_event,
            emit=request.emit,
            ensure_not_canceled=request.ensure_not_canceled,
            parent_step_id=request.parent_step_id,
        )

    async def _run_mcp_action(self, request: _ActionExecutionRequest) -> None:
        await self.execute_mcp_action(
            request.action,
            turn=request.turn,
            session_context=request.session_context,
            emit=request.emit,
            parent_step_id=request.parent_step_id,
        )

    async def _run_sub_agent_action(self, request: _ActionExecutionRequest) -> None:
        await self.execute_sub_agent_action(
            request.action,
            turn=request.turn,
            session_context=request.session_context,
            cancel_event=request.cancel_event,
            emit=request.emit,
            ensure_not_canceled=request.ensure_not_canceled,
            parent_step_id=request.parent_step_id,
        )

    async def _run_llm_action(self, request: _ActionExecutionRequest) -> None:
        await self.execute_llm_action(
            request.action,
            turn=request.turn,
            session_context=request.session_context,
            cancel_event=request.cancel_event,
            emit=request.emit,
            ensure_not_canceled=request.ensure_not_canceled,
            parent_step_id=request.parent_step_id,
        )

    async def execute_system_prompt_action(
        self,
        action: ExecutionAction,
        *,
        turn,
        session_context: ChatSessionContext,
        emit: LiveEmitter,
        parent_step_id: int | None,
    ) -> None:
        assert isinstance(action.config, SystemPromptConfig)
        started_step = await self.step_manager.start_action_step(
            action,
            turn=turn,
            session_context=session_context,
            emit=emit,
            parent_step_id=parent_step_id,
            message="应用系统提示词配置",
            data={
                "instruction_count": len(action.config.instructions),
                "placeholder_count": len(action.config.include_placeholders),
            },
        )
        session_context.apply_system_prompt_config(action, action.config)
        output_row = await started_step.complete(
            output_payload=session_context.prompt_state.to_payload(),
            output_payload_type=ChatPayloadTypeEnum.prompt_context,
        )
        session_context.artifacts.set_prompt_context(
            PromptContextPayload.model_validate(output_row.payload),
        )

    async def plan_tools_with_llm(
        self,
        config: ToolCallConfig,
        *,
        session_context: ChatSessionContext,
        scored_specs: list[tuple[ToolSpec, float]],
    ) -> Any | None:
        return await self.tool_action_executor.plan_tools_with_llm(
            config,
            session_context=session_context,
            scored_specs=scored_specs,
        )

    async def select_tools(
        self,
        action: ExecutionAction,
        *,
        session_context: ChatSessionContext,
    ) -> list[ToolSpec]:
        return await self.tool_action_executor.select_tools(
            action,
            session_context=session_context,
            plan_tools_with_llm=self.plan_tools_with_llm,
        )

    async def execute_tool_action(
        self,
        action: ExecutionAction,
        *,
        turn,
        session_context: ChatSessionContext,
        cancel_event: asyncio.Event,
        emit: LiveEmitter,
        ensure_not_canceled: Callable[[asyncio.Event], Awaitable[None]],
        parent_step_id: int | None,
    ) -> None:
        await self.tool_action_executor.execute_tool_action(
            action,
            turn=turn,
            session_context=session_context,
            cancel_event=cancel_event,
            emit=emit,
            ensure_not_canceled=ensure_not_canceled,
            parent_step_id=parent_step_id,
            selected_tools=await self.select_tools(action, session_context=session_context),
            is_required_action=self.is_required_action(action),
        )

    async def execute_mcp_action(
        self,
        action: ExecutionAction,
        *,
        turn,
        session_context: ChatSessionContext,
        emit: LiveEmitter,
        parent_step_id: int | None,
    ) -> None:
        await self.tool_action_executor.execute_mcp_action(
            action,
            turn=turn,
            session_context=session_context,
            emit=emit,
            parent_step_id=parent_step_id,
        )

    def should_emit_terminal_from_sub_agent(self, action: ExecutionAction) -> bool:
        return self.agent_action_executor.should_emit_terminal_from_sub_agent(action)

    async def execute_sub_agent_action(
        self,
        action: ExecutionAction,
        *,
        turn,
        session_context: ChatSessionContext,
        cancel_event: asyncio.Event,
        emit: LiveEmitter,
        ensure_not_canceled: Callable[[asyncio.Event], Awaitable[None]],
        parent_step_id: int | None,
    ) -> None:
        await self.agent_action_executor.execute_sub_agent_action(
            action,
            turn=turn,
            session_context=session_context,
            cancel_event=cancel_event,
            emit=emit,
            ensure_not_canceled=ensure_not_canceled,
            parent_step_id=parent_step_id,
            build_execution_actions_from_resource_selection=self.build_execution_actions_from_resource_selection,
            execute_actions=self.execute_actions,
            should_emit_terminal_from_sub_agent=self.should_emit_terminal_from_sub_agent,
        )

    async def execute_llm_action(
        self,
        action: ExecutionAction,
        *,
        turn,
        session_context: ChatSessionContext,
        cancel_event: asyncio.Event,
        emit: LiveEmitter,
        ensure_not_canceled: Callable[[asyncio.Event], Awaitable[None]],
        parent_step_id: int | None,
    ) -> None:
        await self.agent_action_executor.execute_llm_action(
            action,
            turn=turn,
            session_context=session_context,
            cancel_event=cancel_event,
            emit=emit,
            ensure_not_canceled=ensure_not_canceled,
            parent_step_id=parent_step_id,
            build_chat_context=self.build_chat_context,
            generate_response=self.generate_response,
        )

    async def generate_response(
        self,
        *,
        query: str,
        llm_model_config_id: int | None,
        context: ChatContextEnvelope,
        session_context: ChatSessionContext,
        cancel_event: asyncio.Event,
        send_delta: Callable[[str], Awaitable[None]],
        ensure_not_canceled: Callable[[asyncio.Event], Awaitable[None]],
        system_prompt_prefix: str | None = None,
        extra_instructions: list[str] | None = None,
        stream: bool = True,
    ) -> tuple[str, UsagePayload]:
        return await self.agent_action_executor.generate_response(
            query=query,
            llm_model_config_id=llm_model_config_id,
            context=context,
            session_context=session_context,
            cancel_event=cancel_event,
            send_delta=send_delta,
            ensure_not_canceled=ensure_not_canceled,
            system_prompt_prefix=system_prompt_prefix,
            extra_instructions=extra_instructions,
            stream=stream,
        )

    async def build_chat_context(
        self,
        *,
        session_context: ChatSessionContext,
        include_history: bool = True,
    ) -> ChatContextEnvelope:
        return await self.agent_action_executor.build_chat_context(
            session_context=session_context,
            include_history=include_history,
        )
