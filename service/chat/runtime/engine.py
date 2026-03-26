from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable
from uuid import UUID

from loguru import logger

from service.chat.capability.schema import CapabilityPlannerModeEnum
from ext.ext_tortoise.enums import (
    ChatDataKindEnum,
    ChatStepKindEnum,
    ChatStepStatusEnum,
    ChatTurnStatusEnum,
    ChatTurnTriggerEnum,
)
from service.chat.domain.schema import CapabilityKindEnum, CapabilityCategoryEnum, CapabilityRuntimeKindEnum
from service.chat.domain.schema import (
    ChatRequestContext,
    ChatErrorCodeEnum,
    ChatEvent,
    ErrorPayload,
    EventNameEnum,
    ChatPayloadTypeEnum,
    StepEventPayload,
    TurnEventPayload,
    TurnStartRequest,
)
from service.chat.domain.errors import ChatCancelledError
from service.chat.execution.manager import ChatExecutionManager
from service.chat.execution.registry import (
    ExecutionActionRegistry,
    create_default_action_registry,
)
from service.chat.runtime.mcp_executor import MCPServerRegistry, create_default_mcp_registry
from service.chat.runtime.planning import (
    RuntimeCapabilityDescriptor,
    RuntimeExecutionPlan,
)
from service.chat.runtime.prompting import ChatPromptBuilder
from service.chat.runtime.session import ChatSessionContext
from service.chat.runtime.tool_executor import ToolRegistry, create_default_tool_registry
from service.chat.store.repository import ChatRepository


EventSender = Callable[[ChatEvent[Any]], Awaitable[None]]


@dataclass
class RunningTurn:
    task: asyncio.Task[None]
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass(slots=True)
class PreparedTurn:
    turn: Any
    session_context: ChatSessionContext


class ChatRuntime:
    def __init__(
        self,
        repository: ChatRepository,
        action_registry: ExecutionActionRegistry | None = None,
        tool_registry: ToolRegistry | None = None,
        mcp_registry: MCPServerRegistry | None = None,
    ) -> None:
        self.repository = repository
        self.running_turns: dict[int, RunningTurn] = {}
        self.pending_turns: dict[int, PreparedTurn] = {}
        self.action_registry = action_registry or create_default_action_registry()
        self.tool_registry = tool_registry or create_default_tool_registry()
        self.mcp_registry = mcp_registry or create_default_mcp_registry()
        self.prompt_builder = ChatPromptBuilder()
        self.execution_manager = ChatExecutionManager(
            repository=self.repository,
            action_registry=self.action_registry,
            tool_registry=self.tool_registry,
            mcp_registry=self.mcp_registry,
            prompt_builder=self.prompt_builder,
        )

    def is_running(self, turn_id: int) -> bool:
        return turn_id in self.running_turns

    async def cancel_turn(self, turn_id: int) -> bool:
        prepared = self.pending_turns.pop(turn_id, None)
        if prepared is not None:
            await self.repository.finalize_turn(
                prepared.turn,
                status=ChatTurnStatusEnum.canceled,
                finished_at=datetime.now(UTC),
            )
            return True
        running = self.running_turns.get(turn_id)
        if not running:
            return False
        running.cancel_event.set()
        running.task.cancel()
        return True

    def start_turn_task(
        self,
        turn_id: int,
        coro: Callable[[asyncio.Event], Awaitable[None]],
    ) -> asyncio.Task[None]:
        cancel_event = asyncio.Event()

        async def runner() -> None:
            try:
                await coro(cancel_event)
            finally:
                self.running_turns.pop(turn_id, None)

        task = asyncio.create_task(runner())
        self.running_turns[turn_id] = RunningTurn(task=task, cancel_event=cancel_event)
        return task

    async def ensure_not_canceled(self, cancel_event: asyncio.Event) -> None:
        if cancel_event.is_set():
            raise ChatCancelledError("turn cancelled by user")

    def clear_current_task_cancellation(self) -> None:
        task = asyncio.current_task()
        if task is None:
            return
        while task.cancelling():
            task.uncancel()

    async def emit_turn_event(
        self,
        *,
        session_id: UUID,
        conversation_id: int,
        turn_id: int,
        send_event: EventSender,
        counter: dict[str, int],
        event: str,
        payload: Any,
    ) -> None:
        seq = counter["value"]
        counter["value"] += 1
        await send_event(
            ChatEvent(
                id=f"evt_{turn_id}_{seq}",
                session_id=session_id,
                conversation_id=conversation_id,
                turn_id=turn_id,
                seq=seq,
                event=event,
                ts=datetime.now(UTC),
                payload=payload,
            ),
        )

    async def persist_unhandled_error_step(
        self,
        *,
        turn,
        session_context: ChatSessionContext,
        emit: Callable[[str, Any], Awaitable[None]],
        message: str,
    ) -> int:
        step = await self.repository.create_step(
            conversation_id=session_context.conversation_id,
            turn_id=turn.id,
            name="turn_error",
            kind=ChatStepKindEnum.system,
            sequence=self.execution_manager.next_step_sequence(session_context),
        )
        error_row = await self.repository.create_data(
            conversation_id=session_context.conversation_id,
            turn_id=turn.id,
            step_id=step.id,
            kind=ChatDataKindEnum.output,
            payload_type=ChatPayloadTypeEnum.error,
            payload=ErrorPayload(message=message, code=ChatErrorCodeEnum.chat_error),
        )
        await self.repository.update_step(
            step,
            status=ChatStepStatusEnum.failed,
            finished_at=datetime.now(UTC),
        )
        await emit(
            EventNameEnum.step_failed.value,
            StepEventPayload(
                step=await self.repository.summarize_step(step),
                data=await self.repository.summarize_data(error_row),
            ),
        )
        return error_row.id

    async def finalize_turn_execution(
        self,
        *,
        turn,
        session_context: ChatSessionContext,
        emit: Callable[[str, Any], Awaitable[None]],
        status: ChatTurnStatusEnum,
        event: EventNameEnum,
        finished_at: datetime,
        output_data_id: int | None = None,
        active_step_status: ChatStepStatusEnum | None = None,
        persist_usage: bool = False,
    ) -> None:
        if active_step_status is not None:
            await self.execution_manager.finalize_active_steps(
                session_context=session_context,
                status=active_step_status,
                finished_at=finished_at,
            )
        finalize_kwargs: dict[str, Any] = {
            "status": status,
            "finished_at": finished_at,
        }
        if output_data_id is not None:
            finalize_kwargs["output_root_data_id"] = output_data_id
        if persist_usage:
            finalize_kwargs["usage"] = session_context.artifacts.usage or {}
        await self.repository.finalize_turn(turn, **finalize_kwargs)
        await emit(
            event.value,
            TurnEventPayload(turn=await self.repository.summarize_turn(turn)),
        )

    async def execute_turn(
        self,
        *,
        context: ChatRequestContext,
        conversation,
        turn_request: TurnStartRequest,
        send_event: EventSender,
        agent=None,
        execution_plan: RuntimeExecutionPlan | None = None,
    ) -> int:
        turn_id = await self.prepare_turn(
            context=context,
            conversation=conversation,
            turn_request=turn_request,
            agent=agent,
            execution_plan=execution_plan,
        )
        await self.launch_prepared_turn(turn_id, send_event=send_event)
        return turn_id

    async def prepare_turn(
        self,
        *,
        context: ChatRequestContext,
        conversation,
        turn_request: TurnStartRequest,
        agent=None,
        execution_plan: RuntimeExecutionPlan | None = None,
    ) -> int:
        resolved_selection = self.action_registry.normalize_selection(turn_request.resource_selection)
        runtime_plan = execution_plan or self.build_fallback_plan(resolved_selection)
        runtime_plan_payload = {
            **runtime_plan.model_dump(mode="json"),
            "selected_capability_keys": runtime_plan.selected_capability_keys,
        }
        resolved_actions = self.action_registry.build_actions(resolved_selection)
        turn = await self.repository.create_turn(
            conversation_id=conversation.id,
            agent_key=turn_request.agent_key or conversation.agent_key,
            request_id=turn_request.request_id,
            trigger=ChatTurnTriggerEnum.user,
            resource_selection=resolved_selection,
            planner_mode=runtime_plan.planner_mode.value,
            planner_summary=runtime_plan.summary,
            execution_plan=runtime_plan_payload,
        )

        session_context = ChatSessionContext(
            request_context=context.with_conversation(conversation.id),
            conversation=conversation,
            turn_request=turn_request.model_copy(
                update={
                    "conversation_id": conversation.id,
                    "agent_key": turn_request.agent_key or conversation.agent_key,
                    "resource_selection": resolved_selection,
                },
            ),
            resolved_selection=resolved_selection,
            resolved_actions=resolved_actions,
            agent=agent,
            execution_plan=runtime_plan,
        )
        self.pending_turns[turn.id] = PreparedTurn(
            turn=turn,
            session_context=session_context,
        )
        return turn.id

    def build_fallback_plan(self, resolved_selection) -> RuntimeExecutionPlan:
        selected_capability_keys = [
            str(item.capability_key) for item in resolved_selection.normalized_capabilities() if item.capability_key
        ]
        selected_capabilities = [
            RuntimeCapabilityDescriptor(
                capability_key=capability_key,
                capability_kind=CapabilityKindEnum.extension,
                category=CapabilityCategoryEnum.domain,
                runtime_kind=CapabilityRuntimeKindEnum.local_toolset,
                name=capability_key,
                explicit=True,
            )
            for capability_key in selected_capability_keys
        ]
        return RuntimeExecutionPlan(
            planner_mode=CapabilityPlannerModeEnum.disabled,
            summary=("使用回退执行计划" if selected_capabilities else "未提供能力计划，回退到基础聊天回答"),
            selected_capabilities=selected_capabilities,
            actions=list(resolved_selection.actions),
        )

    async def launch_prepared_turn(
        self,
        turn_id: int,
        *,
        send_event: EventSender,
    ) -> None:
        prepared = self.pending_turns.pop(turn_id, None)
        if prepared is None:
            raise ValueError(f"turn `{turn_id}` 未准备或已启动")
        turn = prepared.turn
        session_context = prepared.session_context
        session_id = session_context.session_id
        conversation = session_context.conversation
        emit_counter = {"value": 1}

        async def emit(event: str, payload: Any) -> None:
            await self.emit_turn_event(
                session_id=session_id,
                conversation_id=conversation.id,
                turn_id=turn.id,
                send_event=send_event,
                counter=emit_counter,
                event=event,
                payload=payload,
            )

        async def run(cancel_event: asyncio.Event) -> None:
            try:
                await emit(
                    EventNameEnum.turn_started.value,
                    TurnEventPayload(turn=await self.repository.summarize_turn(turn)),
                )
                await self.execution_manager.record_turn_input(
                    turn=turn,
                    session_context=session_context,
                    emit=emit,
                )
                await self.execution_manager.execute_actions(
                    turn=turn,
                    session_context=session_context,
                    cancel_event=cancel_event,
                    emit=emit,
                    ensure_not_canceled=self.ensure_not_canceled,
                )
                output_data_id = (
                    session_context.artifacts.terminal_output.data_id
                    if session_context.artifacts.terminal_output is not None
                    else None
                )
                if output_data_id is None:
                    raise ValueError("turn 未产出最终结果")
                await self.finalize_turn_execution(
                    turn=turn,
                    session_context=session_context,
                    emit=emit,
                    status=ChatTurnStatusEnum.completed,
                    event=EventNameEnum.turn_completed,
                    finished_at=datetime.now(UTC),
                    output_data_id=output_data_id,
                    persist_usage=True,
                )
            except (ChatCancelledError, asyncio.CancelledError):
                self.clear_current_task_cancellation()
                finished_at = datetime.now(UTC)
                await self.finalize_turn_execution(
                    turn=turn,
                    session_context=session_context,
                    emit=emit,
                    status=ChatTurnStatusEnum.canceled,
                    event=EventNameEnum.turn_canceled,
                    finished_at=finished_at,
                    active_step_status=ChatStepStatusEnum.canceled,
                )
            except Exception as exc:
                logger.exception("Chat turn failed: turn_id={}", turn.id)
                finished_at = datetime.now(UTC)
                error_data_id = session_context.get_state("last_error_data_id")
                if error_data_id is None:
                    error_data_id = await self.persist_unhandled_error_step(
                        turn=turn,
                        session_context=session_context,
                        emit=emit,
                        message=str(exc) or "执行失败",
                    )
                await self.finalize_turn_execution(
                    turn=turn,
                    session_context=session_context,
                    emit=emit,
                    status=ChatTurnStatusEnum.failed,
                    event=EventNameEnum.turn_failed,
                    finished_at=finished_at,
                    output_data_id=error_data_id,
                    active_step_status=ChatStepStatusEnum.failed,
                    persist_usage=True,
                )

        self.start_turn_task(turn.id, run)

    async def abort_prepared_turn(self, turn_id: int) -> bool:
        prepared = self.pending_turns.pop(turn_id, None)
        if prepared is None:
            return False
        await self.repository.finalize_turn(
            prepared.turn,
            status=ChatTurnStatusEnum.canceled,
            finished_at=datetime.now(UTC),
        )
        return True
