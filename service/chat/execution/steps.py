from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable

from pydantic import JsonValue

from ext.ext_tortoise.enums import ChatDataKindEnum, ChatStepKindEnum, ChatStepStatusEnum
from service.chat.domain.schema import (
    ActionResultDispositionEnum,
    ChatErrorCodeEnum,
    ChatPayloadTypeEnum,
    ErrorPayload,
    EventNameEnum,
    MessageBundlePayload,
    PersistedStepIOPayload,
    StepEventPayload,
    StepIOPhaseEnum,
)
from service.chat.execution.registry import ExecutionAction
from service.chat.runtime.session import ChatSessionContext
from service.chat.store.repository import ChatRepository

LiveEmitter = Callable[[str, Any], Awaitable[None]]
ResultApplier = Callable[..., None]
ExceptionMessageBuilder = Callable[[Exception], str]
CancellationChecker = Callable[[BaseException], bool]


@dataclass(slots=True)
class StartedActionStep:
    manager: ChatExecutionStepManager
    action: ExecutionAction
    turn: Any
    session_context: ChatSessionContext
    emit: LiveEmitter
    step: Any

    async def complete(
        self,
        *,
        output_payload: Any,
        output_payload_type: ChatPayloadTypeEnum,
    ):
        return await self.manager.complete_step(
            step=self.step,
            turn=self.turn,
            session_context=self.session_context,
            emit=self.emit,
            output_payload=output_payload,
            output_payload_type=output_payload_type,
        )

    async def fail(
        self,
        *,
        message: str,
        code: ChatErrorCodeEnum = ChatErrorCodeEnum.chat_error,
        retryable: bool = False,
    ):
        return await self.manager.fail_step(
            step=self.step,
            turn=self.turn,
            session_context=self.session_context,
            emit=self.emit,
            message=message,
            code=code,
            retryable=retryable,
        )

    async def fail_for_exception(self, exc: Exception, *, fallback: str) -> None:
        if self.manager.is_cancellation_exception(exc):
            raise exc
        await self.fail(
            message=self.manager.exception_message(exc, fallback=fallback),
        )
        raise exc

    def apply_result(
        self,
        *,
        title: str | None,
        disposition: ActionResultDispositionEnum,
        text: str | None,
        summary: str | None,
        data: JsonValue | None,
        terminal_payload: MessageBundlePayload | None,
        fallback: str,
        output_data_id: int | None = None,
    ) -> None:
        self.manager.apply_result(
            self.action,
            session_context=self.session_context,
            title=title,
            disposition=disposition,
            text=text,
            summary=summary,
            data=data,
            terminal_payload=terminal_payload,
            fallback=fallback,
            output_data_id=output_data_id,
        )


class ChatExecutionStepManager:
    def __init__(
        self,
        repository: ChatRepository,
        *,
        apply_result: ResultApplier,
        exception_message: ExceptionMessageBuilder,
        is_cancellation_exception: CancellationChecker,
    ) -> None:
        self.repository = repository
        self.apply_result = apply_result
        self.exception_message = exception_message
        self.is_cancellation_exception = is_cancellation_exception

    def next_step_sequence(self, session_context: ChatSessionContext) -> int:
        current = int(session_context.get_state("next_step_sequence", 0)) + 10
        session_context.set_state("next_step_sequence", current)
        return current

    def build_action_step_input(
        self,
        action: ExecutionAction,
        *,
        message: str,
        data: JsonValue | None = None,
    ) -> PersistedStepIOPayload:
        return PersistedStepIOPayload(
            phase=StepIOPhaseEnum.input,
            action_id=action.action_id,
            action_name=action.name,
            action_kind=action.kind,
            message=message,
            data=data,
        )

    async def start_step(
        self,
        *,
        turn,
        session_context: ChatSessionContext,
        action: ExecutionAction | None = None,
        name: str,
        kind: ChatStepKindEnum,
        emit: LiveEmitter,
        input_payload: Any | None = None,
        input_payload_type: ChatPayloadTypeEnum | None = None,
        parent_step_id: int | None = None,
    ):
        action_metadata = (
            action.metadata.model_dump(
                mode="json",
                exclude_none=True,
                exclude_unset=True,
            )
            if action is not None
            else {}
        )
        step = await self.repository.create_step(
            conversation_id=session_context.conversation_id,
            turn_id=turn.id,
            name=name,
            kind=kind,
            sequence=self.next_step_sequence(session_context),
            parent_step_id=parent_step_id,
            capability_key=(
                str(action.metadata.capability.capability_key)
                if (
                    action is not None
                    and action.metadata.capability is not None
                    and action.metadata.capability.capability_key
                )
                else None
            ),
            operation_key=action.action_id if action is not None else None,
            metadata=(
                {
                    "action_id": action.action_id,
                    "action_kind": action.kind.value,
                    "action_name": action.name,
                    "action_source": action.source,
                    "action_priority": action.priority,
                    **({"action_metadata": action_metadata} if action_metadata else {}),
                }
                if action is not None
                else {}
            ),
        )
        session_context.track_active_step(step.id)
        step_summary = await self.repository.summarize_step(step)
        input_data_summary = None
        if input_payload is not None and input_payload_type is not None:
            input_row = await self.repository.create_data(
                conversation_id=session_context.conversation_id,
                turn_id=turn.id,
                step_id=step.id,
                kind=ChatDataKindEnum.input,
                payload_type=input_payload_type,
                payload=input_payload,
            )
            input_data_summary = await self.repository.summarize_data(input_row)
        await emit(
            EventNameEnum.step_started.value,
            StepEventPayload(step=step_summary, data=input_data_summary),
        )
        return step

    async def start_action_step(
        self,
        action: ExecutionAction,
        *,
        turn,
        session_context: ChatSessionContext,
        emit: LiveEmitter,
        parent_step_id: int | None,
        message: str,
        data: JsonValue | None = None,
        name: str | None = None,
    ) -> StartedActionStep:
        step = await self.start_step(
            turn=turn,
            session_context=session_context,
            action=action,
            name=name or action.name,
            kind=action.step_kind,
            emit=emit,
            input_payload=self.build_action_step_input(
                action,
                message=message,
                data=data,
            ),
            input_payload_type=ChatPayloadTypeEnum.step_io,
            parent_step_id=parent_step_id,
        )
        return StartedActionStep(
            manager=self,
            action=action,
            turn=turn,
            session_context=session_context,
            emit=emit,
            step=step,
        )

    async def complete_step(
        self,
        *,
        step,
        turn,
        session_context: ChatSessionContext,
        emit: LiveEmitter,
        output_payload: Any,
        output_payload_type: ChatPayloadTypeEnum,
    ):
        output_row = await self.repository.create_data(
            conversation_id=session_context.conversation_id,
            turn_id=turn.id,
            step_id=step.id,
            kind=ChatDataKindEnum.output,
            payload_type=output_payload_type,
            payload=output_payload,
        )
        await self.repository.update_step(
            step,
            status=ChatStepStatusEnum.completed,
            finished_at=datetime.now(UTC),
        )
        session_context.finish_active_step(step.id)
        await emit(
            EventNameEnum.step_completed.value,
            StepEventPayload(
                step=await self.repository.summarize_step(step),
                data=await self.repository.summarize_data(output_row),
            ),
        )
        return output_row

    async def fail_step(
        self,
        *,
        step,
        turn,
        session_context: ChatSessionContext,
        emit: LiveEmitter,
        message: str,
        code: ChatErrorCodeEnum = ChatErrorCodeEnum.chat_error,
        retryable: bool = False,
    ):
        error_payload = ErrorPayload(message=message, code=code, retryable=retryable)
        output_row = await self.repository.create_data(
            conversation_id=session_context.conversation_id,
            turn_id=turn.id,
            step_id=step.id,
            kind=ChatDataKindEnum.output,
            payload_type=ChatPayloadTypeEnum.error,
            payload=error_payload,
        )
        await self.repository.update_step(
            step,
            status=ChatStepStatusEnum.failed,
            finished_at=datetime.now(UTC),
        )
        session_context.finish_active_step(step.id)
        await emit(
            EventNameEnum.step_failed.value,
            StepEventPayload(
                step=await self.repository.summarize_step(step),
                data=await self.repository.summarize_data(output_row),
            ),
        )
        session_context.set_state("last_error_data_id", output_row.id)
        return output_row

    async def finalize_active_steps(
        self,
        *,
        session_context: ChatSessionContext,
        status: ChatStepStatusEnum,
        finished_at: datetime,
    ) -> None:
        step_ids = session_context.active_step_ids
        if not step_ids:
            return
        await self.repository.bulk_update_steps_status(
            step_ids=step_ids,
            status=status,
            finished_at=finished_at,
        )
        for step_id in step_ids:
            session_context.finish_active_step(step_id)
