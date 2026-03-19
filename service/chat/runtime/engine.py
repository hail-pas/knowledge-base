from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Any, Callable, Awaitable
from datetime import UTC, datetime
from dataclasses import field, dataclass

from loguru import logger

from ext.ext_tortoise.enums import (
    ChatDataKindEnum,
    ChatStepKindEnum,
    ChatStepStatusEnum,
    ChatTurnStatusEnum,
    ChatTurnTriggerEnum,
)
from service.chat.execution import (
    ExecutionAction,
    ChatExecutionManager,
    ExecutionActionRegistry,
    create_default_action_registry,
)
from service.chat.domain.schema import (
    ChatEvent,
    EventNameEnum,
    ChatDataSchema,
    DataEventPayload,
    StepEventPayload,
    TurnEventPayload,
    TurnStartRequest,
    ResourceSelection,
    StepMetricPayload,
    ChatActionKindEnum,
    ChatPayloadTypeEnum,
    CapabilityPlanPayload,
    CapabilityPlanCandidate,
)
from service.chat.runtime.session import ChatSessionContext
from service.chat.store.repository import ChatRepository
from service.chat.runtime.prompting import ChatPromptBuilder
from service.chat.runtime.function_tools import (
    FunctionToolRegistry,
    create_default_function_tool_registry,
)


class TurnCancelledError(Exception):
    pass


EventSender = Callable[[ChatEvent[Any]], Awaitable[None]]


@dataclass
class RunningTurn:
    task: asyncio.Task[None]
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)


class ChatRuntime:
    def __init__(
        self,
        repository: ChatRepository,
        action_registry: ExecutionActionRegistry | None = None,
        function_tool_registry: FunctionToolRegistry | None = None,
    ) -> None:
        self.repository = repository
        self.running_turns: dict[int, RunningTurn] = {}
        self.action_registry = action_registry or create_default_action_registry()
        self.function_tool_registry = function_tool_registry or create_default_function_tool_registry()
        self.prompt_builder = ChatPromptBuilder()
        self.execution_manager = ChatExecutionManager(
            repository=self.repository,
            action_registry=self.action_registry,
            function_tool_registry=self.function_tool_registry,
            prompt_builder=self.prompt_builder,
        )

    def is_running(self, turn_id: int) -> bool:
        return turn_id in self.running_turns

    async def cancel_turn(self, turn_id: int) -> bool:
        running = self.running_turns.get(turn_id)
        if not running:
            return False
        running.cancel_event.set()
        running.task.cancel()
        await self.repository.mark_turn_cancel_requested(turn_id)
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
            raise TurnCancelledError

    def clear_current_task_cancellation(self) -> None:
        task = asyncio.current_task()
        if task is None:
            return
        while task.cancelling():
            task.uncancel()

    async def finalize_canceled_turn(
        self,
        *,
        turn,
        emit: Callable[..., Awaitable[None]],
    ) -> None:
        seq = await self.next_seq(turn.id)
        await self.repository.finalize_turn(
            turn,
            status=ChatTurnStatusEnum.canceled,
            finished_at=datetime.now(UTC),
            set_head=False,
        )
        await emit(
            EventNameEnum.turn_canceled.value,
            TurnEventPayload(turn=await self.repository.summarize_turn(turn)),
            seq=seq,
        )

    def resolve_execution_actions(self, resource_selection: ResourceSelection) -> list[ExecutionAction]:
        actions: list[ExecutionAction] = []
        for action_def in resource_selection.normalized_actions():
            action = self.action_registry.build(action_def)
            if action is None:
                logger.warning("Ignoring unknown chat action: {}", action_def.kind)
                continue
            actions.append(action)
        return actions

    def build_capability_execution_groups(
        self,
        actions: list[ExecutionAction],
        *,
        selected_capability_ids: list[int],
    ) -> tuple[
        list[ExecutionAction],
        list[tuple[dict[str, Any], list[ExecutionAction]]],
        list[tuple[dict[str, Any], list[ExecutionAction]]],
        list[ExecutionAction],
    ]:
        standalone: list[ExecutionAction] = []
        llm_actions: list[ExecutionAction] = []
        grouped: dict[int, list[ExecutionAction]] = {}
        metadata_map: dict[int, dict[str, Any]] = {}

        for action in actions:
            if action.kind == ChatActionKindEnum.llm_response:
                llm_actions.append(action)
                continue
            capability_id = action.metadata.get("capability_id")
            if isinstance(capability_id, int):
                grouped.setdefault(capability_id, []).append(action)
                metadata_map.setdefault(
                    capability_id,
                    {
                        "capability_id": capability_id,
                        "capability_key": action.metadata.get("capability_key", f"capability_{capability_id}"),
                        "capability_name": action.metadata.get("capability_name", action.name),
                        "capability_kind": action.metadata.get("capability_kind", "unknown"),
                        "capability_order": action.metadata.get("capability_order", 999),
                        "capability_required": bool(action.metadata.get("capability_required")),
                    },
                )
                continue
            standalone.append(action)

        ordered_capability_ids: list[int] = [item for item in selected_capability_ids if item in grouped]
        for capability_id, _metadata in sorted(
            metadata_map.items(),
            key=lambda item: (item[1].get("capability_order", 999), item[0]),
        ):
            if capability_id not in ordered_capability_ids:
                ordered_capability_ids.append(capability_id)

        required_groups: list[tuple[dict[str, Any], list[ExecutionAction]]] = []
        optional_groups: list[tuple[dict[str, Any], list[ExecutionAction]]] = []
        for capability_id in ordered_capability_ids:
            items = sorted(
                grouped.get(capability_id, []),
                key=lambda item: (item.priority, item.kind.value, item.action_id),
            )
            if items:
                target = required_groups if metadata_map[capability_id].get("capability_required") else optional_groups
                target.append((metadata_map[capability_id], items))
        return standalone, required_groups, optional_groups, llm_actions

    def should_skip_for_terminal_output(self, *, terminal_exists: bool, required: bool) -> bool:
        return terminal_exists and not required

    async def execute_capability_group(
        self,
        *,
        capability_metadata: dict[str, Any],
        actions: list[ExecutionAction],
        turn,
        turn_root_step,
        session_context: ChatSessionContext,
        cancel_event: asyncio.Event,
        send_event: EventSender,
        emit: Callable[..., Awaitable[None]],
        seq: int,
        step_sequence: int,
    ) -> int:
        started = perf_counter()
        capability_step = await self.repository.create_step(
            turn_id=turn.id,
            parent_step_id=turn_root_step.id,
            root_step_id=turn_root_step.id,
            name=(
                "capability:"
                f"{capability_metadata.get('capability_key', capability_metadata.get('capability_id', 'unknown'))}"
            ),
            kind=ChatStepKindEnum.system,
            sequence=step_sequence,
            metadata=capability_metadata,
        )
        await emit(
            EventNameEnum.step_created.value,
            StepEventPayload(step=await self.repository.summarize_step(capability_step)),
            seq=seq,
            step_id=capability_step.id,
        )
        seq += 1
        await self.repository.update_step(
            capability_step,
            status=ChatStepStatusEnum.running,
            started_at=datetime.now(UTC),
        )
        await emit(
            EventNameEnum.step_started.value,
            StepEventPayload(step=await self.repository.summarize_step(capability_step)),
            seq=seq,
            step_id=capability_step.id,
        )
        seq += 1
        try:
            for index, action in enumerate(actions, start=1):
                await self.ensure_not_canceled(cancel_event)
                if self.should_skip_for_terminal_output(
                    terminal_exists=session_context.artifacts.terminal_output is not None,
                    required=bool(action.metadata.get("capability_required")),
                ):
                    break
                seq = await self.execution_manager.execute_action(
                    action,
                    turn=turn,
                    root_step=capability_step,
                    session_context=session_context,
                    cancel_event=cancel_event,
                    send_event=send_event,
                    emit=emit,
                    seq=seq,
                    step_sequence=step_sequence + index,
                    ensure_not_canceled=self.ensure_not_canceled,
                )
            await self.repository.update_step(
                capability_step,
                status=ChatStepStatusEnum.completed,
                finished_at=datetime.now(UTC),
                metrics=StepMetricPayload(latency_ms=int((perf_counter() - started) * 1000)).model_dump(mode="json"),
            )
            await emit(
                EventNameEnum.step_completed.value,
                StepEventPayload(step=await self.repository.summarize_step(capability_step)),
                seq=seq,
                step_id=capability_step.id,
            )
            return seq + 1
        except Exception:
            await self.repository.update_step(
                capability_step,
                status=ChatStepStatusEnum.failed,
                finished_at=datetime.now(UTC),
                metrics=StepMetricPayload(latency_ms=int((perf_counter() - started) * 1000)).model_dump(mode="json"),
            )
            await emit(
                EventNameEnum.step_failed.value,
                StepEventPayload(step=await self.repository.summarize_step(capability_step)),
                seq=seq,
                step_id=capability_step.id,
            )
            raise

    async def execute_turn(
        self,
        *,
        ws_session_id: int | None,
        ws_public_session_id: str | None,
        conversation,
        turn_request: TurnStartRequest,
        account_id: int | None,
        is_staff: bool,
        send_event: EventSender,
    ) -> int:
        conversation_id = turn_request.conversation_id
        if conversation_id is None:
            raise ValueError("conversation_id is required before starting a turn")

        resolved_actions = self.resolve_execution_actions(turn_request.resource_selection)
        session_context = ChatSessionContext(
            account_id=account_id,
            is_staff=is_staff,
            ws_session_id=ws_session_id,
            ws_public_session_id=ws_public_session_id,
            conversation=conversation,
            turn_request=turn_request,
            resolved_selection=turn_request.resource_selection,
            resolved_actions=resolved_actions,
        )

        turn = await self.repository.create_turn(
            conversation_id=conversation_id,
            request_id=turn_request.request_id,
            trigger=ChatTurnTriggerEnum.user,
            resource_selection=turn_request.resource_selection,
            metadata=turn_request.metadata,
        )

        async def emit(
            event_name: str,
            payload: Any,
            *,
            seq: int,
            step_id: int | None = None,
            data_id: int | None = None,
        ) -> None:
            event = ChatEvent[Any](
                id=f"evt_{turn.id}_{seq}",
                session_id=ws_public_session_id,
                conversation_id=conversation_id,
                turn_id=turn.id,
                seq=seq,
                event=event_name,
                ts=datetime.now(UTC),
                payload=payload,
            )
            await self.repository.create_event_log(
                conversation_id=conversation_id,
                turn_id=turn.id,
                ws_session_id=ws_session_id,
                seq=seq,
                event=event_name,
                payload=event.model_dump(mode="json"),
                step_id=step_id,
                data_id=data_id,
            )
            try:
                await send_event(event)
            except Exception:
                logger.warning(
                    "Failed to push event to websocket, turn_id={}, event={}, seq={}",
                    turn.id,
                    event_name,
                    seq,
                )

        async def run(cancel_event: asyncio.Event) -> None:
            seq = 1
            root_step = await self.repository.create_step(
                turn_id=turn.id,
                name="turn_root",
                kind=ChatStepKindEnum.system,
                sequence=0,
            )
            await self.repository.update_step(root_step, root_step_id=root_step.id)
            await self.repository.update_turn(
                turn,
                status=ChatTurnStatusEnum.accepted,
                root_step_id=root_step.id,
                started_at=datetime.now(UTC),
            )

            input_data = await self.repository.create_data(
                turn_id=turn.id,
                step_id=root_step.id,
                kind=ChatDataKindEnum.input,
                payload_type=ChatPayloadTypeEnum.message_bundle,
                payload=turn_request.input.model_dump(mode="json"),
                role=turn_request.input.role,
                is_final=True,
            )
            await self.repository.update_turn(turn, input_root_data_id=input_data.id)
            await self.repository.update_step(root_step, input_data_ids=[input_data.id])

            await emit(
                EventNameEnum.turn_accepted.value,
                TurnEventPayload(turn=await self.repository.summarize_turn(turn)),
                seq=seq,
            )
            seq += 1
            await self.repository.update_turn(turn, status=ChatTurnStatusEnum.running)
            await emit(
                EventNameEnum.turn_started.value,
                TurnEventPayload(turn=await self.repository.summarize_turn(turn)),
                seq=seq,
            )
            seq += 1

            try:
                await self.ensure_not_canceled(cancel_event)

                capability_plan = (
                    turn_request.metadata.get("capability_plan") if isinstance(turn_request.metadata, dict) else None
                )
                if isinstance(capability_plan, dict):
                    planning_step = await self.repository.create_step(
                        turn_id=turn.id,
                        parent_step_id=root_step.id,
                        root_step_id=root_step.id,
                        name="capability_routing",
                        kind=ChatStepKindEnum.system,
                        sequence=5,
                        metadata={"source": "capability_router"},
                    )
                    await emit(
                        EventNameEnum.step_created.value,
                        StepEventPayload(step=await self.repository.summarize_step(planning_step)),
                        seq=seq,
                        step_id=planning_step.id,
                    )
                    seq += 1
                    await self.repository.update_step(
                        planning_step,
                        status=ChatStepStatusEnum.running,
                        started_at=datetime.now(UTC),
                    )
                    await emit(
                        EventNameEnum.step_started.value,
                        StepEventPayload(step=await self.repository.summarize_step(planning_step)),
                        seq=seq,
                        step_id=planning_step.id,
                    )
                    seq += 1

                    plan_payload = CapabilityPlanPayload(
                        mode=str(capability_plan.get("mode", "heuristic")),
                        summary=str(capability_plan.get("summary", "")),
                        selected_capability_ids=[
                            int(item) for item in capability_plan.get("selected_capability_ids", [])
                        ],
                        selected_capability_keys=[
                            candidate.get("capability_key", "")
                            for candidate in capability_plan.get("candidates", [])
                            if candidate.get("selected")
                        ],
                        candidates=[
                            CapabilityPlanCandidate.model_validate(
                                {
                                    "capability_id": candidate.get("capability_id"),
                                    "capability_key": candidate.get("capability_key"),
                                    "capability_kind": candidate.get("capability_kind"),
                                    "score": candidate.get("score", 0.0),
                                    "selected": candidate.get("selected", False),
                                    "reasons": candidate.get("reasons", []),
                                },
                            )
                            for candidate in capability_plan.get("candidates", [])
                        ],
                    )
                    plan_data = await self.repository.create_data(
                        turn_id=turn.id,
                        step_id=planning_step.id,
                        kind=ChatDataKindEnum.control,
                        payload_type=ChatPayloadTypeEnum.capability_plan,
                        payload=plan_payload.model_dump(mode="json"),
                        is_final=True,
                        metadata={"selected_capability_ids": plan_payload.selected_capability_ids},
                    )
                    await self.repository.update_step(
                        planning_step,
                        status=ChatStepStatusEnum.completed,
                        finished_at=datetime.now(UTC),
                        output_data_ids=[plan_data.id],
                        metrics=StepMetricPayload(latency_ms=0).model_dump(mode="json"),
                    )
                    await emit(
                        EventNameEnum.data_created.value,
                        DataEventPayload[CapabilityPlanPayload](
                            data=ChatDataSchema(
                                id=plan_data.id,
                                turn_id=turn.id,
                                step_id=planning_step.id,
                                kind=str(plan_data.kind),
                                payload_type=self.execution_manager.normalize_payload_type(plan_data.payload_type),
                                role=plan_data.role,
                                is_final=plan_data.is_final,
                                is_visible=plan_data.is_visible,
                                payload=plan_payload,
                                refs=[],
                                metadata=plan_data.metadata or {},
                            ),
                        ),
                        seq=seq,
                        step_id=planning_step.id,
                        data_id=plan_data.id,
                    )
                    seq += 1
                    await emit(
                        EventNameEnum.step_completed.value,
                        StepEventPayload(step=await self.repository.summarize_step(planning_step)),
                        seq=seq,
                        step_id=planning_step.id,
                    )
                    seq += 1

                selected_capability_ids = (
                    [int(item) for item in capability_plan.get("selected_capability_ids", [])]
                    if isinstance(capability_plan, dict)
                    else []
                )
                standalone_actions, required_capability_groups, capability_groups, llm_actions = (
                    self.build_capability_execution_groups(
                        session_context.resolved_actions,
                        selected_capability_ids=selected_capability_ids,
                    )
                )
                step_sequence = 10
                for capability_metadata, capability_actions in required_capability_groups:
                    await self.ensure_not_canceled(cancel_event)
                    seq = await self.execute_capability_group(
                        capability_metadata=capability_metadata,
                        actions=capability_actions,
                        turn=turn,
                        turn_root_step=root_step,
                        session_context=session_context,
                        cancel_event=cancel_event,
                        send_event=send_event,
                        emit=emit,
                        seq=seq,
                        step_sequence=step_sequence,
                    )
                    step_sequence += 10
                for action in standalone_actions:
                    await self.ensure_not_canceled(cancel_event)
                    if self.should_skip_for_terminal_output(
                        terminal_exists=session_context.artifacts.terminal_output is not None,
                        required=bool(action.metadata.get("required")),
                    ):
                        break
                    seq = await self.execution_manager.execute_action(
                        action,
                        turn=turn,
                        root_step=root_step,
                        session_context=session_context,
                        cancel_event=cancel_event,
                        send_event=send_event,
                        emit=emit,
                        seq=seq,
                        step_sequence=step_sequence,
                        ensure_not_canceled=self.ensure_not_canceled,
                    )
                    step_sequence += 10
                for capability_metadata, capability_actions in capability_groups:
                    await self.ensure_not_canceled(cancel_event)
                    if self.should_skip_for_terminal_output(
                        terminal_exists=session_context.artifacts.terminal_output is not None,
                        required=bool(capability_metadata.get("capability_required")),
                    ):
                        break
                    seq = await self.execute_capability_group(
                        capability_metadata=capability_metadata,
                        actions=capability_actions,
                        turn=turn,
                        turn_root_step=root_step,
                        session_context=session_context,
                        cancel_event=cancel_event,
                        send_event=send_event,
                        emit=emit,
                        seq=seq,
                        step_sequence=step_sequence,
                    )
                    step_sequence += 10
                for action in llm_actions:
                    await self.ensure_not_canceled(cancel_event)
                    seq = await self.execution_manager.execute_action(
                        action,
                        turn=turn,
                        root_step=root_step,
                        session_context=session_context,
                        cancel_event=cancel_event,
                        send_event=send_event,
                        emit=emit,
                        seq=seq,
                        step_sequence=step_sequence,
                        ensure_not_canceled=self.ensure_not_canceled,
                    )
                    step_sequence += 10
                session_context.set_state("active_action_id", None)

                if (
                    session_context.artifacts.output_payload is None
                    and session_context.artifacts.terminal_output is None
                ):
                    raise ValueError("No execution action produced final chat output")

                await emit(
                    EventNameEnum.turn_completed.value,
                    TurnEventPayload(turn=await self.repository.summarize_turn(turn)),
                    seq=seq,
                )
            except TurnCancelledError:
                await self.finalize_canceled_turn(turn=turn, emit=emit)
            except asyncio.CancelledError:
                self.clear_current_task_cancellation()
                await self.finalize_canceled_turn(turn=turn, emit=emit)
            except Exception as exc:
                logger.exception("Chat turn failed: turn_id={}", turn.id)
                await self.repository.finalize_turn(
                    turn,
                    status=ChatTurnStatusEnum.failed,
                    error_message=str(exc),
                    finished_at=datetime.now(UTC),
                    set_head=False,
                )
                seq = await self.next_seq(turn.id)
                await emit(
                    EventNameEnum.turn_failed.value,
                    TurnEventPayload(turn=await self.repository.summarize_turn(turn)),
                    seq=seq,
                )

        self.start_turn_task(turn.id, run)
        return turn.id

    async def next_seq(self, turn_id: int) -> int:
        return await self.repository.get_last_event_seq(turn_id) + 1
