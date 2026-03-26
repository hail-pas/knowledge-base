from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

from tortoise.transactions import in_transaction

from core.types import ApiException
from ext.ext_tortoise.enums import (
    ChatDataKindEnum,
    ChatStepStatusEnum,
    ChatTurnStatusEnum,
    ChatTurnTriggerEnum,
)
from ext.ext_tortoise.models.knowledge_base import (
    ChatConversation,
    ChatData,
    ChatStep,
    ChatTurn,
)
from service.chat.domain.schema import (
    ChatDataSchema,
    ChatHistoryItem,
    ChatRequestContext,
    ConversationSummary,
    ErrorPayload,
    MessageBundlePayload,
    MCPResultPayload,
    PersistedPayload,
    ResourceSelection,
    StepSummary,
    SubAgentResultPayload,
    ToolResultPayload,
    TurnSummary,
    UsagePayload,
    parse_resource_selection,
    validate_payload_for_type,
)


ACTIVE_TURN_STATUSES = {
    ChatTurnStatusEnum.running,
}


class ChatRepository:
    async def create_conversation(
        self,
        *,
        agent_key: str = "orchestrator.default",
        title: str,
        user_id: int | None,
        resource_selection: ResourceSelection,
    ) -> ChatConversation:
        return await ChatConversation.create(
            agent_key=agent_key,
            title=title,
            user_id=user_id,
            default_resource_config=resource_selection.model_dump(mode="json"),
        )

    async def get_conversation(self, conversation_id: int) -> ChatConversation | None:
        return await ChatConversation.get_or_none(id=conversation_id, deleted_at=0)

    async def get_accessible_conversation(
        self,
        conversation_id: int,
        *,
        context: ChatRequestContext,
    ) -> ChatConversation | None:
        queryset = ChatConversation.filter(id=conversation_id, deleted_at=0)
        if not context.is_staff:
            if context.account_id is None:
                return None
            queryset = queryset.filter(user_id=context.account_id)
        return await queryset.first()

    async def list_conversations(
        self,
        *,
        context: ChatRequestContext,
        limit: int = 50,
    ) -> list[ChatConversation]:
        queryset = ChatConversation.filter(deleted_at=0)
        if not context.is_staff:
            if context.account_id is None:
                return []
            queryset = queryset.filter(user_id=context.account_id)
        return await queryset.order_by("-updated_at", "-id").limit(limit)

    async def update_conversation_default_resource_selection(
        self,
        conversation_id: int,
        resource_selection: ResourceSelection,
    ) -> ChatConversation | None:
        conversation = await self.get_conversation(conversation_id)
        if conversation is None:
            return None
        conversation.default_resource_config = resource_selection.model_dump(mode="json")
        await conversation.save(update_fields=["default_resource_config"])
        return conversation

    async def get_turn(self, turn_id: int) -> ChatTurn | None:
        return await ChatTurn.get_or_none(id=turn_id, deleted_at=0)

    async def get_accessible_turn(
        self,
        turn_id: int,
        *,
        context: ChatRequestContext,
    ) -> ChatTurn | None:
        queryset = ChatTurn.filter(id=turn_id, deleted_at=0).prefetch_related("conversation")
        turn = await queryset.first()
        if turn is None:
            return None
        if context.is_staff:
            return turn
        conversation = cast(ChatConversation | None, getattr(turn, "conversation", None))
        if conversation is None or conversation.user_id != context.account_id:
            return None
        return turn

    async def list_turns(self, conversation_id: int) -> list[ChatTurn]:
        return await ChatTurn.filter(conversation_id=conversation_id, deleted_at=0).order_by("seq")

    async def get_latest_turn(self, conversation_id: int) -> ChatTurn | None:
        return (
            await ChatTurn.filter(conversation_id=conversation_id, deleted_at=0)
            .order_by("-seq")
            .first()
        )

    async def list_latest_turns_by_conversation_ids(self, conversation_ids: list[int]) -> dict[int, ChatTurn]:
        if not conversation_ids:
            return {}
        turns = (
            await ChatTurn.filter(conversation_id__in=conversation_ids, deleted_at=0)
            .order_by("conversation_id", "-seq", "-id")
        )
        latest_by_conversation_id: dict[int, ChatTurn] = {}
        for turn in turns:
            conversation_id = cast(int, turn.conversation_id)
            latest_by_conversation_id.setdefault(conversation_id, turn)
        return latest_by_conversation_id

    async def create_turn(
        self,
        *,
        conversation_id: int,
        agent_key: str = "orchestrator.default",
        request_id: UUID | None,
        trigger: ChatTurnTriggerEnum,
        resource_selection: ResourceSelection,
        planner_mode: str | None = None,
        planner_summary: str | None = None,
        execution_plan: dict[str, Any] | None = None,
    ) -> ChatTurn:
        async with in_transaction(connection_name=ChatConversation._meta.app) as conn:
            conversation = (
                await ChatConversation.filter(id=conversation_id, deleted_at=0)
                .using_db(conn)
                .select_for_update()
                .first()
            )
            if conversation is None:
                raise ValueError(f"Conversation not found: {conversation_id}")

            active_turn = (
                await ChatTurn.filter(
                    conversation_id=conversation_id,
                    deleted_at=0,
                    status__in=[status.value for status in ACTIVE_TURN_STATUSES],
                )
                .using_db(conn)
                .order_by("-id")
                .first()
            )
            if active_turn is not None:
                raise ApiException("当前会话已有进行中的 turn")

            latest_turn = (
                await ChatTurn.filter(conversation_id=conversation_id, deleted_at=0)
                .using_db(conn)
                .order_by("-seq")
                .first()
            )
            seq = (latest_turn.seq if latest_turn else 0) + 1
            return await ChatTurn.create(
                conversation_id=conversation_id,
                agent_key=agent_key,
                seq=seq,
                status=ChatTurnStatusEnum.running,
                request_id=request_id,
                trigger=trigger,
                resource_selection=resource_selection.model_dump(mode="json"),
                planner_mode=planner_mode,
                planner_summary=planner_summary,
                execution_plan=execution_plan or {},
                started_at=datetime.now(UTC),
                using_db=conn,
            )

    async def update_turn(
        self,
        turn: ChatTurn,
        *,
        status: ChatTurnStatusEnum | None = None,
        input_root_data_id: int | None = None,
        output_root_data_id: int | None = None,
        usage: UsagePayload | dict[str, Any] | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        planner_mode: str | None = None,
        planner_summary: str | None = None,
        execution_plan: dict[str, Any] | None = None,
    ) -> ChatTurn:
        update_fields: list[str] = []
        if status is not None:
            turn.status = status
            update_fields.append("status")
        if input_root_data_id is not None:
            turn.input_root_data_id = input_root_data_id
            update_fields.append("input_root_data_id")
        if output_root_data_id is not None:
            turn.output_root_data_id = output_root_data_id
            update_fields.append("output_root_data_id")
        if usage is not None:
            turn.usage = usage.model_dump(mode="json") if isinstance(usage, UsagePayload) else usage
            update_fields.append("usage")
        if planner_mode is not None:
            turn.planner_mode = planner_mode
            update_fields.append("planner_mode")
        if planner_summary is not None:
            turn.planner_summary = planner_summary
            update_fields.append("planner_summary")
        if execution_plan is not None:
            turn.execution_plan = execution_plan
            update_fields.append("execution_plan")
        if started_at is not None:
            turn.started_at = started_at
            update_fields.append("started_at")
        if finished_at is not None:
            turn.finished_at = finished_at
            update_fields.append("finished_at")
        if update_fields:
            await turn.save(update_fields=update_fields)
        return turn

    async def finalize_turn(
        self,
        turn: ChatTurn,
        *,
        status: ChatTurnStatusEnum,
        finished_at: datetime,
        output_root_data_id: int | None = None,
        usage: UsagePayload | dict[str, Any] | None = None,
    ) -> ChatTurn:
        return await self.update_turn(
            turn,
            status=status,
            finished_at=finished_at,
            output_root_data_id=output_root_data_id,
            usage=usage,
        )

    async def create_step(
        self,
        *,
        conversation_id: int,
        turn_id: int,
        name: str,
        kind: str,
        sequence: int,
        parent_step_id: int | None = None,
        capability_key: str | None = None,
        operation_key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatStep:
        return await ChatStep.create(
            conversation_id=conversation_id,
            turn_id=turn_id,
            name=name,
            kind=kind,
            capability_key=capability_key,
            operation_key=operation_key,
            sequence=sequence,
            parent_step_id=parent_step_id,
            metadata=metadata or {},
            status=ChatStepStatusEnum.running,
            started_at=datetime.now(UTC),
        )

    async def update_step(
        self,
        step: ChatStep,
        *,
        status: ChatStepStatusEnum | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
    ) -> ChatStep:
        update_fields: list[str] = []
        if status is not None:
            step.status = status
            update_fields.append("status")
        if started_at is not None:
            step.started_at = started_at
            update_fields.append("started_at")
        if finished_at is not None:
            step.finished_at = finished_at
            update_fields.append("finished_at")
        if update_fields:
            await step.save(update_fields=update_fields)
        return step

    async def bulk_update_steps_status(
        self,
        *,
        step_ids: list[int],
        status: ChatStepStatusEnum,
        finished_at: datetime,
    ) -> int:
        if not step_ids:
            return 0
        return await ChatStep.filter(
            id__in=step_ids,
            deleted_at=0,
            status=ChatStepStatusEnum.running,
        ).update(
            status=status,
            finished_at=finished_at,
        )

    async def list_steps(self, turn_id: int) -> list[ChatStep]:
        return await ChatStep.filter(turn_id=turn_id, deleted_at=0).order_by("sequence", "id")

    async def list_steps_by_turn_ids(self, turn_ids: list[int]) -> dict[int, list[ChatStep]]:
        if not turn_ids:
            return {}
        steps = await ChatStep.filter(turn_id__in=turn_ids, deleted_at=0).order_by("turn_id", "sequence", "id")
        grouped: dict[int, list[ChatStep]] = {}
        for step in steps:
            grouped.setdefault(cast(int, step.turn_id), []).append(step)
        return grouped

    async def create_data(
        self,
        *,
        conversation_id: int,
        turn_id: int,
        step_id: int,
        kind: ChatDataKindEnum,
        payload_type,
        payload: PersistedPayload | dict[str, Any],
    ) -> ChatData:
        validated_payload = validate_payload_for_type(payload_type, payload)
        return await ChatData.create(
            conversation_id=conversation_id,
            turn_id=turn_id,
            step_id=step_id,
            kind=kind,
            payload_type=str(getattr(payload_type, "value", payload_type)),
            payload=validated_payload.model_dump(mode="json"),
        )

    async def get_data(self, data_id: int) -> ChatData | None:
        return await ChatData.get_or_none(id=data_id, deleted_at=0)

    async def get_data_map(self, data_ids: list[int]) -> dict[int, ChatData]:
        if not data_ids:
            return {}
        rows = await ChatData.filter(id__in=data_ids, deleted_at=0)
        return {row.id: row for row in rows}

    async def list_data_by_turn_ids(self, turn_ids: list[int]) -> dict[int, list[ChatData]]:
        if not turn_ids:
            return {}
        rows = await ChatData.filter(turn_id__in=turn_ids, deleted_at=0).order_by("turn_id", "id")
        grouped: dict[int, list[ChatData]] = {}
        for row in rows:
            grouped.setdefault(cast(int, row.turn_id), []).append(row)
        return grouped

    async def build_history(
        self,
        conversation_id: int,
        *,
        limit: int = 10,
    ) -> list[ChatHistoryItem]:
        turns = (
            await ChatTurn.filter(
                conversation_id=conversation_id,
                deleted_at=0,
                status=ChatTurnStatusEnum.completed,
            )
            .order_by("-seq")
            .limit(limit)
        )
        data_map = await self.get_data_map(
            [
                data_id
                for turn in turns
                for data_id in [turn.input_root_data_id, turn.output_root_data_id]
                if data_id is not None
            ],
        )
        history: list[ChatHistoryItem] = []
        for turn in reversed(turns):
            if not turn.input_root_data_id or not turn.output_root_data_id:
                continue
            input_row = data_map.get(turn.input_root_data_id)
            output_row = data_map.get(turn.output_root_data_id)
            if input_row is None or output_row is None:
                continue
            input_payload = MessageBundlePayload.model_validate(input_row.payload)
            output_payload = validate_payload_for_type(output_row.payload_type, output_row.payload)
            assistant_text = self._history_output_text(output_payload)
            if not input_payload.text or not assistant_text:
                continue
            history.append(
                ChatHistoryItem(
                    user_text=input_payload.text,
                    assistant_text=assistant_text,
                ),
            )
        return history

    def _history_output_text(self, payload: PersistedPayload) -> str | None:
        if isinstance(payload, MessageBundlePayload):
            return payload.text or None
        if isinstance(payload, ErrorPayload):
            return payload.message or None
        if isinstance(payload, ToolResultPayload | MCPResultPayload | SubAgentResultPayload):
            return str(payload.content_text or payload.summary or "").strip() or None
        return None

    def _serialize_conversation(self, conversation: ChatConversation) -> ConversationSummary:
        return ConversationSummary(
            id=conversation.id,
            agent_key=conversation.agent_key,
            title=conversation.title,
            user_id=conversation.user_id,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            default_resource_selection=parse_resource_selection(conversation.default_resource_config),
        )

    async def summarize_conversation(self, conversation: ChatConversation) -> ConversationSummary:
        return self._serialize_conversation(conversation)

    async def summarize_conversations(self, conversations: list[ChatConversation]) -> list[ConversationSummary]:
        return [self._serialize_conversation(conversation) for conversation in conversations]

    def _serialize_turn(self, turn: ChatTurn) -> TurnSummary:
        turn_record = cast(Any, turn)
        return TurnSummary(
            id=turn.id,
            conversation_id=cast(int, turn_record.conversation_id),
            agent_key=turn.agent_key,
            seq=turn.seq,
            status=str(turn.status),
            trigger=str(turn.trigger),
            request_id=turn.request_id,
            input_root_data_id=turn.input_root_data_id,
            output_root_data_id=turn.output_root_data_id,
            started_at=turn.started_at,
            finished_at=turn.finished_at,
            created_at=turn.created_at,
            updated_at=turn.updated_at,
            planner_mode=turn.planner_mode,
            planner_summary=turn.planner_summary,
            selected_capability_keys=[
                str(item)
                for item in ((turn.execution_plan or {}).get("selected_capability_keys") or [])
                if str(item).strip()
            ],
            usage=UsagePayload.model_validate(turn.usage or {}),
            resource_selection=parse_resource_selection(turn.resource_selection),
        )

    async def summarize_turn(self, turn: ChatTurn) -> TurnSummary:
        return self._serialize_turn(turn)

    async def summarize_turns(self, turns: list[ChatTurn]) -> list[TurnSummary]:
        return [self._serialize_turn(turn) for turn in turns]

    def _serialize_step(self, step: ChatStep) -> StepSummary:
        step_record = cast(Any, step)
        return StepSummary(
            id=step.id,
            conversation_id=cast(int, step_record.conversation_id),
            turn_id=cast(int, step_record.turn_id),
            parent_step_id=cast(int | None, step_record.parent_step_id),
            kind=str(step.kind),
            capability_key=step.capability_key,
            operation_key=step.operation_key,
            name=step.name,
            status=str(step.status),
            sequence=step.sequence,
            metadata=step.metadata or {},
            started_at=step.started_at,
            finished_at=step.finished_at,
        )

    async def summarize_step(self, step: ChatStep) -> StepSummary:
        return self._serialize_step(step)

    async def summarize_steps(self, steps: list[ChatStep]) -> list[StepSummary]:
        return [self._serialize_step(step) for step in steps]

    def _serialize_data(self, data: ChatData) -> ChatDataSchema[PersistedPayload]:
        data_record = cast(Any, data)
        payload_type = data.payload_type
        return ChatDataSchema(
            id=data.id,
            conversation_id=cast(int, data_record.conversation_id),
            turn_id=cast(int, data_record.turn_id),
            step_id=cast(int, data_record.step_id),
            kind=ChatDataKindEnum(str(data.kind)),
            payload_type=payload_type,
            payload=validate_payload_for_type(payload_type, data.payload),
        )

    async def summarize_data(self, data: ChatData) -> ChatDataSchema[PersistedPayload]:
        return self._serialize_data(data)

    async def summarize_data_rows(self, rows: list[ChatData]) -> list[ChatDataSchema[PersistedPayload]]:
        return [self._serialize_data(row) for row in rows]
