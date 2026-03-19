from __future__ import annotations

from typing import Any, cast
from datetime import UTC, datetime

from tortoise.transactions import in_transaction

from core.types import ApiException
from ext.ext_tortoise.enums import (
    ChatDataKindEnum,
    ChatStepStatusEnum,
    ChatTurnStatusEnum,
    ChatTurnTriggerEnum,
    ChatConversationStatusEnum,
)
from service.chat.domain.schema import (
    StepSummary,
    TurnSummary,
    UsagePayload,
    ResourceSelection,
    StepMetricPayload,
    ChatPayloadTypeEnum,
    ConversationSummary,
    MessageBundlePayload,
    parse_resource_selection,
)
from ext.ext_tortoise.models.knowledge_base import (
    ChatData,
    ChatStep,
    ChatTurn,
    ChatEventLog,
    ChatConversation,
    ChatTurnCheckpoint,
    ChatWebSocketSession,
)

ACTIVE_TURN_STATUSES = {
    ChatTurnStatusEnum.pending,
    ChatTurnStatusEnum.accepted,
    ChatTurnStatusEnum.running,
    ChatTurnStatusEnum.streaming,
}


class ChatRepository:
    async def create_conversation(
        self,
        *,
        title: str,
        user_id: int | None,
        resource_selection: ResourceSelection,
        metadata: dict[str, Any] | None = None,
    ) -> ChatConversation:
        return await ChatConversation.create(
            title=title,
            user_id=user_id,
            default_resource_config=resource_selection.model_dump(mode="json"),
            metadata=metadata or {},
            status=ChatConversationStatusEnum.active,
        )

    async def get_conversation(self, conversation_id: int) -> ChatConversation | None:
        return await ChatConversation.get_or_none(id=conversation_id, deleted_at=0)

    async def get_accessible_conversation(
        self,
        conversation_id: int,
        *,
        account_id: int | None,
        is_staff: bool,
    ) -> ChatConversation | None:
        queryset = ChatConversation.filter(id=conversation_id, deleted_at=0)
        if not is_staff:
            if account_id is None:
                return None
            queryset = queryset.filter(user_id=account_id)
        return await queryset.first()

    async def list_conversations(
        self,
        *,
        account_id: int | None,
        is_staff: bool,
        limit: int = 50,
    ) -> list[ChatConversation]:
        queryset = ChatConversation.filter(deleted_at=0)
        if not is_staff:
            if account_id is None:
                return []
            queryset = queryset.filter(user_id=account_id)
        return await queryset.order_by("-updated_at", "-id").limit(limit)

    async def create_ws_session(
        self,
        *,
        session_id: str,
        account_id: int | None,
        conversation_id: int | None,
        client_info: dict[str, Any],
    ) -> ChatWebSocketSession:
        return await ChatWebSocketSession.create(
            session_id=session_id,
            account_id=account_id,
            conversation_id=conversation_id,
            client_info=client_info,
        )

    async def touch_ws_session(
        self,
        session_id: str,
        *,
        conversation_id: int | None = None,
        status: str | None = None,
        last_ack_seq: int | None = None,
    ) -> ChatWebSocketSession | None:
        session = await ChatWebSocketSession.get_or_none(session_id=session_id, deleted_at=0)
        if not session:
            return None
        if conversation_id is not None:
            session.conversation_id = conversation_id
        if status is not None:
            session.status = status
        if last_ack_seq is not None:
            session.last_ack_seq = last_ack_seq
        await session.save()
        return session

    async def get_ws_session(self, session_id: str) -> ChatWebSocketSession | None:
        return await ChatWebSocketSession.get_or_none(session_id=session_id, deleted_at=0)

    async def get_ws_session_by_id(self, ws_session_id: int) -> ChatWebSocketSession | None:
        return await ChatWebSocketSession.get_or_none(id=ws_session_id, deleted_at=0)

    async def get_turn(self, turn_id: int) -> ChatTurn | None:
        return await ChatTurn.get_or_none(id=turn_id, deleted_at=0)

    async def list_turns(self, conversation_id: int) -> list[ChatTurn]:
        return await ChatTurn.filter(conversation_id=conversation_id, deleted_at=0).order_by("seq")

    async def get_data_map(self, data_ids: list[int]) -> dict[int, ChatData]:
        if not data_ids:
            return {}
        items = await ChatData.filter(id__in=data_ids, deleted_at=0)
        return {item.id: item for item in items}

    async def list_event_logs_by_turn_ids(self, turn_ids: list[int]) -> dict[int, list[ChatEventLog]]:
        if not turn_ids:
            return {}
        items = await ChatEventLog.filter(turn_id__in=turn_ids, deleted_at=0).order_by("turn_id", "seq")
        grouped: dict[int, list[ChatEventLog]] = {}
        for item in items:
            grouped.setdefault(cast(int, item.turn_id), []).append(item)
        return grouped

    async def create_turn(
        self,
        *,
        conversation_id: int,
        request_id: str | None,
        trigger: ChatTurnTriggerEnum,
        resource_selection: ResourceSelection,
        metadata: dict[str, Any] | None = None,
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

            if conversation.active_turn_id is not None:
                active_turn = await ChatTurn.filter(id=conversation.active_turn_id, deleted_at=0).using_db(conn).first()
                if active_turn is not None and active_turn.status in ACTIVE_TURN_STATUSES:
                    raise ApiException("当前会话已有进行中的 turn")

            latest_turn = (
                await ChatTurn.filter(conversation_id=conversation_id, deleted_at=0)
                .using_db(conn)
                .order_by("-seq")
                .first()
            )
            seq = (latest_turn.seq if latest_turn else 0) + 1
            turn = await ChatTurn.create(
                conversation_id=conversation_id,
                seq=seq,
                request_id=request_id,
                trigger=trigger,
                resource_selection=resource_selection.model_dump(mode="json"),
                metadata=metadata or {},
                using_db=conn,
            )
            conversation.active_turn_id = turn.id
            await conversation.save(using_db=conn, update_fields=["active_turn_id"])
        return turn

    async def update_turn(
        self,
        turn: ChatTurn,
        *,
        status: ChatTurnStatusEnum | None = None,
        input_root_data_id: int | None = None,
        output_root_data_id: int | None = None,
        root_step_id: int | None = None,
        error_message: str | None = None,
        usage: UsagePayload | dict[str, Any] | None = None,
        cancel_requested_at: datetime | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
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
        if root_step_id is not None:
            turn.root_step_id = root_step_id
            update_fields.append("root_step_id")
        if error_message is not None:
            turn.error_message = error_message
            update_fields.append("error_message")
        if usage is not None:
            turn.usage = usage.model_dump(mode="json") if isinstance(usage, UsagePayload) else usage
            update_fields.append("usage")
        if cancel_requested_at is not None:
            turn.cancel_requested_at = cancel_requested_at
            update_fields.append("cancel_requested_at")
        if started_at is not None:
            turn.started_at = started_at
            update_fields.append("started_at")
        if finished_at is not None:
            turn.finished_at = finished_at
            update_fields.append("finished_at")
        if update_fields:
            await turn.save(update_fields=update_fields)
        return turn

    async def create_step(
        self,
        *,
        turn_id: int,
        name: str,
        kind: str,
        sequence: int,
        parent_step_id: int | None = None,
        root_step_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatStep:
        return await ChatStep.create(
            turn_id=turn_id,
            name=name,
            kind=kind,
            sequence=sequence,
            parent_step_id=parent_step_id,
            root_step_id=root_step_id,
            metadata=metadata or {},
        )

    async def update_step(
        self,
        step: ChatStep,
        *,
        status: ChatStepStatusEnum | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        metrics: dict[str, Any] | None = None,
        error_message: str | None = None,
        input_data_ids: list[int] | None = None,
        output_data_ids: list[int] | None = None,
        root_step_id: int | None = None,
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
        if metrics is not None:
            step.metrics = metrics
            update_fields.append("metrics")
        if error_message is not None:
            step.error_message = error_message
            update_fields.append("error_message")
        if input_data_ids is not None:
            step.input_data_ids = input_data_ids
            update_fields.append("input_data_ids")
        if output_data_ids is not None:
            step.output_data_ids = output_data_ids
            update_fields.append("output_data_ids")
        if root_step_id is not None:
            step.root_step_id = root_step_id
            update_fields.append("root_step_id")
        if update_fields:
            await step.save(update_fields=update_fields)
        return step

    async def create_data(
        self,
        *,
        turn_id: int,
        step_id: int | None,
        kind: ChatDataKindEnum,
        payload_type: ChatPayloadTypeEnum,
        payload: dict[str, Any],
        role: str | None = None,
        is_final: bool = False,
        is_visible: bool = True,
        refs: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatData:
        return await ChatData.create(
            turn_id=turn_id,
            step_id=step_id,
            kind=kind,
            payload_type=payload_type,
            payload=payload,
            role=role,
            is_final=is_final,
            is_visible=is_visible,
            refs=refs or [],
            metadata=metadata or {},
        )

    async def create_event_log(
        self,
        *,
        conversation_id: int,
        turn_id: int,
        ws_session_id: int | None,
        seq: int,
        event: str,
        payload: dict[str, Any],
        step_id: int | None = None,
        data_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatEventLog:
        return await ChatEventLog.create(
            conversation_id=conversation_id,
            turn_id=turn_id,
            ws_session_id=ws_session_id,
            seq=seq,
            event=event,
            payload=payload,
            step_id=step_id,
            data_id=data_id,
            metadata=metadata or {},
        )

    async def create_checkpoint(
        self,
        *,
        turn_id: int,
        checkpoint_no: int,
        snapshot: dict[str, Any],
        latest_event_seq: int,
    ):
        return await ChatTurnCheckpoint.create(
            turn_id=turn_id,
            checkpoint_no=checkpoint_no,
            snapshot=snapshot,
            latest_event_seq=latest_event_seq,
        )

    async def list_event_logs(self, turn_id: int, *, min_seq: int = 0) -> list[ChatEventLog]:
        return await ChatEventLog.filter(turn_id=turn_id, seq__gt=min_seq, deleted_at=0).order_by("seq")

    async def get_last_event_seq(self, turn_id: int) -> int:
        latest = await ChatEventLog.filter(turn_id=turn_id, deleted_at=0).order_by("-seq").first()
        return latest.seq if latest else 0

    async def mark_turn_cancel_requested(self, turn_id: int) -> ChatTurn | None:
        turn = await ChatTurn.get_or_none(id=turn_id, deleted_at=0)
        if not turn:
            return None
        turn.cancel_requested_at = datetime.now(UTC)
        await turn.save(update_fields=["cancel_requested_at"])
        return turn

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

    async def finalize_turn(
        self,
        turn: ChatTurn,
        *,
        status: ChatTurnStatusEnum,
        finished_at: datetime,
        output_root_data_id: int | None = None,
        error_message: str | None = None,
        usage: UsagePayload | dict[str, Any] | None = None,
        set_head: bool,
    ) -> ChatTurn:
        turn_record = cast(Any, turn)
        conversation_id = cast(int, turn_record.conversation_id)
        async with in_transaction(connection_name=ChatConversation._meta.app) as conn:
            locked_turn = await ChatTurn.filter(id=turn.id, deleted_at=0).using_db(conn).select_for_update().first()
            if locked_turn is None:
                raise ValueError(f"Turn not found: {turn.id}")
            locked_conversation = (
                await ChatConversation.filter(id=conversation_id, deleted_at=0)
                .using_db(conn)
                .select_for_update()
                .first()
            )
            if locked_conversation is None:
                raise ValueError(f"Conversation not found: {conversation_id}")

            turn_update_fields = ["status", "finished_at"]
            locked_turn.status = status
            locked_turn.finished_at = finished_at
            if output_root_data_id is not None:
                locked_turn.output_root_data_id = output_root_data_id
                turn_update_fields.append("output_root_data_id")
            if error_message is not None:
                locked_turn.error_message = error_message
                turn_update_fields.append("error_message")
            if usage is not None:
                locked_turn.usage = usage.model_dump(mode="json") if isinstance(usage, UsagePayload) else usage
                turn_update_fields.append("usage")
            await locked_turn.save(using_db=conn, update_fields=turn_update_fields)

            conversation_update_fields: list[str] = []
            if locked_conversation.active_turn_id == locked_turn.id:
                cast(Any, locked_conversation).active_turn_id = None
                conversation_update_fields.append("active_turn_id")
            if set_head:
                locked_conversation.head_turn_id = locked_turn.id
                conversation_update_fields.append("head_turn_id")
            if conversation_update_fields:
                await locked_conversation.save(using_db=conn, update_fields=conversation_update_fields)

        turn.status = locked_turn.status
        turn.finished_at = locked_turn.finished_at
        turn.output_root_data_id = locked_turn.output_root_data_id
        turn.error_message = locked_turn.error_message
        turn.usage = locked_turn.usage
        return turn

    async def build_history(
        self,
        conversation_id: int,
        *,
        limit: int = 10,
    ) -> list[tuple[MessageBundlePayload, MessageBundlePayload]]:
        turns = (
            await ChatTurn.filter(
                conversation_id=conversation_id,
                deleted_at=0,
                status=ChatTurnStatusEnum.completed,
            )
            .order_by("-seq")
            .limit(limit)
        )
        history: list[tuple[MessageBundlePayload, MessageBundlePayload]] = []
        for turn in reversed(turns):
            if not turn.input_root_data_id or not turn.output_root_data_id:
                continue
            input_data = await ChatData.get_or_none(id=turn.input_root_data_id, deleted_at=0)
            output_data = await ChatData.get_or_none(id=turn.output_root_data_id, deleted_at=0)
            if not input_data or not output_data:
                continue
            history.append(
                (
                    MessageBundlePayload.model_validate(input_data.payload),
                    MessageBundlePayload.model_validate(output_data.payload),
                ),
            )
        return history

    async def summarize_conversation(self, conversation: ChatConversation) -> ConversationSummary:
        return ConversationSummary(
            id=conversation.id,
            title=conversation.title,
            status=str(conversation.status),
            user_id=conversation.user_id,
            active_turn_id=conversation.active_turn_id,
            head_turn_id=conversation.head_turn_id,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            default_resource_selection=parse_resource_selection(conversation.default_resource_config),
            metadata=conversation.metadata or {},
        )

    async def summarize_turn(self, turn: ChatTurn) -> TurnSummary:
        turn_record = cast(Any, turn)
        return TurnSummary(
            id=turn.id,
            conversation_id=cast(int, turn_record.conversation_id),
            seq=turn.seq,
            status=str(turn.status),
            trigger=str(turn.trigger),
            input_root_data_id=turn.input_root_data_id,
            output_root_data_id=turn.output_root_data_id,
            root_step_id=turn.root_step_id,
            started_at=turn.started_at,
            finished_at=turn.finished_at,
            created_at=turn.created_at,
            updated_at=turn.updated_at,
            error_message=turn.error_message,
            usage=UsagePayload.model_validate(turn.usage or {}),
            resource_selection=parse_resource_selection(turn.resource_selection),
        )

    async def summarize_step(self, step: ChatStep) -> StepSummary:
        step_record = cast(Any, step)
        return StepSummary(
            id=step.id,
            turn_id=cast(int, step_record.turn_id),
            parent_step_id=cast(int | None, step_record.parent_step_id),
            root_step_id=step.root_step_id,
            kind=str(step.kind),
            name=step.name,
            status=str(step.status),
            sequence=step.sequence,
            attempt=step.attempt,
            started_at=step.started_at,
            finished_at=step.finished_at,
            metrics=StepMetricPayload.model_validate(step.metrics or {}),
            error_message=step.error_message,
            input_data_ids=[int(item) for item in (step.input_data_ids or [])],
            output_data_ids=[int(item) for item in (step.output_data_ids or [])],
            metadata=step.metadata or {},
        )
