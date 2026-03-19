from __future__ import annotations

from uuid import uuid4
from typing import Any, Callable, Awaitable

from loguru import logger

from core.types import ApiException
from service.chat.domain.schema import (
    ChatEvent,
    TurnStartRequest,
    ResourceSelection,
    TurnReplayRequest,
    TurnStartAccepted,
    ConversationSummary,
    ConversationListItem,
    ConversationTimeline,
    MessageBundlePayload,
    ConversationTurnDetail,
)
from service.chat.runtime.engine import ChatRuntime
from service.chat.store.repository import ChatRepository
from service.chat.capability.service import ChatCapabilityService
from service.chat.execution.registry import (
    ExecutionActionRegistry,
    create_default_action_registry,
)


class ChatApplicationService:
    def __init__(self) -> None:
        self.repository: ChatRepository = ChatRepository()
        self.action_registry: ExecutionActionRegistry = create_default_action_registry()
        self.capability_service: ChatCapabilityService = ChatCapabilityService(
            action_registry=self.action_registry,
        )
        self.runtime: ChatRuntime = ChatRuntime(self.repository, self.action_registry)

    async def list_conversations(
        self,
        *,
        account_id: int | None,
        is_staff: bool,
        limit: int = 50,
    ) -> list[ConversationListItem]:
        conversations = await self.repository.list_conversations(
            account_id=account_id,
            is_staff=is_staff,
            limit=limit,
        )
        items: list[ConversationListItem] = []
        for conversation in conversations:
            latest_user_text: str | None = None
            latest_assistant_text: str | None = None
            if conversation.head_turn_id:
                head_turn = await self.repository.get_turn(conversation.head_turn_id)
                if head_turn is not None:
                    data_map = await self.repository.get_data_map(
                        [
                            data_id
                            for data_id in [head_turn.input_root_data_id, head_turn.output_root_data_id]
                            if data_id is not None
                        ],
                    )
                    input_data = (
                        MessageBundlePayload.model_validate(data_map[head_turn.input_root_data_id].payload)
                        if head_turn.input_root_data_id in data_map
                        else None
                    )
                    output_data = (
                        MessageBundlePayload.model_validate(data_map[head_turn.output_root_data_id].payload)
                        if head_turn.output_root_data_id in data_map
                        else None
                    )
                    latest_user_text = input_data.text if input_data else None
                    latest_assistant_text = output_data.text if output_data else None
            items.append(
                ConversationListItem(
                    conversation=await self.repository.summarize_conversation(conversation),
                    latest_user_text=latest_user_text,
                    latest_assistant_text=latest_assistant_text,
                ),
            )
        return items

    async def get_conversation(
        self,
        conversation_id: int,
        *,
        account_id: int | None,
        is_staff: bool,
    ) -> ConversationSummary:
        conversation = await self._get_accessible_conversation(
            conversation_id,
            account_id=account_id,
            is_staff=is_staff,
        )
        if conversation is None:
            raise ApiException("会话不存在")
        return await self.repository.summarize_conversation(conversation)

    async def get_conversation_timeline(
        self,
        conversation_id: int,
        *,
        account_id: int | None,
        is_staff: bool,
    ) -> ConversationTimeline:
        conversation = await self._get_accessible_conversation(
            conversation_id,
            account_id=account_id,
            is_staff=is_staff,
        )
        if conversation is None:
            raise ApiException("会话不存在")

        turns = await self.repository.list_turns(conversation_id)
        data_ids = [
            data_id
            for turn in turns
            for data_id in [turn.input_root_data_id, turn.output_root_data_id]
            if data_id is not None
        ]
        data_map = await self.repository.get_data_map(data_ids)
        event_map = await self.repository.list_event_logs_by_turn_ids([turn.id for turn in turns])

        timeline_turns: list[ConversationTurnDetail] = []
        for turn in turns:
            input_payload = (
                MessageBundlePayload.model_validate(data_map[turn.input_root_data_id].payload)
                if turn.input_root_data_id in data_map
                else None
            )
            output_payload = (
                MessageBundlePayload.model_validate(data_map[turn.output_root_data_id].payload)
                if turn.output_root_data_id in data_map
                else None
            )
            timeline_turns.append(
                ConversationTurnDetail(
                    turn=await self.repository.summarize_turn(turn),
                    input=input_payload,
                    output=output_payload,
                    events=[event.payload for event in event_map.get(turn.id, [])],
                ),
            )

        return ConversationTimeline(
            conversation=await self.repository.summarize_conversation(conversation),
            turns=timeline_turns,
        )

    async def open_ws_session(
        self,
        *,
        account_id: int | None,
        client_info: dict[str, Any],
    ) -> tuple[str, int]:
        session_id = uuid4().hex
        ws_session = await self.repository.create_ws_session(
            session_id=session_id,
            account_id=account_id,
            conversation_id=None,
            client_info=client_info,
        )
        return session_id, ws_session.id

    async def update_conversation_resource_selection(
        self,
        conversation_id: int,
        payload: ResourceSelection,
        *,
        account_id: int | None,
        is_staff: bool,
    ) -> ConversationSummary:
        normalized_payload = self._normalize_inline_resource_selection(payload)
        conversation = await self._get_accessible_conversation(
            conversation_id,
            account_id=account_id,
            is_staff=is_staff,
        )
        if conversation is None:
            raise ApiException("会话不存在")
        conversation = await self.repository.update_conversation_default_resource_selection(
            conversation_id,
            normalized_payload,
        )
        assert conversation is not None
        return await self.repository.summarize_conversation(conversation)

    async def start_turn(
        self,
        payload: TurnStartRequest,
        *,
        ws_session_db_id: int | None,
        ws_public_session_id: str | None,
        account_id: int | None,
        is_staff: bool,
        send_event: Callable[[ChatEvent[Any]], Awaitable[None]],
    ) -> TurnStartAccepted:
        conversation = None
        if payload.conversation_id is not None:
            conversation = await self._get_accessible_conversation(
                payload.conversation_id,
                account_id=account_id,
                is_staff=is_staff,
            )
            if conversation is None:
                raise ApiException("会话不存在")
        else:
            normalized_resource_selection = self._normalize_inline_resource_selection(payload.resource_selection)
            conversation = await self.repository.create_conversation(
                title=self._derive_conversation_title(payload),
                user_id=account_id,
                resource_selection=normalized_resource_selection,
            )

        await self._validate_ws_session_access(
            ws_session_db_id=ws_session_db_id,
            ws_public_session_id=ws_public_session_id,
            conversation_id=conversation.id,
            account_id=account_id,
            is_staff=is_staff,
        )
        if ws_public_session_id and conversation.id is not None:
            await self.repository.touch_ws_session(
                ws_public_session_id,
                conversation_id=conversation.id,
            )
        resolved_selection = self._normalize_inline_resource_selection(payload.resource_selection)
        capability_selection, capability_decision = await self.capability_service.resolve_turn_capabilities(
            query=payload.input.text,
            resource_selection=resolved_selection,
            account_id=account_id,
            is_staff=is_staff,
        )
        logger.info(
            "Chat turn capability plan ready: conversation_id={}, "
            "selected_capability_ids={}, selected_capability_keys={}",
            conversation.id,
            capability_decision.selected_capability_ids,
            [candidate.capability_key for candidate in capability_decision.candidates if candidate.selected],
        )
        merged_selection = self.action_registry.normalize_selection(
            ResourceSelection(
                use_system_defaults=resolved_selection.use_system_defaults,
                use_conversation_defaults=resolved_selection.use_conversation_defaults,
                capabilities=[
                    *resolved_selection.normalized_capabilities(),
                    *capability_selection.normalized_capabilities(),
                ],
                actions=[
                    *capability_selection.normalized_actions(),
                ],
                metadata={
                    **resolved_selection.metadata,
                    **capability_selection.metadata,
                },
            ),
        )
        conversation_summary = await self.repository.summarize_conversation(conversation)
        turn_id = await self.runtime.execute_turn(
            ws_session_id=ws_session_db_id,
            ws_public_session_id=ws_public_session_id,
            conversation=conversation_summary,
            turn_request=payload.model_copy(
                update={
                    "conversation_id": conversation.id,
                    "resource_selection": merged_selection,
                    "metadata": {
                        **payload.metadata,
                        "capability_plan": capability_decision.model_dump(mode="json"),
                    },
                },
            ),
            account_id=account_id,
            is_staff=is_staff,
            send_event=send_event,
        )
        return TurnStartAccepted(turn_id=turn_id, conversation=conversation_summary)

    def _normalize_inline_resource_selection(self, selection: ResourceSelection) -> ResourceSelection:
        return self.action_registry.normalize_selection(
            ResourceSelection(
                use_system_defaults=False,
                use_conversation_defaults=False,
                actions=self.action_registry.assign_inline_action_ids(
                    selection.actions,
                    source="inline",
                    prefix="request:inline",
                ),
                metadata=selection.metadata,
            ),
        )

    async def cancel_turn(
        self,
        turn_id: int,
        *,
        account_id: int | None,
        is_staff: bool,
    ) -> bool:
        await self._get_accessible_turn(
            turn_id,
            account_id=account_id,
            is_staff=is_staff,
        )
        return await self.runtime.cancel_turn(turn_id)

    async def replay_turn_events(
        self,
        payload: TurnReplayRequest,
        *,
        account_id: int | None,
        is_staff: bool,
    ) -> list[dict[str, Any]]:
        await self._get_accessible_turn(
            payload.turn_id,
            account_id=account_id,
            is_staff=is_staff,
        )
        events = await self.repository.list_event_logs(payload.turn_id, min_seq=payload.last_seq)
        return [event.payload for event in events]

    async def _get_accessible_conversation(
        self,
        conversation_id: int,
        *,
        account_id: int | None,
        is_staff: bool,
    ):
        return await self.repository.get_accessible_conversation(
            conversation_id,
            account_id=account_id,
            is_staff=is_staff,
        )

    async def _get_accessible_turn(
        self,
        turn_id: int,
        *,
        account_id: int | None,
        is_staff: bool,
    ):
        turn = await self.repository.get_turn(turn_id)
        if turn is None:
            raise ApiException("turn不存在")
        conversation = await self._get_accessible_conversation(
            turn.conversation_id,  # type: ignore[arg-type]
            account_id=account_id,
            is_staff=is_staff,
        )
        if conversation is None:
            raise ApiException("turn不存在")
        return turn

    async def _validate_ws_session_access(
        self,
        *,
        ws_session_db_id: int | None,
        ws_public_session_id: str | None,
        conversation_id: int,
        account_id: int | None,
        is_staff: bool,
    ) -> None:
        if ws_session_db_id is None:
            return

        ws_session = await self.repository.get_ws_session_by_id(ws_session_db_id)
        if ws_session is None:
            raise ApiException("会话连接不存在")
        if ws_public_session_id and ws_session.session_id != ws_public_session_id:
            raise ApiException("会话连接不存在")
        if not is_staff and ws_session.account_id != account_id:
            raise ApiException("会话连接不存在")

    def _derive_conversation_title(self, payload: TurnStartRequest) -> str:
        if payload.conversation_title:
            return payload.conversation_title.strip()
        query = payload.input.text.strip()
        if not query:
            return "新会话"
        first_line = query.splitlines()[0].strip()
        if not first_line:
            return "新会话"
        if len(first_line) <= 32:
            return first_line
        return f"{first_line[:32].rstrip()}..."


chat_app_service = ChatApplicationService()
