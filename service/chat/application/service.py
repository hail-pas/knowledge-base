from __future__ import annotations

from typing import Any, Awaitable, Callable
from loguru import logger

from core.types import ApiException
from service.chat.agent.service import ChatAgentService
from service.chat.capability.service import ChatCapabilityService
from service.chat.domain.schema import (
    ChatDataSchema,
    ChatEvent,
    ChatRequestContext,
    ConversationListItem,
    ConversationStepTrace,
    ConversationSummary,
    ConversationTimeline,
    ConversationTurnDetail,
    ErrorPayload,
    MessageBundlePayload,
    PersistedPayload,
    ResourceSelection,
    SessionReadyPayload,
    TurnStartAccepted,
    TurnStartRequest,
)
from service.chat.execution.registry import ExecutionActionRegistry, create_default_action_registry
from service.chat.platform_service import ChatPlatformService
from service.chat.runtime.engine import ChatRuntime
from service.chat.store.repository import ChatRepository


class ChatApplicationService:
    def __init__(self) -> None:
        self.repository: ChatRepository = ChatRepository()
        self.action_registry: ExecutionActionRegistry = create_default_action_registry()
        self.capability_service: ChatCapabilityService = ChatCapabilityService(
            action_registry=self.action_registry,
        )
        self.agent_service: ChatAgentService = ChatAgentService(
            action_registry=self.action_registry,
        )
        self.platform_service: ChatPlatformService = ChatPlatformService(
            capability_service=self.capability_service,
            agent_service=self.agent_service,
            action_registry=self.action_registry,
        )
        self.runtime: ChatRuntime = ChatRuntime(self.repository, self.action_registry)

    async def list_conversations(
        self,
        *,
        context: ChatRequestContext,
        limit: int = 50,
    ) -> list[ConversationListItem]:
        conversations = await self.repository.list_conversations(
            context=context,
            limit=limit,
        )
        conversation_summaries = await self.repository.summarize_conversations(conversations)
        latest_turn_map = await self.repository.list_latest_turns_by_conversation_ids(
            [item.id for item in conversations]
        )
        latest_data_by_id = await self._summarize_data_map(
            await self.repository.get_data_map(
                [
                    data_id
                    for turn in latest_turn_map.values()
                    for data_id in [turn.input_root_data_id, turn.output_root_data_id]
                    if data_id is not None
                ],
            ),
        )
        return [
            ConversationListItem(
                conversation=summary,
                latest_user_text=self._extract_data_payload_text(
                    latest_data_by_id, latest_turn_map.get(conversation.id), "input"
                ),
                latest_assistant_text=self._extract_data_payload_text(
                    latest_data_by_id,
                    latest_turn_map.get(conversation.id),
                    "output",
                ),
            )
            for conversation, summary in zip(conversations, conversation_summaries, strict=False)
        ]

    async def get_conversation(
        self,
        conversation_id: int,
        *,
        context: ChatRequestContext,
    ) -> ConversationSummary:
        conversation = await self._require_accessible_conversation(
            conversation_id,
            context=context,
        )
        return await self.repository.summarize_conversation(conversation)

    async def get_conversation_timeline(
        self,
        conversation_id: int,
        *,
        context: ChatRequestContext,
    ) -> ConversationTimeline:
        conversation = await self._require_accessible_conversation(
            conversation_id,
            context=context,
        )

        turns = await self.repository.list_turns(conversation_id)
        turn_ids = [turn.id for turn in turns]
        root_data_ids = [
            data_id
            for turn in turns
            for data_id in [turn.input_root_data_id, turn.output_root_data_id]
            if data_id is not None
        ]
        root_data_map = await self.repository.get_data_map(root_data_ids)
        step_map = await self.repository.list_steps_by_turn_ids(turn_ids)
        turn_data_map = await self.repository.list_data_by_turn_ids(turn_ids)
        conversation_summary = await self.repository.summarize_conversation(conversation)
        turn_summaries = {item.id: item for item in await self.repository.summarize_turns(turns)}
        step_summaries = {
            item.id: item
            for item in await self.repository.summarize_steps(
                [step for steps in step_map.values() for step in steps],
            )
        }
        summarized_data_by_id = await self._summarize_data_map(
            {row.id: row for rows in [list(root_data_map.values()), *turn_data_map.values()] for row in rows},
        )
        timeline_turns = [
            ConversationTurnDetail(
                turn=turn_summaries[turn.id],
                input=self._message_bundle_payload(summarized_data_by_id.get(turn.input_root_data_id)),
                output=self._persisted_payload(summarized_data_by_id.get(turn.output_root_data_id)),
                steps=self._build_conversation_step_traces(
                    turn_id=turn.id,
                    step_map=step_map,
                    step_summaries=step_summaries,
                    turn_data_map=turn_data_map,
                    summarized_data_by_id=summarized_data_by_id,
                ),
            )
            for turn in turns
        ]

        return ConversationTimeline(
            conversation=conversation_summary,
            turns=timeline_turns,
        )

    async def describe_session(
        self,
        *,
        context: ChatRequestContext,
    ) -> SessionReadyPayload:
        agents = await self.agent_service.list_agents(
            account_id=context.account_id,
            is_staff=context.is_staff,
            is_enabled=True,
        )
        packages = await self.capability_service.list_packages(
            query=self.capability_service.build_ready_query(),
            account_id=context.account_id,
            is_staff=context.is_staff,
        )
        return SessionReadyPayload(
            session_id=context.require_session_id(),
            default_agent_key=self.agent_service.DEFAULT_ORCHESTRATOR_KEY,
            available_agent_keys=[item.agent_key for item in agents],
            available_capability_keys=[item.capability_key for item in packages],
        )

    async def update_conversation_resource_selection(
        self,
        conversation_id: int,
        payload: ResourceSelection,
        *,
        context: ChatRequestContext,
    ) -> ConversationSummary:
        normalized_payload = self._normalize_inline_resource_selection(payload)
        conversation = await self._require_accessible_conversation(
            conversation_id,
            context=context,
        )
        conversation = await self.repository.update_conversation_default_resource_selection(
            conversation_id,
            normalized_payload,
        )
        if conversation is None:
            raise ApiException("会话更新失败")
        return await self.repository.summarize_conversation(conversation)

    async def start_turn(
        self,
        payload: TurnStartRequest,
        *,
        context: ChatRequestContext,
        send_event: Callable[[ChatEvent[Any]], Awaitable[None]],
    ) -> TurnStartAccepted:
        accepted = await self.prepare_turn(
            payload,
            context=context,
        )
        await self.launch_prepared_turn(accepted.turn_id, send_event=send_event)
        return accepted

    async def prepare_turn(
        self,
        payload: TurnStartRequest,
        *,
        context: ChatRequestContext,
    ) -> TurnStartAccepted:
        request_agent_key = payload.agent_key or self.agent_service.DEFAULT_ORCHESTRATOR_KEY
        normalized_request_selection = self._normalize_inline_resource_selection(payload.resource_selection)

        if payload.conversation_id is not None:
            conversation = await self._require_accessible_conversation(
                payload.conversation_id,
                context=context,
            )
        else:
            await self.agent_service.get_agent_by_key(
                request_agent_key,
                account_id=context.account_id,
                is_staff=context.is_staff,
            )
            conversation = await self.repository.create_conversation(
                agent_key=request_agent_key,
                title=self._derive_conversation_title(payload),
                user_id=context.account_id,
                resource_selection=ResourceSelection(),
            )

        conversation_summary = await self.repository.summarize_conversation(conversation)
        resolved_resources = await self.platform_service.resolve_turn_resources(
            query=payload.input.text,
            requested_agent_key=payload.agent_key or conversation_summary.agent_key,
            conversation_selection=conversation_summary.default_resource_selection,
            request_selection=normalized_request_selection,
            account_id=context.account_id,
            is_staff=context.is_staff,
        )
        logger.info(
            "Chat turn resources ready: conversation_id={}, agent_key={}",
            conversation.id,
            resolved_resources.agent.agent_key,
        )
        turn_id = await self.runtime.prepare_turn(
            context=context.with_conversation(conversation.id),
            conversation=conversation_summary,
            agent=resolved_resources.agent,
            turn_request=payload.model_copy(
                update={
                    "conversation_id": conversation.id,
                    "agent_key": resolved_resources.agent.agent_key,
                    "resource_selection": resolved_resources.resource_selection,
                },
            ),
            execution_plan=resolved_resources.runtime_plan,
        )
        return TurnStartAccepted(turn_id=turn_id, conversation=conversation_summary)

    async def launch_prepared_turn(
        self,
        turn_id: int,
        *,
        send_event: Callable[[ChatEvent[Any]], Awaitable[None]],
    ) -> None:
        await self.runtime.launch_prepared_turn(turn_id, send_event=send_event)

    async def abort_prepared_turn(self, turn_id: int) -> bool:
        return await self.runtime.abort_prepared_turn(turn_id)

    async def cancel_turn(
        self,
        turn_id: int,
        *,
        context: ChatRequestContext,
    ) -> bool:
        turn = await self.repository.get_accessible_turn(
            turn_id,
            context=context,
        )
        if turn is None:
            raise ApiException("会话不存在")
        return await self.runtime.cancel_turn(turn_id)

    def _normalize_inline_resource_selection(self, selection: ResourceSelection) -> ResourceSelection:
        return self.action_registry.normalize_inline_selection(
            selection,
            source="inline",
            prefix="request:inline",
        )

    async def _get_accessible_conversation(
        self,
        conversation_id: int,
        *,
        context: ChatRequestContext,
    ):
        return await self.repository.get_accessible_conversation(
            conversation_id,
            context=context,
        )

    async def _require_accessible_conversation(
        self,
        conversation_id: int,
        *,
        context: ChatRequestContext,
    ):
        conversation = await self._get_accessible_conversation(
            conversation_id,
            context=context,
        )
        if conversation is None:
            raise ApiException("会话不存在")
        return conversation

    def _derive_conversation_title(self, payload: TurnStartRequest) -> str:
        if payload.conversation_title:
            return payload.conversation_title.strip()[:255]
        query_text = payload.input.text.strip()
        if not query_text:
            return "新会话"
        return query_text[:40]

    def _payload_text(self, payload: PersistedPayload | dict[str, Any]) -> str | None:
        if isinstance(payload, MessageBundlePayload):
            return payload.text or None
        if isinstance(payload, ErrorPayload):
            return payload.message or None
        if not isinstance(payload, dict):
            return (
                str(getattr(payload, "content_text", None) or getattr(payload, "summary", None) or "").strip() or None
            )
        payload_type = str(payload.get("type") or "")
        if payload_type == "message_bundle":
            return MessageBundlePayload.model_validate(payload).text
        if payload_type == "error":
            return ErrorPayload.model_validate(payload).message
        return str(payload.get("content_text") or payload.get("summary") or "").strip() or None

    async def _summarize_data_map(self, rows_by_id: dict[int, Any]) -> dict[int, ChatDataSchema[PersistedPayload]]:
        return {item.id: item for item in await self.repository.summarize_data_rows(list(rows_by_id.values()))}

    def _extract_data_payload_text(
        self,
        data_by_id: dict[int, ChatDataSchema[PersistedPayload]],
        turn: Any | None,
        kind: str,
    ) -> str | None:
        if turn is None:
            return None
        data_id = turn.input_root_data_id if kind == "input" else turn.output_root_data_id
        data_summary = data_by_id.get(data_id)
        return self._payload_text(data_summary.payload) if data_summary is not None else None

    def _build_conversation_step_traces(
        self,
        *,
        turn_id: int,
        step_map: dict[int, list[Any]],
        step_summaries: dict[int, Any],
        turn_data_map: dict[int, list[Any]],
        summarized_data_by_id: dict[int, ChatDataSchema[PersistedPayload]],
    ) -> list[ConversationStepTrace]:
        step_data_by_step_id: dict[int, dict[str, ChatDataSchema[PersistedPayload]]] = {}
        for data_row in turn_data_map.get(turn_id, []):
            data_summary = summarized_data_by_id.get(data_row.id)
            if data_summary is None:
                continue
            step_data_by_step_id.setdefault(data_summary.step_id, {})[data_summary.kind.value] = data_summary
        return [
            ConversationStepTrace(
                step=step_summaries[step.id],
                input=step_data_by_step_id.get(step.id, {}).get("input"),
                output=step_data_by_step_id.get(step.id, {}).get("output"),
            )
            for step in step_map.get(turn_id, [])
        ]

    def _message_bundle_payload(
        self,
        data_summary: ChatDataSchema[PersistedPayload] | None,
    ) -> MessageBundlePayload | None:
        if data_summary is None or not isinstance(data_summary.payload, MessageBundlePayload):
            return None
        return data_summary.payload

    def _persisted_payload(self, data_summary: ChatDataSchema[PersistedPayload] | None) -> PersistedPayload | None:
        return data_summary.payload if data_summary is not None else None


chat_app_service = ChatApplicationService()
