from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from fastapi import WebSocket
from loguru import logger
from pydantic import ValidationError

from core.types import ApiException
from ext.ext_tortoise.models.user_center import Account
from service.chat.application.service import ChatApplicationService
from service.chat.domain.schema import (
    AckPayload,
    ChatErrorCodeEnum,
    ChatEvent,
    ChatRequestContext,
    ClientCommandEnum,
    ErrorPayload,
    EventNameEnum,
    PingRequest,
    TurnCancelRequest,
    TurnStartRequest,
    ValidatedClientCommand,
    parse_client_command,
)


class ChatWebSocketSession:
    def __init__(
        self,
        websocket: WebSocket,
        *,
        app_service: ChatApplicationService,
        context: ChatRequestContext,
    ) -> None:
        self.websocket = websocket
        self.app_service = app_service
        self.context = context
        self._handlers = {
            ClientCommandEnum.turn_start: self._handle_turn_start,
            ClientCommandEnum.turn_cancel: self._handle_turn_cancel,
            ClientCommandEnum.ping: self._handle_ping,
        }

    @property
    def session_id(self) -> UUID | None:
        return self.context.session_id

    def bind_account(self, account: Account) -> None:
        self.context = self.context.model_copy(update={"account": account})

    async def send(
        self,
        *,
        event: str,
        payload: Any,
        event_id: str,
        turn_id: int | None = None,
        conversation_id: int | None = None,
    ) -> None:
        await self.websocket.send_json(
            ChatEvent[Any](
                id=event_id,
                session_id=self.context.session_id,
                conversation_id=self.context.conversation_id if conversation_id is None else conversation_id,
                turn_id=turn_id,
                seq=0,
                event=event,
                ts=datetime.now(UTC),
                payload=payload,
            ).model_dump(mode="json"),
        )

    async def send_error(
        self,
        *,
        code: ChatErrorCodeEnum,
        message: str,
        event_id: str | None = None,
        turn_id: int | None = None,
    ) -> None:
        await self.send(
            event=EventNameEnum.error.value,
            payload=ErrorPayload(message=message, code=code),
            event_id=event_id or f"error_{uuid4().hex[:12]}",
            turn_id=turn_id,
        )

    async def send_session_ready(self) -> None:
        payload = await self.app_service.describe_session(context=self.context)
        await self.send(
            event=EventNameEnum.session_ready.value,
            payload=payload,
            event_id="session_ready",
            conversation_id=None,
        )

    async def serve(self) -> None:
        while True:
            raw = await self.websocket.receive_json()
            command = await self._parse_command(raw)
            if command is None:
                continue
            try:
                await self.dispatch(command)
            except ApiException as exc:
                await self.send_error(
                    code=ChatErrorCodeEnum.command_error,
                    message=exc.message,
                )
            except Exception as exc:
                logger.exception("Chat websocket command failed")
                await self.send_error(
                    code=ChatErrorCodeEnum.command_error,
                    message=str(exc) or "命令执行失败",
                )

    async def _parse_command(self, raw: Any) -> ValidatedClientCommand | None:
        try:
            return parse_client_command(raw)
        except (ValidationError, ValueError) as exc:
            await self.send_error(
                code=ChatErrorCodeEnum.invalid_command_payload,
                message=str(exc) or "命令载荷非法",
            )
            return None

    async def dispatch(self, command: ValidatedClientCommand) -> None:
        handler = self._handlers[ClientCommandEnum(command.command)]
        await handler(command.payload)

    async def _send_ack(
        self,
        *,
        command: ClientCommandEnum,
        turn_id: int | None = None,
        request_id: UUID | None = None,
        accepted: bool = True,
    ) -> None:
        ack_suffix = turn_id if turn_id is not None else uuid4().hex[:12]
        await self.send(
            event=EventNameEnum.ack.value,
            payload=AckPayload(command=command, accepted=accepted, request_id=request_id),
            event_id=f"ack_{ack_suffix}",
            turn_id=turn_id,
        )

    async def _handle_turn_start(self, payload: TurnStartRequest) -> None:
        start_payload = payload.reuse_bound_conversation(self.context.conversation_id)
        accepted = await self.app_service.prepare_turn(
            start_payload,
            context=self.context,
        )
        self.context = self.context.with_conversation(accepted.conversation.id)
        try:
            await self._send_ack(
                command=ClientCommandEnum.turn_start,
                turn_id=accepted.turn_id,
                request_id=start_payload.request_id,
            )
            await self.app_service.launch_prepared_turn(
                accepted.turn_id,
                send_event=self._forward_turn_event,
            )
        except Exception:
            await self.app_service.abort_prepared_turn(accepted.turn_id)
            raise

    async def _handle_turn_cancel(self, payload: TurnCancelRequest) -> None:
        canceled = await self.app_service.cancel_turn(
            payload.turn_id,
            context=self.context,
        )
        await self._send_ack(
            command=ClientCommandEnum.turn_cancel,
            turn_id=payload.turn_id,
            accepted=canceled,
        )

    async def _handle_ping(self, payload: PingRequest) -> None:
        await self.send(
            event=EventNameEnum.ping.value,
            payload=payload,
            event_id=f"pong_{payload.nonce}" if payload.nonce else "pong",
        )

    async def _forward_turn_event(self, event: ChatEvent[Any]) -> None:
        enriched = ChatEvent[Any](
            id=event.id,
            session_id=self.context.session_id,
            conversation_id=self.context.conversation_id if event.conversation_id is None else event.conversation_id,
            turn_id=event.turn_id,
            seq=event.seq,
            event=event.event,
            ts=event.ts,
            payload=event.payload,
        )
        await self.websocket.send_json(enriched.model_dump(mode="json"))
