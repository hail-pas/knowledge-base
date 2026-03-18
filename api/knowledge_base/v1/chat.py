from __future__ import annotations

from typing import Any, Annotated
from pathlib import Path
from datetime import UTC, datetime

from loguru import logger
from fastapi import (
    Query,
    Depends,
    Request,
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import FileResponse

from core.types import ApiException
from service.chat import chat_app_service
from core.response import Resp
from service.depend import api_permission_check, authenticate_websocket
from service.chat.domain.schema import (
    ChatEvent,
    AckPayload,
    AckRequest,
    PingRequest,
    ErrorPayload,
    EventNameEnum,
    TurnStartRequest,
    ChatErrorCodeEnum,
    ClientCommandEnum,
    ResourceSelection,
    TurnCancelRequest,
    TurnReplayRequest,
    ConversationSummary,
    ConversationListItem,
    ConversationTimeline,
)
from ext.ext_tortoise.models.user_center import Account

router = APIRouter()
_DEMO_FILE = Path(__file__).resolve().parents[1] / "static" / "chat_demo" / "index.html"


def _current_account(request: Request) -> Account:
    return request.scope["user"]


@router.get("/demo", include_in_schema=False)
async def chat_demo_page() -> FileResponse:
    return FileResponse(
        _DEMO_FILE,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.get(
    "/conversation",
    response_model=Resp[list[ConversationListItem]],
    dependencies=[Depends(api_permission_check)],
    summary="聊天会话列表",
)
async def list_conversations(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> Resp[list[ConversationListItem]]:
    account = _current_account(request)
    data = await chat_app_service.list_conversations(
        account_id=account.id,
        is_staff=account.is_staff,
        limit=limit,
    )
    return Resp(data=data)


@router.get(
    "/conversation/{conversation_id}",
    response_model=Resp[ConversationSummary],
    dependencies=[Depends(api_permission_check)],
    summary="聊天会话详情",
)
async def get_conversation(request: Request, conversation_id: int) -> Resp[ConversationSummary]:
    account = _current_account(request)
    data = await chat_app_service.get_conversation(
        conversation_id,
        account_id=account.id,
        is_staff=account.is_staff,
    )
    return Resp(data=data)


@router.get(
    "/conversation/{conversation_id}/timeline",
    response_model=Resp[ConversationTimeline],
    dependencies=[Depends(api_permission_check)],
    summary="聊天会话完整时间线",
)
async def get_conversation_timeline(request: Request, conversation_id: int) -> Resp[ConversationTimeline]:
    account = _current_account(request)
    data = await chat_app_service.get_conversation_timeline(
        conversation_id,
        account_id=account.id,
        is_staff=account.is_staff,
    )
    return Resp(data=data)


@router.put(
    "/conversation/{conversation_id}/resource-selection",
    response_model=Resp[ConversationSummary],
    dependencies=[Depends(api_permission_check)],
    summary="更新会话默认能力选择",
)
async def update_conversation_resource_selection(
    request: Request,
    conversation_id: int,
    payload: ResourceSelection,
) -> Resp[ConversationSummary]:
    account = _current_account(request)
    data = await chat_app_service.update_conversation_resource_selection(
        conversation_id,
        payload,
        account_id=account.id,
        is_staff=account.is_staff,
    )
    return Resp(data=data)


@router.get(
    "/turn/{turn_id}/events",
    response_model=Resp[list[dict[str, Any]]],
    dependencies=[Depends(api_permission_check)],
    summary="回放turn事件",
)
async def replay_turn_events(
    request: Request,
    turn_id: int,
    last_seq: Annotated[int, Query(ge=0)] = 0,
) -> Resp[list[dict[str, Any]]]:
    account = _current_account(request)
    data = await chat_app_service.replay_turn_events(
        TurnReplayRequest(turn_id=turn_id, last_seq=last_seq),
        account_id=account.id,
        is_staff=account.is_staff,
    )
    return Resp(data=data)


@router.websocket("/ws")
async def chat_websocket(websocket: WebSocket) -> None:
    logger.info("Chat websocket connect requested: path={}, client={}", websocket.url.path, websocket.client)
    account = None
    startup_error: ErrorPayload | None = None
    startup_close_code = status.WS_1011_INTERNAL_ERROR

    try:
        account = await authenticate_websocket(websocket)
    except ApiException as exc:
        startup_error = ErrorPayload(message=exc.message, code=ChatErrorCodeEnum.auth_failed)
        startup_close_code = status.WS_1008_POLICY_VIOLATION
        logger.warning("Chat websocket auth failed: {}", exc.message)
    except Exception as exc:
        startup_error = ErrorPayload(
            message=str(exc) or "WebSocket 初始化失败",
            code=ChatErrorCodeEnum.socket_init_failed,
        )
        logger.exception("Chat websocket init failed before accept")

    await websocket.accept()
    session_id: str | None = None
    ws_session_db_id: int | None = None
    if startup_error is not None:
        await websocket.send_json(
            ChatEvent[ErrorPayload](
                id="socket_startup_error",
                session_id=None,
                conversation_id=None,
                seq=0,
                event=EventNameEnum.error.value,
                ts=datetime.now(UTC),
                payload=startup_error,
            ).model_dump(mode="json"),
        )
        await websocket.close(code=startup_close_code, reason=startup_error.message[:120])
        return

    assert account is not None
    try:
        session_id, ws_session_db_id = await chat_app_service.open_ws_session(
            account_id=account.id,
            client_info={
                "client": "browser",
                "user_agent": websocket.headers.get("user-agent", ""),
            },
        )
    except Exception as exc:
        logger.exception("Chat websocket init failed after accept")
        await websocket.send_json(
            ChatEvent[ErrorPayload](
                id="socket_session_error",
                session_id=None,
                conversation_id=None,
                seq=0,
                event=EventNameEnum.error.value,
                ts=datetime.now(UTC),
                payload=ErrorPayload(
                    message=str(exc) or "WebSocket 会话初始化失败",
                    code=ChatErrorCodeEnum.socket_session_error,
                ),
            ).model_dump(mode="json"),
        )
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        return

    conversation_id: int | None = None

    async def send_event(event: ChatEvent[Any]) -> None:
        await websocket.send_json(event.model_dump(mode="json"))

    try:
        while True:
            raw = await websocket.receive_json()
            try:
                try:
                    command = ClientCommandEnum(raw.get("command"))
                except ValueError:
                    command = None
                payload = raw.get("payload", {})

                if command == ClientCommandEnum.turn_start:
                    has_conversation_id = "conversation_id" in payload
                    start_payload = TurnStartRequest.model_validate(payload)
                    if start_payload.conversation_id is None and not has_conversation_id:
                        start_payload = start_payload.model_copy(update={"conversation_id": conversation_id})
                    accepted = await chat_app_service.start_turn(
                        start_payload,
                        ws_session_db_id=ws_session_db_id,
                        ws_public_session_id=session_id,
                        account_id=account.id,
                        is_staff=account.is_staff,
                        send_event=send_event,
                    )
                    turn_id = accepted.turn_id
                    conversation_id = accepted.conversation.id
                    await send_event(
                        ChatEvent[AckPayload](
                            id=f"ack_{turn_id}",
                            session_id=session_id,
                            conversation_id=conversation_id,
                            turn_id=turn_id,
                            seq=0,
                            event=EventNameEnum.ack.value,
                            ts=datetime.now(UTC),
                            payload=AckPayload(command=command, request_id=start_payload.request_id),
                        ),
                    )
                    continue

                if command == ClientCommandEnum.turn_cancel:
                    cancel_payload = TurnCancelRequest.model_validate(payload)
                    canceled = await chat_app_service.cancel_turn(
                        cancel_payload.turn_id,
                        account_id=account.id,
                        is_staff=account.is_staff,
                    )
                    await send_event(
                        ChatEvent[AckPayload](
                            id=f"ack_cancel_{cancel_payload.turn_id}",
                            session_id=session_id,
                            conversation_id=conversation_id,
                            turn_id=cancel_payload.turn_id,
                            seq=0,
                            event=EventNameEnum.ack.value,
                            ts=datetime.now(UTC),
                            payload=AckPayload(command=command, accepted=canceled),
                        ),
                    )
                    continue

                if command == ClientCommandEnum.turn_resume:
                    replay_payload = TurnReplayRequest.model_validate(payload)
                    for item in await chat_app_service.replay_turn_events(
                        replay_payload,
                        account_id=account.id,
                        is_staff=account.is_staff,
                    ):
                        await websocket.send_json(item)
                    continue

                if command == ClientCommandEnum.ack:
                    ack_payload = AckRequest.model_validate(payload)
                    if session_id:
                        await chat_app_service.repository.touch_ws_session(session_id, last_ack_seq=ack_payload.seq)
                    continue

                if command == ClientCommandEnum.ping:
                    ping_payload = PingRequest.model_validate(payload)
                    await send_event(
                        ChatEvent[PingRequest](
                            id="pong",
                            session_id=session_id,
                            conversation_id=conversation_id,
                            seq=0,
                            event=EventNameEnum.ping.value,
                            ts=datetime.now(UTC),
                            payload=ping_payload,
                        ),
                    )
                    continue

                await send_event(
                    ChatEvent[ErrorPayload](
                        id="unknown_command",
                        session_id=session_id,
                        conversation_id=conversation_id,
                        seq=0,
                        event=EventNameEnum.error.value,
                        ts=datetime.now(UTC),
                        payload=ErrorPayload(
                            message=f"Unknown command: {raw.get('command')}",
                            code=ChatErrorCodeEnum.unknown_command,
                        ),
                    ),
                )
            except ApiException as exc:
                await send_event(
                    ChatEvent[ErrorPayload](
                        id="command_error",
                        session_id=session_id,
                        conversation_id=conversation_id,
                        seq=0,
                        event=EventNameEnum.error.value,
                        ts=datetime.now(UTC),
                        payload=ErrorPayload(message=exc.message, code=ChatErrorCodeEnum.command_error),
                    ),
                )
    except WebSocketDisconnect:
        if session_id:
            await chat_app_service.repository.touch_ws_session(session_id, status="disconnected")
