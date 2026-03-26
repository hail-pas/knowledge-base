from __future__ import annotations

from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import FileResponse
from loguru import logger

from core.response import Resp
from core.types import ApiException
from ext.ext_tortoise.models.user_center import Account
from service.chat import chat_app_service
from service.chat.application.websocket_session import ChatWebSocketSession
from service.chat.domain.schema import (
    ChatErrorCodeEnum,
    ChatRequestContext,
    ConversationListItem,
    ConversationSummary,
    ConversationTimeline,
    ErrorPayload,
    ResourceSelection,
)
from service.depend import api_permission_check, authenticate_websocket

router = APIRouter()
_DEBUG_FILE = Path(__file__).resolve().parents[1] / "static" / "chat_debug" / "index.html"


@router.get("/debug", include_in_schema=False)
async def chat_debug_page() -> FileResponse:
    return FileResponse(
        _DEBUG_FILE,
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
    data = await chat_app_service.list_conversations(
        context=ChatRequestContext(account=request.scope["user"]),
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
    data = await chat_app_service.get_conversation(
        conversation_id,
        context=ChatRequestContext(account=request.scope["user"]),
    )
    return Resp(data=data)


@router.get(
    "/conversation/{conversation_id}/timeline",
    response_model=Resp[ConversationTimeline],
    dependencies=[Depends(api_permission_check)],
    summary="聊天会话完整时间线",
)
async def get_conversation_timeline(request: Request, conversation_id: int) -> Resp[ConversationTimeline]:
    data = await chat_app_service.get_conversation_timeline(
        conversation_id,
        context=ChatRequestContext(account=request.scope["user"]),
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
    data = await chat_app_service.update_conversation_resource_selection(
        conversation_id,
        payload,
        context=ChatRequestContext(account=request.scope["user"]),
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
        logger.warning("Chat websocket auth failed: {}", exc.message)
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=exc.message[:120])
        return
    except Exception as exc:
        logger.exception("Chat websocket init failed before accept")
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason=str(exc)[:120])
        return

    await websocket.accept()
    session = ChatWebSocketSession(
        websocket,
        app_service=chat_app_service,
        context=ChatRequestContext(account=account, session_id=uuid4()),
    )

    session.bind_account(account)
    await session.send_session_ready()

    try:
        await session.serve()
    except WebSocketDisconnect:
        logger.info("Chat websocket disconnected: session_id={}", session.session_id)
