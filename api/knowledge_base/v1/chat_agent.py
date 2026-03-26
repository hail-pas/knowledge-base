from __future__ import annotations

from typing import Annotated

from fastapi import Query, Depends, Request, APIRouter

from service.chat import chat_app_service
from core.response import Resp
from service.depend import api_permission_check
from service.chat.agent.schema import (
    AgentMountCreate,
    AgentMountSummary,
    AgentMountUpdate,
    AgentProfileCreate,
    AgentProfileSummary,
    AgentProfileUpdate,
)
from service.chat.domain.schema import AgentRoleEnum
from ext.ext_tortoise.models.user_center import Account

router = APIRouter(dependencies=[Depends(api_permission_check)])


def _current_account(request: Request) -> Account:
    return request.scope["user"]


@router.post("", response_model=Resp[AgentProfileSummary], summary="创建聊天 Agent")
async def create_agent(
    request: Request,
    payload: AgentProfileCreate,
) -> Resp[AgentProfileSummary]:
    account = _current_account(request)
    data = await chat_app_service.agent_service.create_agent(
        payload,
        account_id=account.id,
        is_staff=account.is_staff,
    )
    return Resp(data=data)


@router.put("/{agent_id}", response_model=Resp[AgentProfileSummary], summary="更新聊天 Agent")
async def update_agent(
    request: Request,
    agent_id: int,
    payload: AgentProfileUpdate,
) -> Resp[AgentProfileSummary]:
    account = _current_account(request)
    data = await chat_app_service.agent_service.update_agent(
        agent_id,
        payload,
        account_id=account.id,
        is_staff=account.is_staff,
    )
    return Resp(data=data)


@router.get("", response_model=Resp[list[AgentProfileSummary]], summary="聊天 Agent 列表")
async def list_agents(
    request: Request,
    role: Annotated[AgentRoleEnum | None, Query()] = None,
    is_enabled: Annotated[bool | None, Query()] = None,
) -> Resp[list[AgentProfileSummary]]:
    account = _current_account(request)
    data = await chat_app_service.agent_service.list_agents(
        account_id=account.id,
        is_staff=account.is_staff,
        role=role,
        is_enabled=is_enabled,
    )
    return Resp(data=data)


@router.post("/mount", response_model=Resp[AgentMountSummary], summary="创建 Agent Mount")
async def create_agent_mount(
    request: Request,
    payload: AgentMountCreate,
) -> Resp[AgentMountSummary]:
    _ = _current_account(request)
    data = await chat_app_service.agent_service.create_mount(payload)
    return Resp(data=data)


@router.put("/mount/{mount_id}", response_model=Resp[AgentMountSummary], summary="更新 Agent Mount")
async def update_agent_mount(
    request: Request,
    mount_id: int,
    payload: AgentMountUpdate,
) -> Resp[AgentMountSummary]:
    _ = _current_account(request)
    data = await chat_app_service.agent_service.update_mount(mount_id, payload)
    return Resp(data=data)


@router.get("/mount/list", response_model=Resp[list[AgentMountSummary]], summary="Agent Mount 列表")
async def list_agent_mounts(
    request: Request,
    source_agent_id: Annotated[int | None, Query(ge=1)] = None,
    mounted_agent_id: Annotated[int | None, Query(ge=1)] = None,
    is_enabled: Annotated[bool | None, Query()] = None,
) -> Resp[list[AgentMountSummary]]:
    _ = _current_account(request)
    data = await chat_app_service.agent_service.list_mounts(
        source_agent_id=source_agent_id,
        mounted_agent_id=mounted_agent_id,
        is_enabled=is_enabled,
    )
    return Resp(data=data)


@router.get("/{agent_id}", response_model=Resp[AgentProfileSummary], summary="聊天 Agent 详情")
async def get_agent(request: Request, agent_id: int) -> Resp[AgentProfileSummary]:
    account = _current_account(request)
    data = await chat_app_service.agent_service.get_agent(
        agent_id,
        account_id=account.id,
        is_staff=account.is_staff,
    )
    return Resp(data=data)
