from __future__ import annotations

from typing import Annotated

from fastapi import Query, Depends, Request, APIRouter

from service.chat import chat_app_service
from core.response import Resp
from service.depend import api_permission_check
from ext.ext_tortoise.curd import DeleteResp
from service.chat.domain.schema import ChatCapabilityKindEnum
from service.chat.capability.schema import (
    CapabilityBindingQuery,
    CapabilityProfileQuery,
    CapabilityBindingCreate,
    CapabilityBindingDetail,
    CapabilityBindingUpdate,
    CapabilityProfileCreate,
    CapabilityProfileUpdate,
    CapabilityProfileSummary,
    ChatCapabilityBindingOwnerEnum,
    ChatCapabilityProfileScopeEnum,
)
from ext.ext_tortoise.models.user_center import Account

router = APIRouter(dependencies=[Depends(api_permission_check)])


def _current_account(request: Request) -> Account:
    return request.scope["user"]


@router.post("/profile", response_model=Resp[CapabilityProfileSummary], summary="创建聊天能力配置")
async def create_capability_profile(
    request: Request,
    payload: CapabilityProfileCreate,
) -> Resp[CapabilityProfileSummary]:
    account = _current_account(request)
    data = await chat_app_service.capability_service.create_profile(
        payload,
        account_id=account.id,
        is_staff=account.is_staff,
    )
    return Resp(data=data)


@router.put("/profile/{profile_id}", response_model=Resp[CapabilityProfileSummary], summary="更新聊天能力配置")
async def update_capability_profile(
    request: Request,
    profile_id: int,
    payload: CapabilityProfileUpdate,
) -> Resp[CapabilityProfileSummary]:
    account = _current_account(request)
    data = await chat_app_service.capability_service.update_profile(
        profile_id,
        payload,
        account_id=account.id,
        is_staff=account.is_staff,
    )
    return Resp(data=data)


@router.get("/profile", response_model=Resp[list[CapabilityProfileSummary]], summary="聊天能力配置列表")
async def list_capability_profiles(
    request: Request,
    scope: Annotated[ChatCapabilityProfileScopeEnum, Query()] = ChatCapabilityProfileScopeEnum.all,
    kind: Annotated[ChatCapabilityKindEnum | None, Query()] = None,
    is_enabled: Annotated[bool | None, Query()] = None,
    name: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
) -> Resp[list[CapabilityProfileSummary]]:
    account = _current_account(request)
    data = await chat_app_service.capability_service.list_profiles(
        CapabilityProfileQuery(scope=scope, kind=kind, is_enabled=is_enabled, name=name),
        account_id=account.id,
        is_staff=account.is_staff,
    )
    return Resp(data=data)


@router.get("/profile/{profile_id}", response_model=Resp[CapabilityProfileSummary], summary="聊天能力配置详情")
async def get_capability_profile(request: Request, profile_id: int) -> Resp[CapabilityProfileSummary]:
    account = _current_account(request)
    data = await chat_app_service.capability_service.get_profile(
        profile_id,
        account_id=account.id,
        is_staff=account.is_staff,
    )
    return Resp(data=data)


@router.delete("/profile/{profile_id}", response_model=Resp[DeleteResp], summary="删除聊天能力配置")
async def delete_capability_profile(request: Request, profile_id: int) -> Resp[DeleteResp]:
    account = _current_account(request)
    deleted = await chat_app_service.capability_service.delete_profile(
        profile_id,
        account_id=account.id,
        is_staff=account.is_staff,
    )
    return Resp(data=DeleteResp(deleted=deleted))


@router.post("/binding", response_model=Resp[CapabilityBindingDetail], summary="创建聊天能力绑定")
async def create_capability_binding(
    request: Request,
    payload: CapabilityBindingCreate,
) -> Resp[CapabilityBindingDetail]:
    account = _current_account(request)
    data = await chat_app_service.capability_service.create_binding(
        payload,
        account_id=account.id,
        is_staff=account.is_staff,
    )
    return Resp(data=data)


@router.put("/binding/{binding_id}", response_model=Resp[CapabilityBindingDetail], summary="更新聊天能力绑定")
async def update_capability_binding(
    request: Request,
    binding_id: int,
    payload: CapabilityBindingUpdate,
) -> Resp[CapabilityBindingDetail]:
    account = _current_account(request)
    data = await chat_app_service.capability_service.update_binding(
        binding_id,
        payload,
        account_id=account.id,
        is_staff=account.is_staff,
    )
    return Resp(data=data)


@router.get("/binding", response_model=Resp[list[CapabilityBindingDetail]], summary="聊天能力绑定列表")
async def list_capability_bindings(
    request: Request,
    owner_type: Annotated[ChatCapabilityBindingOwnerEnum | None, Query()] = None,
    owner_id: Annotated[int | None, Query(ge=1)] = None,
    capability_profile_id: Annotated[int | None, Query(ge=1)] = None,
    is_enabled: Annotated[bool | None, Query()] = None,
) -> Resp[list[CapabilityBindingDetail]]:
    account = _current_account(request)
    data = await chat_app_service.capability_service.list_bindings(
        CapabilityBindingQuery(
            owner_type=owner_type,
            owner_id=owner_id,
            capability_profile_id=capability_profile_id,
            is_enabled=is_enabled,
        ),
        account_id=account.id,
        is_staff=account.is_staff,
    )
    return Resp(data=data)


@router.get("/binding/{binding_id}", response_model=Resp[CapabilityBindingDetail], summary="聊天能力绑定详情")
async def get_capability_binding(request: Request, binding_id: int) -> Resp[CapabilityBindingDetail]:
    account = _current_account(request)
    data = await chat_app_service.capability_service.get_binding(
        binding_id,
        account_id=account.id,
        is_staff=account.is_staff,
    )
    return Resp(data=data)


@router.delete("/binding/{binding_id}", response_model=Resp[DeleteResp], summary="删除聊天能力绑定")
async def delete_capability_binding(request: Request, binding_id: int) -> Resp[DeleteResp]:
    account = _current_account(request)
    deleted = await chat_app_service.capability_service.delete_binding(
        binding_id,
        account_id=account.id,
        is_staff=account.is_staff,
    )
    return Resp(data=DeleteResp(deleted=deleted))
