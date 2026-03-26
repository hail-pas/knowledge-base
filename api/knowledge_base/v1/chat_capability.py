from __future__ import annotations

from typing import Annotated

from fastapi import Query, Depends, Request, APIRouter

from service.chat import chat_app_service
from core.response import Resp
from service.depend import api_permission_check
from ext.ext_tortoise.curd import DeleteResp
from service.chat.capability.schema import (
    CapabilityKindEnum,
    CapabilityCategoryEnum,
    CapabilityRuntimeKindEnum,
    CapabilityScopeEnum,
    CapabilityPackageQuery,
    CapabilityPackageCreate,
    CapabilityPackageUpdate,
    CapabilityPackageSummary,
)
from ext.ext_tortoise.models.user_center import Account

router = APIRouter(dependencies=[Depends(api_permission_check)])


def _current_account(request: Request) -> Account:
    return request.scope["user"]


@router.post("", response_model=Resp[CapabilityPackageSummary], summary="创建聊天 Capability Package")
async def create_capability_package(
    request: Request,
    payload: CapabilityPackageCreate,
) -> Resp[CapabilityPackageSummary]:
    account = _current_account(request)
    data = await chat_app_service.capability_service.create_package(
        payload,
        account_id=account.id,
        is_staff=account.is_staff,
    )
    return Resp(data=data)


@router.put("/{capability_id}", response_model=Resp[CapabilityPackageSummary], summary="更新聊天 Capability Package")
async def update_capability_package(
    request: Request,
    capability_id: int,
    payload: CapabilityPackageUpdate,
) -> Resp[CapabilityPackageSummary]:
    account = _current_account(request)
    data = await chat_app_service.capability_service.update_package(
        capability_id,
        payload,
        account_id=account.id,
        is_staff=account.is_staff,
    )
    return Resp(data=data)


@router.get("", response_model=Resp[list[CapabilityPackageSummary]], summary="聊天 Capability Package 列表")
async def list_capability_packages(
    request: Request,
    scope: Annotated[CapabilityScopeEnum, Query()] = CapabilityScopeEnum.all,
    kind: Annotated[CapabilityKindEnum | None, Query()] = None,
    category: Annotated[CapabilityCategoryEnum | None, Query()] = None,
    runtime_kind: Annotated[CapabilityRuntimeKindEnum | None, Query()] = None,
    is_enabled: Annotated[bool | None, Query()] = None,
    name: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    tags: Annotated[list[str] | None, Query()] = None,
) -> Resp[list[CapabilityPackageSummary]]:
    account = _current_account(request)
    data = await chat_app_service.capability_service.list_packages(
        CapabilityPackageQuery(
            scope=scope,
            kind=kind,
            category=category,
            runtime_kind=runtime_kind,
            is_enabled=is_enabled,
            name=name,
            tags=tags or [],
        ),
        account_id=account.id,
        is_staff=account.is_staff,
    )
    return Resp(data=data)


@router.get("/{capability_id}", response_model=Resp[CapabilityPackageSummary], summary="聊天 Capability Package 详情")
async def get_capability_package(request: Request, capability_id: int) -> Resp[CapabilityPackageSummary]:
    account = _current_account(request)
    data = await chat_app_service.capability_service.get_package(
        capability_id,
        account_id=account.id,
        is_staff=account.is_staff,
    )
    return Resp(data=data)


@router.delete("/{capability_id}", response_model=Resp[DeleteResp], summary="删除聊天 Capability Package")
async def delete_capability_package(request: Request, capability_id: int) -> Resp[DeleteResp]:
    account = _current_account(request)
    deleted = await chat_app_service.capability_service.delete_package(
        capability_id,
        account_id=account.id,
        is_staff=account.is_staff,
    )
    return Resp(data=DeleteResp(deleted=deleted))
