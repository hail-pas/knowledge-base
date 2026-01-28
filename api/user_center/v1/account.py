from uuid import UUID
from typing import Annotated

from fastapi import Depends, Request, APIRouter

#  tortoise-orm
from tortoise.queryset import QuerySet

from api.depend import api_permission_check
from core.schema import CRUDPager
from core.response import Resp, PageData
from ext.ext_tortoise.curd import (
    list_view,
    create_obj,
    delete_view,
    detail_view,
    update_view,
    pagination_factory,
)
from api.service.account.schema import (
    AccountList,
    AccountCreate,
    AccountDetail,
    AccountUpdate,
    AccountFilterSchema,
)
from ext.ext_tortoise.models.user_center import Account

router = APIRouter(dependencies=[Depends(api_permission_check)])


def get_queryset(request: Request) -> QuerySet[Account]:
    queryset = Account.filter(
        deleted_at=0,
    )
    return queryset


@router.post("", description=f"创建{Account.Meta.table_description}", summary=f"创建{Account.Meta.table_description}")
async def create_account(request: Request, schema: AccountCreate) -> Resp:
    await create_obj(Account, schema.model_dump(exclude_unset=True))
    return Resp()


@router.put(
    "/{pk}", description=f"更新{Account.Meta.table_description}", summary=f"更新{Account.Meta.table_description}",
)
async def update_account(request: Request, pk: int, schema: AccountUpdate) -> Resp:
    return await update_view(get_queryset(request), pk, schema)  # type: ignore


@router.get("", description=f"{Account.Meta.table_description}列表", summary=f"{Account.Meta.table_description}列表")
async def get_account_list(
    request: Request,
    filter_: Annotated[AccountFilterSchema, Depends(AccountFilterSchema.as_query)],  # type: ignore
    pager: CRUDPager = pagination_factory(
        db_model=Account,
        search_fields=set(),
        order_fields={
            "id",
        },
        list_schema=AccountList,
        max_limit=1000,
    ),
) -> Resp[PageData[AccountList]]:
    return await list_view(get_queryset(request), filter_, pager)


@router.get(
    "/{pk}",
    description=f"获取{Account.Meta.table_description}详情",
    summary=f"获取{Account.Meta.table_description}详情",
)
async def get_account_detail(request: Request, pk: int) -> Resp[AccountDetail]:
    return await detail_view(get_queryset(request), pk, AccountDetail)


@router.delete(
    "/{pk}", description=f"删除{Account.Meta.table_description}", summary=f"删除{Account.Meta.table_description}",
)
async def delete_account(request: Request, pk: UUID) -> Resp:
    return await delete_view(pk, get_queryset(request))
