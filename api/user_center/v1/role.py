from typing import Annotated

from fastapi import Depends, Request, APIRouter
from tortoise.queryset import QuerySet
from tortoise.transactions import in_transaction

from api.depend import api_permission_check
from core.types import ApiException
from core.schema import CRUDPager
from core.response import Resp, PageData
from ext.ext_tortoise.curd import (
    list_view,
    create_obj,
    delete_view,
    update_view,
    pagination_factory,
    obj_prefetch_fields,
)
from api.service.role.schema import (
    RoleList,
    RoleCreate,
    RoleDetail,
    RoleUpdate,
    RoleFilterSchema,
)
from api.service.resource.helper import resource_list_to_trees
from ext.ext_tortoise.models.user_center import Role

router = APIRouter(dependencies=[Depends(api_permission_check)])


def get_queryset(request: Request) -> QuerySet[Role]:
    filter_ = {"deleted_at": 0}
    return Role.filter(**filter_)


@router.get(
    "",
    description=f"{Role.Meta.table_description}列表",
    summary=f"{Role.Meta.table_description}列表",
)
async def get_role_list(
    request: Request,
    filter_: Annotated[RoleFilterSchema, Depends(RoleFilterSchema.as_query)],  # type: ignore
    pager: CRUDPager = pagination_factory(
        db_model=Role,
        list_schema=RoleList,
        search_fields=set(),
        order_fields={
            "created_at",
        },
        max_limit=1000,
    ),
) -> Resp[PageData[RoleList]]:
    return await list_view(get_queryset(request), filter_, pager)


@router.get(
    "/{pk}",
    description=f"获取{Role.Meta.table_description}详情",
    summary=f"获取{Role.Meta.table_description}详情",
)
async def get_role_detail(request: Request, pk: int) -> Resp[RoleDetail]:
    queryset = get_queryset(request)
    obj: Role | None = await queryset.get_or_none(
        **{queryset.model._meta.pk_attr: pk},
    )
    if not obj:
        raise ApiException("对象不存在")
    obj = await obj_prefetch_fields(obj, RoleDetail)  # type: ignore
    data = RoleDetail.model_validate(obj)
    data.resources = resource_list_to_trees(await obj.resources.filter(enabled=True))  # type: ignore
    return Resp(
        data=data,  # type: ignore
    )


@router.post(
    "",
    description=f"创建{Role.Meta.table_description}",
    summary=f"创建{Role.Meta.table_description}",
)
async def create_role(request: Request, schema: RoleCreate) -> Resp:
    async with in_transaction(connection_name=Role.Meta.app):
        await create_obj(Role, schema.model_dump(exclude_unset=True))

    return Resp()


@router.put("/{pk}", description=f"更新{Role.Meta.table_description}", summary=f"更新{Role.Meta.table_description}")
async def update_role(request: Request, pk: int, schema: RoleUpdate) -> Resp:
    return await update_view(get_queryset(request), pk, schema)  # type: ignore


@router.delete(
    "/{pk}",
    description=f"删除{Role.Meta.table_description}",
    summary=f"删除{Role.Meta.table_description}",
)
async def delete_role(request: Request, pk: int) -> Resp:
    return await delete_view(pk, get_queryset(request))
