from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from tortoise.queryset import QuerySet
from tortoise.transactions import in_transaction

from core.types import ApiException
from core.schema import CRUDPager
from core.response import Resp, PageData
from service.depend import CurrentUser, api_permission_check, ensure_super_admin
from service.role.schema import RoleList, RoleCreate, RoleDetail, RoleUpdate, RoleFilterSchema
from service.resource.helper import resource_list_to_trees
from ext.ext_tortoise.curd import list_view, pagination_factory, obj_prefetch_fields, delete_view
from ext.ext_tortoise.models.authorization import Role, Resource, UserRole

router = APIRouter(dependencies=[Depends(api_permission_check)])


def get_queryset(request: Request) -> QuerySet[Role]:
    return Role.filter(deleted_at=0)


async def _set_role_resources(role: Role, resource_ids: list[int]) -> None:
    unique_ids = list(dict.fromkeys(resource_ids))
    resources = await Resource.filter(id__in=unique_ids, enabled=True, assignable=True)
    if len(resources) != len(unique_ids):
        raise ApiException("包含不存在、已禁用或不可分配的资源", code=422)
    await role.resources.clear()
    if resources:
        await role.resources.add(*resources)


@router.get("", summary="角色列表")
async def get_role_list(
    request: Request,
    filter_: Annotated[RoleFilterSchema, Query()],
    pager: CRUDPager = pagination_factory(
        db_model=Role,
        list_schema=RoleList,
        search_fields={"label"},
        order_fields={"created_at", "id"},
        max_limit=1000,
    ),
) -> Resp[PageData[RoleList]]:
    return await list_view(get_queryset(request), filter_, pager)


@router.get("/{pk}", summary="角色详情")
async def get_role_detail(request: Request, pk: int) -> Resp[RoleDetail]:
    obj = await get_queryset(request).get_or_none(id=pk)
    if not obj:
        raise ApiException("角色不存在", code=404)
    obj = await obj_prefetch_fields(obj, RoleDetail)  # type: ignore
    data = RoleDetail.model_validate(obj)
    data.resources = await resource_list_to_trees(await obj.resources.filter(enabled=True, assignable=True))  # type: ignore
    return Resp(data=data)


@router.post("", summary="创建角色")
async def create_role(request: Request, schema: RoleCreate) -> Resp[RoleList]:
    current_user: CurrentUser = request.scope["user"]
    await ensure_super_admin(current_user.id)
    async with in_transaction(connection_name=Role.Meta.app):
        obj = await Role.create(
            label=schema.label,
            remark=schema.remark,
            is_super_admin=schema.is_super_admin,
        )
        await _set_role_resources(obj, schema.resources)
    return Resp(data=RoleList.model_validate(obj))


@router.put("/{pk}", summary="更新角色")
async def update_role(request: Request, pk: int, schema: RoleUpdate) -> Resp:
    current_user: CurrentUser = request.scope["user"]
    await ensure_super_admin(current_user.id)

    obj = await get_queryset(request).get_or_none(id=pk)
    if not obj:
        raise ApiException("角色不存在", code=404)

    data = schema.model_dump(exclude_unset=True)
    resource_ids = data.pop("resources", None)
    async with in_transaction(connection_name=Role.Meta.app):
        if data:
            await get_queryset(request).filter(id=pk).update(**data)
        if resource_ids is not None:
            await _set_role_resources(obj, resource_ids)
    return Resp()


@router.delete("/{pk}", summary="删除角色")
async def delete_role(request: Request, pk: int) -> Resp:
    current_user: CurrentUser = request.scope["user"]
    await ensure_super_admin(current_user.id)

    obj = await get_queryset(request).get_or_none(id=pk)
    if not obj:
        raise ApiException("角色不存在", code=404)
    await UserRole.filter(role_id=pk).delete()
    await obj.resources.clear()
    return await delete_view(pk, get_queryset(request))
