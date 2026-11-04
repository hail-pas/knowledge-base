from fastapi import APIRouter, Depends, Query, Request
from tortoise.transactions import in_transaction

from core.types import ApiException
from core.response import Resp
from service.depend import CurrentUser, api_permission_check, require_user
from service.resource.helper import resource_list_to_trees
from service.resource.schema import (
    ResourceCreateSchema,
    ResourceUpdateSchema,
    ResourceLevelTreeNode,
    ResourceLevelTreeBaseNode,
)
from ext.ext_tortoise.models.authorization import Permission, Resource, Role, UserRole

router = APIRouter(dependencies=[Depends(api_permission_check)])
my_router = APIRouter(dependencies=[Depends(require_user)])


async def _sync_resource_permissions(resource: Resource, codes: list[str]) -> None:
    unique_codes = list(dict.fromkeys(codes))
    permissions = await Permission.filter(code__in=unique_codes, is_deprecated=False)
    if len(permissions) != len(unique_codes):
        raise ApiException("包含不存在或已废弃的权限编码", code=422)
    await resource.permissions.clear()
    if permissions:
        await resource.permissions.add(*permissions)


@router.post("", summary="创建系统资源/菜单")
async def create_resource(request: Request, schema: ResourceCreateSchema) -> Resp[ResourceLevelTreeBaseNode]:
    data = schema.model_dump(exclude={"permission_codes"})
    if data.get("parent_id") is not None and not await Resource.filter(id=data["parent_id"]).exists():
        raise ApiException("父级资源不存在", code=422)
    async with in_transaction(connection_name=Resource.Meta.app):
        obj = await Resource.create(**data)
        await _sync_resource_permissions(obj, schema.permission_codes)
    await obj.fetch_related("permissions")
    permissions = await obj.permissions.all()
    response = ResourceLevelTreeBaseNode.model_validate(obj)
    response.permission_codes = [permission.code for permission in permissions]
    return Resp(data=response)


@router.get("/trees", summary="获取系统资源与菜单树")
async def resource_trees(
    request: Request,
    assignable: bool | None = Query(default=None, description="是否仅返回允许分配的资源"),
) -> Resp[list[ResourceLevelTreeNode]]:
    filters: dict = {"enabled": True}
    if assignable is not None:
        filters["assignable"] = assignable
    nodes = await Resource.filter(**filters).order_by("parent_id", "order_num")
    return Resp(data=await resource_list_to_trees(nodes))



@my_router.get("/my-trees", summary="获取当前用户有权访问的菜单资源树")
async def my_resource_trees(request: Request) -> Resp[list[ResourceLevelTreeNode]]:
    """Self-service menu endpoint; only resources granted through local roles are returned."""
    user: CurrentUser = request.scope["user"]
    role_ids = await UserRole.filter(user_id=user.id, deleted_at=0).values_list("role_id", flat=True)
    if not role_ids:
        return Resp(data=[])

    roles = await Role.filter(id__in=role_ids, deleted_at=0)
    active_role_ids = [role.id for role in roles]
    if not active_role_ids:
        return Resp(data=[])

    if any(role.is_super_admin for role in roles):
        nodes = await Resource.filter(enabled=True).order_by("parent_id", "order_num")
    else:
        nodes = await Resource.filter(
            roles__id__in=active_role_ids,
            enabled=True,
            assignable=True,
        ).distinct().order_by("parent_id", "order_num")
    return Resp(data=await resource_list_to_trees(nodes))


@router.put("/{pk}", summary="更新系统资源/菜单")
async def update_resource(request: Request, pk: int, schema: ResourceUpdateSchema) -> Resp:
    obj = await Resource.get_or_none(id=pk)
    if not obj:
        raise ApiException("资源不存在", code=404)
    data = schema.model_dump(exclude_unset=True)
    permission_codes = data.pop("permission_codes", None)
    if "parent_id" in data:
        if data["parent_id"] == pk:
            raise ApiException("资源不能将自身设置为父级", code=422)
        if data["parent_id"] is not None and not await Resource.filter(id=data["parent_id"]).exists():
            raise ApiException("父级资源不存在", code=422)
    async with in_transaction(connection_name=Resource.Meta.app):
        if data:
            await Resource.filter(id=pk).update(**data)
        if permission_codes is not None:
            await _sync_resource_permissions(obj, permission_codes)
    return Resp()


@router.delete("/{pk}", summary="删除系统资源/菜单")
async def delete_resource(request: Request, pk: int) -> Resp:
    obj = await Resource.get_or_none(id=pk)
    if not obj:
        raise ApiException("资源不存在", code=404)
    if await Resource.filter(parent_id=pk).exists():
        raise ApiException("请先删除该资源下的子资源", code=409)
    await obj.permissions.clear()
    await obj.roles.clear()
    await Resource.filter(id=pk).delete()
    return Resp()
