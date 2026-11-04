from uuid import UUID

from fastapi import APIRouter, Depends, Request
from tortoise.transactions import in_transaction

from core.types import ApiException
from core.response import Resp
from service.depend import CurrentUser, api_permission_check, ensure_super_admin
from service.role.schema import RoleList
from service.user_role.schema import UserRoleReplace
from ext.ext_tortoise.models.authorization import Role, UserRole

router = APIRouter(dependencies=[Depends(api_permission_check)])


@router.get("/{user_id}/roles", summary="查询用户角色分配")
async def get_user_roles(request: Request, user_id: UUID) -> Resp[dict]:
    assignments = await UserRole.filter(user_id=user_id, deleted_at=0).prefetch_related("role")
    roles = [
        RoleList.model_validate(item.role).model_dump()
        for item in assignments
        if item.role.deleted_at in (None, 0)
    ]
    return Resp(data={"user_id": str(user_id), "roles": roles})


@router.put("/{user_id}/roles", summary="替换用户角色分配")
async def replace_user_roles(request: Request, user_id: UUID, schema: UserRoleReplace) -> Resp[dict]:
    current_user: CurrentUser = request.scope["user"]
    await ensure_super_admin(current_user.id)

    role_ids = list(dict.fromkeys(schema.role_ids))
    roles = await Role.filter(id__in=role_ids, deleted_at=0)
    if len(roles) != len(role_ids):
        raise ApiException("包含不存在或已删除的角色", code=422)


    async with in_transaction(connection_name=Role.Meta.app):
        await UserRole.filter(user_id=user_id).delete()
        if role_ids:
            await UserRole.bulk_create([
                UserRole(user_id=user_id, role_id=role_id)
                for role_id in role_ids
            ])

    return Resp(data={"user_id": str(user_id), "role_ids": role_ids})
