from fastapi import APIRouter, Depends, Query, Request
from tortoise.expressions import Q

from core.types import ApiException
from core.response import Resp
from service.depend import api_permission_check
from service.permission.schema import PermissionList, PermissionDeprecationUpdate
from ext.ext_tortoise.models.authorization import Permission

router = APIRouter(dependencies=[Depends(api_permission_check)])


@router.get("", summary="权限编码列表")
async def list_permissions(
    request: Request,
    keyword: str | None = Query(default=None, description="权限编码或名称关键字"),
    is_deprecated: bool | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> Resp[list[PermissionList]]:
    queryset = Permission.all()
    if keyword:
        queryset = queryset.filter(Q(code__icontains=keyword) | Q(label__icontains=keyword))
    if is_deprecated is not None:
        queryset = queryset.filter(is_deprecated=is_deprecated)
    items = await queryset.order_by("code").offset(offset).limit(limit)
    return Resp(data=[PermissionList.model_validate(item) for item in items])


@router.put("/deprecation", summary="设置权限废弃状态")
async def update_permission_deprecation(
    request: Request,
    schema: PermissionDeprecationUpdate,
) -> Resp:
    permission = await Permission.get_or_none(code=schema.code)
    if not permission:
        raise ApiException("权限不存在", code=404)
    permission.is_deprecated = schema.is_deprecated
    await permission.save(update_fields=["is_deprecated"])
    return Resp()
