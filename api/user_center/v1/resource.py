from fastapi import Query, Depends, Request, APIRouter

from core.response import Resp
from service.depend import api_permission_check
from ext.ext_tortoise.curd import create_obj
from service.resource.helper import resource_list_to_trees
from service.resource.schema import (
    ResourceCreateSchema,
    ResourceLevelTreeNode,
    ResourceLevelTreeBaseNode,
)
from ext.ext_tortoise.models.user_center import Resource

router = APIRouter(dependencies=[Depends(api_permission_check)])


@router.post("", summary="创建系统资源", description="创建系统资源")
async def create_resource(request: Request, schema: ResourceCreateSchema) -> Resp[ResourceLevelTreeBaseNode]:
    obj = await create_obj(Resource, schema.model_dump(exclude_unset=True))
    return Resp(data=ResourceLevelTreeBaseNode.model_validate(obj))


@router.get(
    "/trees",
    summary="获取系统的全层级菜单",
    description="获取系统的全层级菜单",
)
async def resource_trees(
    request: Request,
    assignable: bool | None = Query(default=None, description="是否可分配, assignable为True时则为获取企业可分配权限"),
    # account: Account = Depends(token_required),
) -> Resp[list[ResourceLevelTreeNode]]:
    # if account.is_super_admin:
    #     filter_ = {
    #         "enabled": True,
    #     }
    #     if assignable is not None:
    #         filter_["assignable"] = assignable
    #     nodes = await Resource.filter(**filter_).order_by(
    #         "parent_id",
    #         "order_num",
    #     )
    #     return Resp[list](data=resource_list_to_trees(nodes))

    filter_ = {
        "enabled": True,
    }

    nodes = await Resource.filter(
        **filter_,
        # roles__id=account.role_id,
    ).order_by("parent_id", "order_num")
    return Resp(data=resource_list_to_trees(nodes))
