from pydantic import BaseModel, Field
from tortoise.contrib.pydantic import pydantic_model_creator

from ext.ext_tortoise.models.authorization import Role


class OverridePydanticMeta:
    backward_relations: bool = False


class RoleList(
    pydantic_model_creator(
        Role,
        name="RoleList",
        exclude=("resources", "user_assignments"),
        meta_override=OverridePydanticMeta,
    ),
): ...


class RoleDetail(
    pydantic_model_creator(
        Role,
        name="RoleDetail",
        exclude=("user_assignments",),
        meta_override=OverridePydanticMeta,
    ),
):
    resources: list = Field(default_factory=list, description="关联资源树")


class RoleCreate(BaseModel):
    label: str = Field(min_length=1, max_length=50, description="角色名称")
    remark: str | None = Field(default=None, max_length=200, description="备注")
    is_super_admin: bool = Field(default=False, description="是否为系统超级管理员")
    resources: list[int] = Field(default_factory=list, description="资源ID列表")


class RoleUpdate(BaseModel):
    label: str | None = Field(default=None, min_length=1, max_length=50)
    remark: str | None = Field(default=None, max_length=200)
    is_super_admin: bool | None = None
    resources: list[int] | None = None


class RoleFilterSchema(BaseModel):
    label: str | None = Field(default=None, description="角色名称")
