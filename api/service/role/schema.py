from pydantic import Field, BaseModel
from tortoise.contrib.pydantic import pydantic_model_creator

from enhance.epydantic import as_query, optional
from ext.ext_tortoise.models.user_center import Role


class OverridePydanticMeta:
    backward_relations: bool = False


class RoleList(
    pydantic_model_creator(  # type: ignore
        Role, name="RoleList", exclude=("resources",), meta_override=OverridePydanticMeta
    )
): ...


class RoleDetail(
    pydantic_model_creator(Role, name="RoleDetail", meta_override=OverridePydanticMeta),  # type: ignore
):
    resources: list = Field([], description="系统资源")


class RoleCreate(
    pydantic_model_creator(  # type: ignore
        Role,
        name="RoleCreate",
        include=("label", "remark"),
        # exclude_readonly=True,
    ),
):
    resources: list[int] = Field(..., description="系统资源")


@optional()
class RoleUpdate(
    pydantic_model_creator(  # type: ignore
        Role,
        name="RoleUpdate",
        include=("label", "remark"),
        # exclude_readonly=True,
    ),
):
    resources: list[int] = Field([], description="系统资源")


@as_query
class RoleFilterSchema(BaseModel):
    label: str = Field(None, description="名称")  # type: ignore
    accounts__phone: str = Field(None, max_length=11, description="电话")  # type: ignore
    accounts__username: str = Field(None, max_length=20, description="账号")  # type: ignore
