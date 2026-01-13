from pydantic import Field
from tortoise.contrib.pydantic import pydantic_model_creator

from ext.ext_tortoise.models.user_center import Resource


class OverridePydanticMeta:
    backward_relations: bool = False


class ResourceCreateSchema(pydantic_model_creator(Resource, name="ResourceCreateSchema", exclude_readonly=True)): ...


class ResourceLevelTreeBaseNode(
    pydantic_model_creator(
        Resource,
        name="ResourceLevelTreeBaseNode",
        exclude=("parent", "permissions", "roles"),
        computed=("scene_display", "type_display", "sub_type_display"),
        meta_override=OverridePydanticMeta,
    ),
): ...


class ResourceLevelTreeNode(ResourceLevelTreeBaseNode):  # type: ignore
    children: list = Field([], description="子节点")  # type: ignore
