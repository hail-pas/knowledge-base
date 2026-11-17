from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from ext.ext_tortoise.enums import (
    SystemResourceSubTypeEnum,
    SystemResourceTypeEnum,
    TokenSceneTypeEnum,
)


class ResourceCreateSchema(BaseModel):
    code: str = Field(min_length=1, max_length=32)
    label: str = Field(min_length=1, max_length=64)
    front_route: str | None = Field(default=None, max_length=128)
    resource_type: SystemResourceTypeEnum
    sub_resource_type: SystemResourceSubTypeEnum
    order_num: int = 1
    enabled: bool = True
    assignable: bool = True
    parent_id: int | None = None
    scene: TokenSceneTypeEnum
    permission_codes: list[str] = Field(default_factory=list, description="关联的权限编码")


class ResourceUpdateSchema(BaseModel):
    code: str | None = Field(default=None, min_length=1, max_length=32)
    label: str | None = Field(default=None, min_length=1, max_length=64)
    front_route: str | None = Field(default=None, max_length=128)
    resource_type: SystemResourceTypeEnum | None = None
    sub_resource_type: SystemResourceSubTypeEnum | None = None
    order_num: int | None = None
    enabled: bool | None = None
    assignable: bool | None = None
    parent_id: int | None = None
    scene: TokenSceneTypeEnum | None = None
    permission_codes: list[str] | None = None


class ResourceLevelTreeBaseNode(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    code: str
    label: str
    front_route: str | None = None
    resource_type: SystemResourceTypeEnum
    sub_resource_type: SystemResourceSubTypeEnum
    order_num: int
    enabled: bool
    assignable: bool
    parent_id: int | None = None
    scene: TokenSceneTypeEnum
    scene_display: str
    type_display: str
    sub_type_display: str
    permission_codes: list[str] = Field(default_factory=list, description="关联的API权限编码")


class ResourceLevelTreeNode(ResourceLevelTreeBaseNode):
    children: list["ResourceLevelTreeNode"] = Field(default_factory=list)
