"""Knowledge Base local authorization models.

The user identity itself is owned by Pero. This module stores only local RBAC
objects and the mapping from Pero UUID subjects to local roles.
"""

from tortoise import fields, models

from ext.ext_tortoise.base.models import BaseModel, BigIntegerIDPrimaryKeyModel, CreateOnlyModel
from ext.ext_tortoise.enums import (
    PermissionTypeEnum,
    SystemResourceSubTypeEnum,
    SystemResourceTypeEnum,
    TokenSceneTypeEnum,
)
from ext.ext_tortoise.main import ConnectionNameEnum

_AUTH_CONNECTION = ConnectionNameEnum.authorization.value


class Role(BaseModel):
    label = fields.CharField(max_length=50, description="角色名称")
    remark = fields.CharField(max_length=200, description="备注", null=True)
    is_super_admin = fields.BooleanField(default=False, description="是否为系统超级管理员")

    resources: fields.ManyToManyRelation["Resource"]
    user_assignments: fields.ReverseRelation["UserRole"]

    class Meta:  # type: ignore
        table = "role"
        table_description = "角色"
        app = _AUTH_CONNECTION
        ordering = ["-id"]
        unique_together = (("label", "deleted_at"),)


class Permission(models.Model):
    code = fields.CharField(pk=True, max_length=256, description="权限编码")
    label = fields.CharField(max_length=128, description="权限名称")
    permission_type = fields.CharEnumField(
        max_length=16,
        enum_type=PermissionTypeEnum,
        description=f"权限类型, {PermissionTypeEnum._help_text}",
        default=PermissionTypeEnum.api,
    )
    is_deprecated = fields.BooleanField(default=False, description="是否已废弃")

    resources: fields.ManyToManyRelation["Resource"]

    class Meta:  # type: ignore
        table = "permission"
        table_description = "权限"
        app = _AUTH_CONNECTION
        ordering = ["-code"]


class Resource(BigIntegerIDPrimaryKeyModel, CreateOnlyModel):
    code = fields.CharField(max_length=32, description="资源编码", index=True)
    label = fields.CharField(max_length=64, description="资源名称", index=True)
    front_route = fields.CharField(max_length=128, description="前端路由", null=True, blank=True)
    resource_type = fields.CharEnumField(
        max_length=16,
        enum_type=SystemResourceTypeEnum,
        description=f"资源类型, {SystemResourceTypeEnum._help_text}",
    )
    sub_resource_type = fields.CharEnumField(
        max_length=16,
        enum_type=SystemResourceSubTypeEnum,
        description=f"资源子类型, {SystemResourceSubTypeEnum._help_text}",
    )
    order_num = fields.IntField(default=1, description="排列序号")
    enabled = fields.BooleanField(default=True, description="是否启用")
    assignable = fields.BooleanField(default=True, description="是否允许分配")
    parent = fields.ForeignKeyField(
        f"{_AUTH_CONNECTION}.Resource",
        related_name="children",
        null=True,
        on_delete=fields.CASCADE,
        description="父级资源",
    )
    scene = fields.CharEnumField(
        enum_type=TokenSceneTypeEnum,
        max_length=16,
        description="客户端场景",
    )
    permissions: fields.ManyToManyRelation[Permission] = fields.ManyToManyField(
        f"{_AUTH_CONNECTION}.Permission",
        related_name="resources",
    )
    roles: fields.ManyToManyRelation[Role] = fields.ManyToManyField(
        f"{_AUTH_CONNECTION}.Role",
        related_name="resources",
    )
    children: fields.ReverseRelation["Resource"]

    @property
    def type_display(self) -> str:
        return self.resource_type.label

    @property
    def sub_type_display(self) -> str:
        return self.sub_resource_type.label

    @property
    def scene_display(self) -> str:
        return self.scene.label

    class Meta:  # type: ignore
        table = "resource"
        table_description = "系统资源"
        app = _AUTH_CONNECTION
        ordering = ["order_num"]
        unique_together = (("code", "parent", "scene"),)


class UserRole(BaseModel):
    """Assign one Pero subject UUID to one or more locally managed roles."""

    user_id = fields.UUIDField(index=True, description="Pero 用户 UUID")
    role: fields.ForeignKeyRelation[Role] = fields.ForeignKeyField(
        f"{_AUTH_CONNECTION}.Role",
        related_name="user_assignments",
        on_delete=fields.CASCADE,
        description="本地角色",
    )

    class Meta:  # type: ignore
        table = "user_role"
        table_description = "外部用户角色关联"
        app = _AUTH_CONNECTION
        unique_together = (("user_id", "role", "deleted_at"),)
