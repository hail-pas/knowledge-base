from typing import Self

from tortoise import fields, models

from config.main import local_configs
from constant.regex import EMAIL_REGEX, PHONE_REGEX_CN, ACCOUNT_USERNAME_REGEX
from ext.ext_tortoise import enums
from ext.ext_redis.keys import UserCenterKey
from ext.ext_tortoise.main import ConnectionNameEnum
from ext.ext_tortoise.base.models import (
    BaseModel,
    CreateOnlyModel,
    NotDeletedManager,
    BigIntegerIDPrimaryKeyModel,
)
from ext.ext_tortoise.base.validators import RegexValidator, MinLengthValidator

UserCenterConnection = ConnectionNameEnum.user_center.value


class Account(BaseModel):
    """用户
    Redis 实时缓存用户基本信息和权限码
    """

    username = fields.CharField(
        max_length=20,
        description="用户名",
        validators=[
            MinLengthValidator(
                min_length=4,
                error_message_template="长度需要 >= 4",
                default_ctx={"field_name": "用户名"},
            ),
            RegexValidator(
                ACCOUNT_USERNAME_REGEX.pattern,
                0,
                default_ctx={
                    "field_name": "用户名",
                },
                error_message_template="只能输入字母和数字的组合",
            ),
        ],
    )
    phone = fields.CharField(
        validators=[
            RegexValidator(
                PHONE_REGEX_CN.pattern,
                0,
                default_ctx={"field_name": "手机号"},
            ),
        ],
        max_length=11,
        description="手机号",
    )
    email = fields.CharField(
        max_length=50,
        description="邮箱",
        null=True,
        validators=[
            RegexValidator(
                EMAIL_REGEX.pattern,
                0,
                default_ctx={"field_name": "邮箱"},
            ),
        ],
    )
    # real_name = fields.CharField(max_length=64, description="姓名")
    is_staff = fields.BooleanField(default=False, description="是否是后台管理员")
    is_super_admin = fields.BooleanField(
        default=False,
        description="是否是后台超级管理员",
    )
    status = fields.CharEnumField(
        max_length=16,
        enum_type=enums.StatusEnum,
        description="状态",
        default=enums.StatusEnum.enable,
    )
    last_login_at = fields.DatetimeField(null=True, description="最近一次登录时间")
    remark = fields.CharField(max_length=200, description="备注", default="")
    # 密码加密存储
    password = fields.CharField(max_length=255, description="密码")
    role: fields.ForeignKeyRelation["Role"] = fields.ForeignKeyField(
        f"{UserCenterConnection}.Role",
        related_name="accounts",
        description="角色",
    )

    def __str__(self) -> str:
        return self.username

    def status_display(self) -> str:
        """状态显示"""
        return enums.StatusEnum._dict.get(self.status, "")

    def days_from_last_login(self) -> int | None:
        """距上一次登录天数

        Returns:
            int | None: 从未登录的情况为None
        """
        if not self.last_login_at:
            return None
        return (local_configs.extensions.rdb_user_center.datetime_now - self.last_login_at).days

    async def has_permission(
        self,
        apis: list[str],
        conn_name: ConnectionNameEnum = ConnectionNameEnum.user_center,
    ) -> bool:
        # OR
        async with local_configs.extensions.redis.instance as r:
            result = await r.smismember(
                name=UserCenterKey.AccountApiPermissionSet.format(uuid=str(self.id)),  # type: ignore
                values=apis,
            )
        if 1 in result:
            return True
        return False

    async def update_cache_permissions(
        self,
        conn_name: ConnectionNameEnum = ConnectionNameEnum.user_center,
    ) -> None:
        perms = await self.get_permission_codes()
        async with local_configs.extensions.redis.instance as r:
            async with r.pipeline() as pipe:
                pipe.delete(UserCenterKey.AccountApiPermissionSet.format(uuid=str(self.id)))
                if perms:
                    # 刷新接口权限
                    pipe.sadd(
                        UserCenterKey.AccountApiPermissionSet.format(uuid=str(self.id)),  # type: ignore
                        *perms,  # type: ignore
                    )
                await pipe.execute()

    async def get_permission_codes(self) -> list[str]:
        """获取用户的全部permission codes

        Args:
            account (Account): _description_

        Returns:
            list[str]: _description_
        """
        permission_codes: list[str] = []
        if self.is_super_admin:
            return ["*"]
        resource_ids = await self.get_account_resource_ids()  # type: ignore
        if not resource_ids:
            return permission_codes
        return await Permission.filter(  # type: ignore
            resources__id__in=resource_ids,
        ).values_list("code", flat=True)

    async def get_account_resource_ids(
        self,
    ) -> list[str]:
        _args = []
        _kwargs = {
            "enabled": True,
            "assignable": True,
        }
        if self.is_staff:
            _kwargs = {
                "enabled": True,
            }
        if self.is_super_admin:
            return await Resource.filter(*_args, **_kwargs).values_list("id", flat=True)  # type: ignore
        _kwargs["roles__id"] = self.role_id  # type: ignore
        return await Resource.filter(*_args, **_kwargs).values_list("id", flat=True)  # type: ignore

    @classmethod
    async def get_by_identifier(cls, identifier: str) -> Self | None:
        filter_ = {}
        if PHONE_REGEX_CN.match(identifier):
            filter_["phone"] = identifier
        elif EMAIL_REGEX.match(identifier):
            filter_["email"] = identifier
        elif ACCOUNT_USERNAME_REGEX.match(identifier):
            filter_["username"] = identifier
        else:
            return None

        return await cls.filter(**filter_, deleted_at=0).first()

    class Meta: # type: ignore
        table_description = "用户"
        app = UserCenterConnection
        ordering = ["-id"]
        unique_together = (
            ("username", "deleted_at"),
            ("phone", "deleted_at"),
            ("email", "deleted_at"),
        )
        manager = NotDeletedManager()
        unique_error_messages = {
            "account.uid_account_username_4f1849": "用户名已存在",
            "account.uid_account_phone_9b9e7e": "手机号已存在",
            "account.uid_account_email_a28fc7": "邮箱地址已存在",
        }


class Role(BaseModel):
    label = fields.CharField(max_length=50, description="名称")
    remark = fields.CharField(max_length=200, description="备注", null=True)

    # reversed relations
    accounts: fields.ReverseRelation[Account]
    resources: fields.ManyToManyRelation["Resource"]

    class Meta: # type: ignore
        table_description = "角色"
        ordering = ["-id"]
        app = UserCenterConnection
        unique_together = (("label", "deleted_at"),)


class Permission(models.Model):
    code = fields.CharField(pk=True, max_length=256, description="权限码")
    label = fields.CharField(max_length=128, description="权限名称")
    permission_type = fields.CharEnumField(
        max_length=16,
        enum_type=enums.PermissionTypeEnum,
        description=f"权限类型, {enums.PermissionTypeEnum._help_text}",
        default=enums.PermissionTypeEnum.api,
    )
    is_deprecated = fields.BooleanField(default=False, description="是否废弃")
    # reversed relations
    resources: fields.ManyToManyRelation["Resource"]

    class Meta: # type: ignore
        table_description = "权限"
        ordering = ["-code"]
        app = UserCenterConnection


class Resource(BigIntegerIDPrimaryKeyModel, CreateOnlyModel):
    code = fields.CharField(
        max_length=32,
        description="资源编码{parent}:{current}",
        index=True,
    )
    # icon_path = FileField(
    #     max_length=256,
    #     description="图标",
    #     null=True,
    #     storage=local_configs.extensions.file_source.instance,
    # )
    label = fields.CharField(max_length=64, description="资源名称", index=True)
    front_route = fields.CharField(
        max_length=128,
        description="前端路由",
        null=True,
        blank=True,
    )
    resource_type = fields.CharEnumField(
        max_length=16,
        enum_type=enums.SystemResourceTypeEnum,
        description=f"资源类型, {enums.SystemResourceTypeEnum._help_text}",
    )
    sub_resource_type = fields.CharEnumField(
        max_length=16,
        enum_type=enums.SystemResourceSubTypeEnum,
        description=f"资源类型, {enums.SystemResourceSubTypeEnum._help_text}",
    )
    order_num = fields.IntField(default=1, description="排列序号")
    enabled = fields.BooleanField(default=True, description="是否可用")
    assignable = fields.BooleanField(default=True, description="是否可分配")
    parent = fields.ForeignKeyField(  # type: ignore
        model_name=f"{UserCenterConnection}.Resource",
        related_name="children",
        null=True,
        description="父级",
    )
    scene = fields.CharEnumField(enum_type=enums.TokenSceneTypeEnum, max_length=16, description="场景")
    permissions: fields.ManyToManyRelation[Permission] = fields.ManyToManyField(
        model_name=f"{UserCenterConnection}.Permission",
        related_name="resources",
    )
    roles: fields.ManyToManyRelation["Role"] = fields.ManyToManyField(
        model_name=f"{UserCenterConnection}.Role",
        related_name="resources",
    )

    # reversed relations
    children: fields.ReverseRelation["Resource"]

    def type_display(self) -> str:
        """类型显示"""
        return self.resource_type.label

    def sub_type_display(self) -> str | None:
        """子类型显示"""
        return self.sub_resource_type.label

    def scene_display(self) -> str:
        return self.scene.label

    class Meta: # type: ignore
        table_description = "系统资源"
        ordering = ["order_num"]
        unique_together = (("code", "parent", "scene"),)
        app = UserCenterConnection


class Config(models.Model):
    key = fields.CharField(max_length=128, description="配置项key", unique=True)
    value = fields.JSONField(default=dict, description="配置项值")
    description = fields.CharField(max_length=255, description="配置项描述")

    class Meta: # type: ignore
        table_description = "系统配置"
        ordering = ["-id"]
        app = UserCenterConnection
