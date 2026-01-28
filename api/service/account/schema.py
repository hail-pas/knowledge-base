from pydantic import Field, BaseModel, field_validator
from tortoise.contrib.pydantic import pydantic_model_creator

from core.types import ApiException
from util.encrypt import PasswordUtil
from constant.regex import PASSWORD_REGEX
from ext.ext_tortoise import enums
from enhance.epydantic import as_query, optional
from api.service.role.schema import RoleList
from ext.ext_tortoise.models.user_center import Account


class AccountCreate(
    pydantic_model_creator(Account, name="AccountCreate", exclude=("last_login_at", "status"), exclude_readonly=True),
):
    role_id: int

    @field_validator("password")
    def encrypt_pwd(cls, v: str) -> str:
        # if not PASSWORD_REGEX.match(v):
        #     raise ApiException("密码格式错误")
        return PasswordUtil.get_password_hash(v)


@optional()
class AccountUpdate(
    pydantic_model_creator(
        Account, name="AccountUpdate", exclude=("last_login_at", "password"), exclude_readonly=True,
    ),
): ...


class AccountList(
    pydantic_model_creator(  # type: ignore
        Account,
        name="AccountList",
        exclude=(
            "role",
            "password",
        ),
    ),
):
    pass


class AccountDetail(
    pydantic_model_creator(  # type: ignore
        Account,
        name="AccountDetail",
        exclude=("password",),
    ),
):
    role: RoleList | None


@as_query
class AccountFilterSchema(BaseModel):
    id: int | None = Field(None, description="ID")
    role_id: int | None = Field(None, description="角色ID")
    username: str | None = Field(None, description="用户名", min_length=6, max_length=20)
    email: str | None = Field(None, description="邮箱", max_length=50)
    phone: str | None = Field(None, description="手机号", min_length=11, max_length=11)
    username__icontains: str | None = Field(None, description="用户名包含", max_length=20)
    email__icontains: str | None = Field(None, description="邮箱包含", max_length=50)
    phone__icontains: str | None = Field(None, description="手机号包含", max_length=11)
    is_staff: bool | None = Field(None, description="是否是员工")
    status: enums.StatusEnum | None = Field(None, description=f"状态, {enums.StatusEnum._help_text}")
