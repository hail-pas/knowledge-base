from typing import Self
from datetime import datetime

from pydantic import Field, BaseModel, field_validator, model_validator

from core.types import ApiException
from constant.regex import PASSWORD_REGEX
from ext.ext_tortoise.enums import TokenSceneTypeEnum
from api.service.account.schema import AccountList


class BaseLoginSchema(BaseModel):
    identifier: str = Field(description="唯一标识", max_length=255, min_length=6)
    scene: TokenSceneTypeEnum = Field(description=TokenSceneTypeEnum._help_text)


class CodeLoginSchema(BaseLoginSchema):
    """验证码登录: 手机短信/邮箱"""

    code: str = Field(description="验证码", max_length=6, min_length=4)


class PasswordLoginSchema(BaseLoginSchema):
    """密码登录"""

    password: str = Field(description="密码", max_length=20, min_length=8)


class SSOLoginSchema(BaseModel):
    """sso 登录"""

    code: str = Field(description="sso code", min_length=4, max_length=255)
    source: str = Field(description="来源")


class TokenResponse(BaseModel):
    token_type: str = "Bearer"
    token: str
    expired_at: datetime


class LoginResponse(TokenResponse):
    account: AccountList


class ResetPasswordIn(BaseModel):
    identifier: str = Field(description="唯一标识", max_length=255, min_length=6)
    code: str = Field(description="验证码", max_length=6, min_length=4)
    password: str = Field(description="新密码", min_length=8, max_length=20)
    repeat_password: str = Field(description="重复密码", min_length=8, max_length=20)

    @field_validator("password", mode="after")
    def validate_password(cls, v: str) -> str:
        if not PASSWORD_REGEX.match(v):
            raise ApiException("密码格式错误")
        return v

    @model_validator(mode="after")
    def check_password_match(self) -> Self:
        if self.password != self.repeat_password:
            raise ApiException(message="两次输入的密码不一致")
        return self


class ChangePasswordIn(BaseModel):
    old_password: str = Field(description="旧密码", min_length=8, max_length=20)
    new_password: str = Field(description="新密码", min_length=8, max_length=20)
    repeat_password: str = Field(description="重复密码", min_length=8, max_length=20)

    @field_validator("new_password", mode="after")
    def validate_password(cls, v: str) -> str:
        if not PASSWORD_REGEX.match(v):
            raise ApiException("密码格式错误")
        return v

    @model_validator(mode="after")
    def check_password_match(self) -> Self:
        if self.new_password != self.repeat_password:
            raise ApiException(message="两次输入的新密码不一致")
        return self
