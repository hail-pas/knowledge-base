from urllib.parse import urlsplit

from pydantic import Field, BaseModel, ValidationError as PydanticValidationError, field_validator
from tortoise.expressions import Q
from tortoise.contrib.pydantic import pydantic_model_creator

from core.types import ApiException
from ext.file_source.types import (
    S3ExtraConfig,
    MinIOExtraConfig,
    AliyunOSSExtraConfig,
    LocalFileSourceExtraConfig,
)
from ext.ext_tortoise.enums import FileSourceTypeEnum
from ext.ext_tortoise.models.knowledge_base import FileSource
from service.file_source.security import validate_allowed_prefix_template


class FileSourceCreate(
    pydantic_model_creator(FileSource, name="FileSourceCreate", exclude_readonly=True, exclude=("user_id",)),
):
    access_key: str | None = Field(None, max_length=1024, description="访问密钥")
    secret_key: str | None = Field(None, max_length=1024, description="密钥")

    allowed_prefix_template: str | None = Field(None, description="公共文件源对象路径隔离模板")

    @field_validator("allowed_prefix_template")
    @classmethod
    def validate_allowed_prefix(cls, value: str | None) -> str | None:
        return validate_allowed_prefix_template(value)

    @field_validator("endpoint")
    @classmethod
    def reject_embedded_endpoint_credentials(cls, value: str | None) -> str | None:
        if value:
            try:
                parsed = urlsplit(value)
            except ValueError:
                raise ApiException("文件源 endpoint 格式无效") from None
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ApiException("文件源 endpoint 不允许嵌入凭证、query 或 fragment")
        return value

    @classmethod
    def validate_extra_config_by_type(cls, type_: FileSourceTypeEnum, extra_config: dict) -> None:
        if type_ == FileSourceTypeEnum.local_file:
            LocalFileSourceExtraConfig(**extra_config)
        elif type_ == FileSourceTypeEnum.s3:
            S3ExtraConfig(**extra_config)
        elif type_ == FileSourceTypeEnum.minio:
            MinIOExtraConfig(**extra_config)
        elif type_ == FileSourceTypeEnum.aliyun_oss:
            AliyunOSSExtraConfig(**extra_config)
        elif type_ == FileSourceTypeEnum.sharepoint or type_ == FileSourceTypeEnum.api:
            pass

    def validate_required_fields_by_type(self) -> None:
        type_ = self.type  # type: ignore

        if type_ == FileSourceTypeEnum.local_file:
            if not self.storage_location:  # type: ignore
                raise ApiException("本地文件需要 storage_location")

        elif type_ in (
            FileSourceTypeEnum.s3,
            FileSourceTypeEnum.minio,
            FileSourceTypeEnum.aliyun_oss,
        ):
            required_fields = ["access_key", "secret_key", "endpoint", "storage_location"]
            for field_name in required_fields:
                if not getattr(self, field_name, None):
                    raise ApiException(f"{type_.value} 需要 {field_name}")

        elif type_ in (FileSourceTypeEnum.sharepoint, FileSourceTypeEnum.api):
            raise ApiException(f"{type_.value} 文件源尚未实现，当前版本不支持")

    def validate_extra_config(self) -> None:
        extra_config = self.extra_config or {}  # type: ignore
        try:
            self.validate_extra_config_by_type(self.type, extra_config)  # type: ignore
        except PydanticValidationError:
            raise ApiException("文件源扩展配置格式错误", code=422) from None


class FileSourceUpdate(BaseModel):
    allowed_prefix_template: str | None = Field(None, max_length=500)

    @field_validator("allowed_prefix_template")
    @classmethod
    def validate_allowed_prefix(cls, value: str | None) -> str | None:
        return validate_allowed_prefix_template(value)

    # name: str | None = None
    # access_key: str | None = None
    # secret_key: str | None = None
    # endpoint: str | None = None
    # region: str | None = None
    # use_ssl: bool | None = None
    # verify_ssl: bool | None = None
    # timeout: int | None = None
    # max_retries: int | None = None
    # concurrent_limit: int | None = None
    # max_connections: int | None = None
    is_enabled: bool | None = None
    is_default: bool | None = None
    description: str | None = None


class FileSourceList(
    pydantic_model_creator(
        FileSource,
        name="FileSourceList",
        exclude=("access_key", "secret_key", "extra_config", "storage_location", "user_id"),
    ),
):
    @field_validator("endpoint", mode="before")
    @classmethod
    def sanitize_endpoint(cls, value: str | None) -> str | None:
        if not value:
            return value
        parsed = urlsplit(value)
        if parsed.netloc:
            return parsed._replace(
                netloc=parsed.netloc.rsplit("@", 1)[-1],
                query="",
                fragment="",
            ).geturl()
        return value.split("?", 1)[0].split("#", 1)[0]


class FileSourceDetail(FileSourceList):
    pass


class FileSourceFilterSchema(BaseModel):
    type: FileSourceTypeEnum | None = Field(None, description="文件源类型")
    is_enabled: bool | None = Field(None, description="是否启用")
    is_default: bool | None = Field(None, description="是否默认")
    name__icontains: str | None = Field(None, description="名称包含")
