from uuid import UUID
from typing import Self

from pydantic import Field, BaseModel
from tortoise.expressions import Q
from tortoise.contrib.pydantic import pydantic_model_creator

from core.types import ApiException
from enhance.epydantic import as_query, optional
from ext.file_source.types import (
    S3ExtraConfig,
    MinIOExtraConfig,
    AliyunOSSExtraConfig,
    LocalFileSourceExtraConfig,
)
from ext.ext_tortoise.enums import FileSourceTypeEnum
from ext.ext_tortoise.models.knowledge_base import FileSource


class FileSourceCreate(
    pydantic_model_creator(FileSource, name="FileSourceCreate", exclude_readonly=True),
):
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

        elif type_ == FileSourceTypeEnum.sharepoint:
            if not self.endpoint:  # type: ignore
                raise ApiException("SharePoint 需要 endpoint (site_url)")

        elif type_ == FileSourceTypeEnum.api and not self.endpoint:  # type: ignore
            raise ApiException("API 文件源需要 endpoint")

    def validate_extra_config(self) -> None:
        extra_config = self.extra_config or {}  # type: ignore
        self.validate_extra_config_by_type(self.type, extra_config)  # type: ignore


@optional()
class FileSourceUpdate(BaseModel):
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
    pydantic_model_creator(FileSource, name="FileSourceList", exclude=("secret_key",)),
):
    pass


class FileSourceDetail(FileSourceList):
    pass


@as_query
class FileSourceFilterSchema(BaseModel):
    type: FileSourceTypeEnum | None = Field(None, description="文件源类型")
    is_enabled: bool | None = Field(None, description="是否启用")
    is_default: bool | None = Field(None, description="是否默认")
    name__icontains: str | None = Field(None, description="名称包含")
    user_id: UUID | None = Field(None, description="用户ID")
