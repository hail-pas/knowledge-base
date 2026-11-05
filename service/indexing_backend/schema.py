from typing import Any, cast
from urllib.parse import urlsplit

from pydantic import Field, BaseModel, ValidationError as PydanticValidationError, field_validator
from tortoise.contrib.pydantic import pydantic_model_creator

from core.types import ApiException
from ext.ext_tortoise.enums import IndexingBackendTypeEnum
from ext.indexing.providers.types import MilvusConfig, ElasticsearchConfig
from ext.indexing.providers.postgresql import PostgreSQLConfig
from ext.ext_tortoise.models.knowledge_base import IndexingBackendConfig


class IndexingBackendConfigCreate(
    pydantic_model_creator(IndexingBackendConfig, name="IndexingBackendConfigCreate", exclude_readonly=True),
):
    password: str | None = Field(None, max_length=1024, description="连接密码")

    @field_validator("host")
    @classmethod
    def reject_embedded_host_credentials(cls, value: str | None) -> str | None:
        if value:
            try:
                parsed = urlsplit(value)
            except ValueError:
                raise ApiException("索引后端 host 格式无效") from None
            if parsed.username or parsed.password or parsed.query or parsed.fragment or "@" in value:
                raise ApiException("索引后端 host 不允许嵌入凭证、query 或 fragment")
        return value

    @classmethod
    def validate_extra_config_by_type(cls, type_: IndexingBackendTypeEnum, extra_config: dict) -> None:
        if type_ == IndexingBackendTypeEnum.elasticsearch:
            ElasticsearchConfig(**extra_config)
        elif type_ == IndexingBackendTypeEnum.milvus:
            MilvusConfig(**extra_config)
        elif type_ == IndexingBackendTypeEnum.postgresql:
            PostgreSQLConfig(**extra_config)

    def validate_required_fields_by_type(self) -> None:
        config = cast(Any, self)
        type_ = cast(IndexingBackendTypeEnum, config.type)

        if type_ == IndexingBackendTypeEnum.postgresql and not config.host:
            raise ApiException("PostgreSQL 需要 host")

        if type_ == IndexingBackendTypeEnum.elasticsearch and (not config.host or not config.port):
            raise ApiException("Elasticsearch 需要 host 和 port")

        if type_ == IndexingBackendTypeEnum.milvus and (not config.host or not config.port):
            raise ApiException("Milvus 需要 host 和 port")

    def validate_extra_config(self) -> None:
        config = cast(Any, self)
        try:
            self.validate_extra_config_by_type(
                cast(IndexingBackendTypeEnum, config.type),
                cast(dict[str, Any], config.extra_config),
            )
        except PydanticValidationError:
            raise ApiException("索引后端扩展配置格式错误", code=422) from None


class IndexingBackendConfigList(
    pydantic_model_creator(IndexingBackendConfig, name="IndexingBackendConfigList", exclude=("password", "extra_config")),
):
    @field_validator("host", mode="before")
    @classmethod
    def sanitize_host(cls, value: str | None) -> str | None:
        if not value or "://" not in value:
            return value
        parsed = urlsplit(value)
        return parsed._replace(
            netloc=parsed.netloc.rsplit("@", 1)[-1],
            query="",
            fragment="",
        ).geturl()


class IndexingBackendConfigDetail(IndexingBackendConfigList):
    pass


class IndexingBackendConfigFilterSchema(BaseModel):
    type: IndexingBackendTypeEnum | None = Field(None, description="后端类型")
    is_enabled: bool | None = Field(None, description="是否启用")
    is_default: bool | None = Field(None, description="是否默认")
    name__icontains: str | None = Field(None, description="名称包含")
