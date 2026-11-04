from typing import Any, cast
from urllib.parse import urlsplit

from pydantic import Field, BaseModel, ValidationError as PydanticValidationError, field_validator
from tortoise.contrib.pydantic import pydantic_model_creator

from core.types import ApiException
from ext.ext_tortoise.enums import EmbeddingModelTypeEnum
from ext.embedding.providers.types import OpenAIExtraConfig, AzureOpenAIExtraConfig
from ext.ext_tortoise.models.knowledge_base import EmbeddingModelConfig


class EmbeddingModelConfigCreate(
    pydantic_model_creator(EmbeddingModelConfig, name="EmbeddingModelConfigCreate", exclude_readonly=True),
):
    api_key: str | None = Field(None, max_length=1024, description="API 密钥")
    @field_validator("base_url")
    @classmethod
    def reject_embedded_base_url_credentials(cls, value: str | None) -> str | None:
        if value:
            try:
                parsed = urlsplit(value)
            except ValueError:
                raise ApiException("Embedding base_url 格式无效") from None
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ApiException("Embedding base_url 不允许嵌入凭证、query 或 fragment")
        return value

    @classmethod
    def validate_extra_config_by_type(cls, type_: EmbeddingModelTypeEnum, extra_config: dict) -> None:
        if type_ == EmbeddingModelTypeEnum.openai:
            OpenAIExtraConfig(**extra_config)

    def validate_required_fields_by_type(self) -> None:
        config = cast(Any, self)
        type_ = cast(EmbeddingModelTypeEnum, config.type)

        if type_ == EmbeddingModelTypeEnum.openai and not config.api_key:
            raise ApiException("OpenAI Embedding 需要 api_key")

    def validate_extra_config(self) -> None:
        config = cast(Any, self)
        try:
            self.validate_extra_config_by_type(
                cast(EmbeddingModelTypeEnum, config.type),
                cast(dict[str, Any], config.extra_config),
            )
        except PydanticValidationError:
            raise ApiException("Embedding 模型扩展配置格式错误", code=422) from None


class EmbeddingModelConfigList(
    pydantic_model_creator(EmbeddingModelConfig, name="EmbeddingModelConfigList", exclude=("api_key", "extra_config")),
):
    @field_validator("base_url", mode="before")
    @classmethod
    def sanitize_base_url(cls, value: str | None) -> str | None:
        if not value:
            return value
        parsed_url = urlsplit(value)
        return parsed_url._replace(
            netloc=parsed_url.netloc.rsplit("@", 1)[-1],
            query="",
            fragment="",
        ).geturl()


class EmbeddingModelConfigDetail(EmbeddingModelConfigList):
    pass


class EmbeddingModelConfigFilterSchema(BaseModel):
    type: EmbeddingModelTypeEnum | None = Field(None, description="模型类型")
    is_enabled: bool | None = Field(None, description="是否启用")
    is_default: bool | None = Field(None, description="是否默认")
    name__icontains: str | None = Field(None, description="名称包含")
