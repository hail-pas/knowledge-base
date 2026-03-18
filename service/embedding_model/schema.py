from typing import Any, cast

from pydantic import Field, BaseModel
from tortoise.contrib.pydantic import pydantic_model_creator

from core.types import ApiException
from enhance.epydantic import as_query
from ext.ext_tortoise.enums import EmbeddingModelTypeEnum
from ext.embedding.providers.types import OpenAIExtraConfig, AzureOpenAIExtraConfig
from ext.ext_tortoise.models.knowledge_base import EmbeddingModelConfig


class EmbeddingModelConfigCreate(
    pydantic_model_creator(EmbeddingModelConfig, name="EmbeddingModelConfigCreate", exclude_readonly=True),
):
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
        self.validate_extra_config_by_type(
            cast(EmbeddingModelTypeEnum, config.type),
            cast(dict[str, Any], config.extra_config),
        )


class EmbeddingModelConfigList(
    pydantic_model_creator(EmbeddingModelConfig, name="EmbeddingModelConfigList", exclude=("api_key",)),
):
    pass


class EmbeddingModelConfigDetail(EmbeddingModelConfigList):
    pass


@as_query
class EmbeddingModelConfigFilterSchema(BaseModel):
    type: EmbeddingModelTypeEnum | None = Field(None, description="模型类型")
    is_enabled: bool | None = Field(None, description="是否启用")
    is_default: bool | None = Field(None, description="是否默认")
    name__icontains: str | None = Field(None, description="名称包含")
