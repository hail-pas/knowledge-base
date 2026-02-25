from pydantic import BaseModel, Field
from tortoise.contrib.pydantic import pydantic_model_creator

from enhance.epydantic import as_query
from core.types import ApiException
from ext.ext_tortoise.enums import EmbeddingModelTypeEnum
from ext.ext_tortoise.models.knowledge_base import EmbeddingModelConfig
from ext.embedding.providers.types import OpenAIExtraConfig, AzureOpenAIExtraConfig


class EmbeddingModelConfigCreate(
    pydantic_model_creator(EmbeddingModelConfig, name="EmbeddingModelConfigCreate", exclude_readonly=True),
):
    @classmethod
    def validate_extra_config_by_type(cls, type_: EmbeddingModelTypeEnum, extra_config: dict) -> None:
        if type_ == EmbeddingModelTypeEnum.openai:
            OpenAIExtraConfig(**extra_config)

    def validate_required_fields_by_type(self) -> None:
        type_ = self.type

        if type_ == EmbeddingModelTypeEnum.openai:
            if not self.api_key:
                raise ApiException("OpenAI Embedding 需要 api_key")

    def validate_extra_config(self) -> None:
        self.validate_extra_config_by_type(self.type, self.extra_config)


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
