from typing import Self

from pydantic import Field, BaseModel
from tortoise.contrib.pydantic import pydantic_model_creator

from core.types import ApiException
from service.llm_model.types import (
    OpenAIExtraConfig,
    DeepSeekExtraConfig,
    AnthropicExtraConfig,
    AzureOpenAIExtraConfig,
)
from enhance.epydantic import as_query, optional
from ext.ext_tortoise.enums import LLMModelTypeEnum
from ext.ext_tortoise.models.knowledge_base import LLMModelConfig


class LLMModelConfigCreate(
    pydantic_model_creator(LLMModelConfig, name="LLMModelConfigCreate", exclude_readonly=True),
):
    @classmethod
    def validate_extra_config_by_type(cls, type_: LLMModelTypeEnum, extra_config: dict) -> None:
        if type_ == LLMModelTypeEnum.openai:
            OpenAIExtraConfig(**extra_config)
        elif type_ == LLMModelTypeEnum.azure_openai:
            AzureOpenAIExtraConfig(**extra_config)
        elif type_ == LLMModelTypeEnum.deepseek:
            DeepSeekExtraConfig(**extra_config)
        elif type_ == LLMModelTypeEnum.anthropic:
            AnthropicExtraConfig(**extra_config)

    def validate_required_fields_by_type(self) -> None:
        type_ = self.type  # type: ignore

        if (
            type_
            in (
                LLMModelTypeEnum.openai,
                LLMModelTypeEnum.azure_openai,
                LLMModelTypeEnum.deepseek,
                LLMModelTypeEnum.anthropic,
            )
            and not self.api_key  # type: ignore
        ):  # type: ignore
            raise ApiException(f"{type_.value} 需要 api_key")

    def validate_extra_config(self) -> None:
        extra_config = self.extra_config or {}  # type: ignore
        self.validate_extra_config_by_type(self.type, extra_config)  # type: ignore


@optional()
class LLMModelConfigUpdate(BaseModel):
    is_enabled: bool | None = None
    is_default: bool | None = None
    description: str | None = None


class LLMModelConfigList(
    pydantic_model_creator(LLMModelConfig, name="LLMModelConfigList", exclude=("api_key",)),
):
    pass


class LLMModelConfigDetail(LLMModelConfigList):
    pass


@as_query
class LLMModelConfigFilterSchema(BaseModel):
    type: LLMModelTypeEnum | None = Field(None, description="模型类型")
    is_enabled: bool | None = Field(None, description="是否启用")
    is_default: bool | None = Field(None, description="是否默认")
    name__icontains: str | None = Field(None, description="名称包含")
