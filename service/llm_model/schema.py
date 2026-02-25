from typing import Self

from pydantic import BaseModel, Field
from tortoise.contrib.pydantic import pydantic_model_creator

from enhance.epydantic import as_query, optional
from core.types import ApiException
from ext.ext_tortoise.enums import LLMModelTypeEnum
from ext.ext_tortoise.models.knowledge_base import LLMModelConfig
from ext.llm.types import (
    BaseExtraConfig,
    OpenAIExtraConfig,
    AzureOpenAIExtraConfig,
    DeepSeekExtraConfig,
    AnthropicExtraConfig,
)


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
        type_ = self.type

        if type_ in (
            LLMModelTypeEnum.openai,
            LLMModelTypeEnum.azure_openai,
            LLMModelTypeEnum.deepseek,
            LLMModelTypeEnum.anthropic,
        ):
            if not self.api_key:
                raise ApiException(f"{type_.value} 需要 api_key")

    def validate_extra_config(self) -> None:
        extra_config = self.extra_config or {}
        self.validate_extra_config_by_type(self.type, extra_config)


@optional()
class LLMModelConfigUpdate(BaseModel):
    name: str | None = None
    model_name: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    max_tokens: int | None = None
    supports_chat: bool | None = None
    supports_completion: bool | None = None
    supports_streaming: bool | None = None
    supports_function_calling: bool | None = None
    supports_vision: bool | None = None
    default_temperature: float | None = None
    default_top_p: float | None = None
    max_retries: int | None = None
    timeout: int | None = None
    extra_config: dict | None = None
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
