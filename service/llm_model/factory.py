import asyncio
from typing import Any

from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from ext.ext_tortoise.enums import LLMModelTypeEnum
from ext.ext_tortoise.models.knowledge_base import LLMModelConfig
from service.llm_model.types import (
    BaseExtraConfig,
    OpenAIExtraConfig,
    DeepSeekExtraConfig,
    AnthropicExtraConfig,
    AzureOpenAIExtraConfig,
)


class LLMModelFactory:
    """Build and cache `pydantic_ai` model instances from database configs."""

    _instances: dict[int, Model] = {}
    _locks: dict[int, asyncio.Lock] = {}

    @classmethod
    async def create(cls, config: LLMModelConfig, use_cache: bool = True) -> Model:
        if not config.is_enabled:
            raise ValueError(f"LLM model config is disabled: {config.name} (id={config.id})")

        if not use_cache or not config._saved_in_db:
            return cls._create_instance(config)

        if config.id in cls._instances:
            return cls._instances[config.id]

        if config.id not in cls._locks:
            cls._locks[config.id] = asyncio.Lock()

        async with cls._locks[config.id]:
            if config.id in cls._instances:
                return cls._instances[config.id]

            model = cls._create_instance(config)
            cls._instances[config.id] = model
            return model

    @classmethod
    async def create_by_id(cls, config_id: int, use_cache: bool = True) -> Model:
        config = await LLMModelConfig.filter(id=config_id, deleted_at=0, is_enabled=True).first()
        if not config:
            raise ValueError(f"未找到 ID 为 '{config_id}' 的已启用 LLM 配置")
        return await cls.create(config, use_cache=use_cache)

    @classmethod
    async def create_by_name(cls, name: str, use_cache: bool = True) -> Model:
        config = await LLMModelConfig.filter(name=name, deleted_at=0, is_enabled=True).first()
        if not config:
            raise ValueError(f"未找到名称为 '{name}' 的已启用 LLM 配置")
        return await cls.create(config, use_cache=use_cache)

    @classmethod
    async def create_default(cls, use_cache: bool = True) -> Model:
        config = await LLMModelConfig.filter(is_enabled=True, is_default=True, deleted_at=0).first()
        if not config:
            config = await LLMModelConfig.filter(is_enabled=True, deleted_at=0).first()
        if not config:
            raise ValueError("未找到可用的 LLM 模型配置")
        return await cls.create(config, use_cache=use_cache)

    @classmethod
    def clear_cache(cls, config_id: int | None = None) -> None:
        if config_id is None:
            cls._instances.clear()
            cls._locks.clear()
            return

        cls._instances.pop(config_id, None)
        cls._locks.pop(config_id, None)

    @classmethod
    def get_cache_info(cls) -> dict[str, Any]:
        return {
            "cached_count": len(cls._instances),
            "cached_ids": list(cls._instances.keys()),
        }

    @classmethod
    def get_model_settings(cls, config: LLMModelConfig) -> ModelSettings | None:
        return cls._build_settings(config)

    @classmethod
    def get_model_profile(cls, config: LLMModelConfig) -> ModelProfile:
        return ModelProfile(
            supports_tools=config.supports_tools,
            supports_image_output=config.supports_image_output,
            supports_json_schema_output=config.supports_json_schema_output,
            supports_json_object_output=config.supports_json_object_output,
            default_structured_output_mode=config.default_structured_output_mode,  # type: ignore[arg-type]
            native_output_requires_schema_in_instructions=(
                config.native_output_requires_schema_in_instructions
            ),
        )

    @classmethod
    def _create_instance(cls, config: LLMModelConfig) -> Model:
        settings = cls._build_settings(config)

        if config.type == LLMModelTypeEnum.openai:
            provider = OpenAIProvider(base_url=config.base_url, api_key=config.api_key)
            extra = OpenAIExtraConfig.from_dict(config.extra_config or {})
            return OpenAIChatModel(
                config.model_name,
                provider=provider,
                settings=cls._merge_extra_settings(settings, extra),
                profile=cls.get_model_profile(config),
            )

        if config.type == LLMModelTypeEnum.deepseek:
            provider = OpenAIProvider(base_url=config.base_url, api_key=config.api_key)
            extra = DeepSeekExtraConfig.from_dict(config.extra_config or {})
            return OpenAIChatModel(
                config.model_name,
                provider=provider,
                settings=cls._merge_extra_settings(settings, extra),
                profile=cls.get_model_profile(config),
            )

        if config.type == LLMModelTypeEnum.azure_openai:
            extra = AzureOpenAIExtraConfig.from_dict(config.extra_config or {})
            provider = AzureProvider(
                azure_endpoint=config.base_url,
                api_version=extra.api_version,  # type: ignore
                api_key=config.api_key,
            )
            return OpenAIChatModel(
                extra.deployment_name or config.model_name,  # type: ignore
                provider=provider,
                settings=cls._merge_extra_settings(settings, extra),
                profile=cls.get_model_profile(config),
            )

        if config.type == LLMModelTypeEnum.anthropic:
            extra = AnthropicExtraConfig.from_dict(config.extra_config or {})
            provider = AnthropicProvider(base_url=config.base_url, api_key=config.api_key)
            return AnthropicModel(
                config.model_name,
                provider=provider,
                settings=cls._merge_extra_settings(settings, extra),
                profile=cls.get_model_profile(config),
            )

        raise ValueError(f"Unsupported model type: {config.type.value}")

    @classmethod
    def _build_settings(cls, config: LLMModelConfig) -> ModelSettings | None:
        settings: ModelSettings = {}

        if config.max_tokens:
            settings["max_tokens"] = config.max_tokens
        if config.temperature is not None:
            settings["temperature"] = config.temperature
        if config.top_p is not None:
            settings["top_p"] = config.top_p
        if config.presence_penalty is not None:
            settings["presence_penalty"] = config.presence_penalty
        if config.frequency_penalty is not None:
            settings["frequency_penalty"] = config.frequency_penalty
        if config.seed is not None:
            settings["seed"] = config.seed
        if config.timeout:
            settings["timeout"] = float(config.timeout)
        if config.parallel_tool_calls is not None:
            settings["parallel_tool_calls"] = config.parallel_tool_calls

        return settings or None

    @classmethod
    def _merge_extra_settings(
        cls,
        settings: ModelSettings | None,
        extra: BaseExtraConfig,
    ) -> ModelSettings | None:
        merged: ModelSettings = dict(settings or {})  # type: ignore
        headers = dict(extra.headers)

        if isinstance(extra, OpenAIExtraConfig):
            if extra.organization:
                headers["OpenAI-Organization"] = extra.organization
            if extra.project:
                headers["OpenAI-Project"] = extra.project

        if headers:
            merged["extra_headers"] = headers
        if extra.extra_body:
            merged["extra_body"] = extra.extra_body

        return merged or None
