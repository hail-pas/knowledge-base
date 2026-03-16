import asyncio
import inspect
from importlib import import_module
from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel

from ext.ext_tortoise.enums import LLMModelTypeEnum
from ext.ext_tortoise.models.knowledge_base import LLMModelConfig
from service.llm_model_langchain.model import CapabilityAwareChatModel, LangChainModelCapabilities
from service.llm_model.types import (
    OpenAIExtraConfig,
    DeepSeekExtraConfig,
    AnthropicExtraConfig,
    AzureOpenAIExtraConfig,
)


class LangChainLLMModelFactory:
    """Build and cache LangChain chat models with minimal provider dependencies."""

    _instances: dict[int, CapabilityAwareChatModel] = {}
    _locks: dict[int, asyncio.Lock] = {}

    @classmethod
    async def create(cls, config: LLMModelConfig, use_cache: bool = True) -> CapabilityAwareChatModel:
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
    async def create_by_id(cls, config_id: int, use_cache: bool = True) -> CapabilityAwareChatModel:
        config = await LLMModelConfig.filter(id=config_id, deleted_at=0, is_enabled=True).first()
        if not config:
            raise ValueError(f"未找到 ID 为 '{config_id}' 的已启用 LLM 配置")
        return await cls.create(config, use_cache=use_cache)

    @classmethod
    async def create_by_name(cls, name: str, use_cache: bool = True) -> CapabilityAwareChatModel:
        config = await LLMModelConfig.filter(name=name, deleted_at=0, is_enabled=True).first()
        if not config:
            raise ValueError(f"未找到名称为 '{name}' 的已启用 LLM 配置")
        return await cls.create(config, use_cache=use_cache)

    @classmethod
    async def create_default(cls, use_cache: bool = True) -> CapabilityAwareChatModel:
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
    def get_model_kwargs(cls, config: LLMModelConfig) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "timeout": float(config.timeout),
        }

        if config.top_p is not None:
            kwargs["top_p"] = config.top_p
        if config.presence_penalty is not None:
            kwargs["presence_penalty"] = config.presence_penalty
        if config.frequency_penalty is not None:
            kwargs["frequency_penalty"] = config.frequency_penalty
        if config.seed is not None:
            kwargs["seed"] = config.seed
        if config.parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = config.parallel_tool_calls

        return kwargs

    @classmethod
    def get_declared_capabilities(cls, config: LLMModelConfig) -> LangChainModelCapabilities:
        return LangChainModelCapabilities(
            supports_tools=config.supports_tools,
            supports_image_output=config.supports_image_output,
            supports_json_schema_output=config.supports_json_schema_output,
            supports_json_object_output=config.supports_json_object_output,
            default_structured_output_mode=config.default_structured_output_mode,
            native_output_requires_schema_in_instructions=(
                config.native_output_requires_schema_in_instructions
            ),
        )

    @classmethod
    def _create_instance(cls, config: LLMModelConfig) -> CapabilityAwareChatModel:
        model: BaseChatModel

        if config.type == LLMModelTypeEnum.openai:
            extra = OpenAIExtraConfig.from_dict(config.extra_config or {})
            model_cls = cls._load_class("langchain_openai", "ChatOpenAI")
            model = cls._instantiate_model(
                model_cls,
                {
                    "model": config.model_name,
                    "api_key": config.api_key,
                    "base_url": config.base_url,
                    "organization": extra.organization,  # type: ignore
                    "default_headers": cls._build_headers(extra),
                    "extra_body": cls._build_extra_body(extra),
                    **cls.get_model_kwargs(config),
                },
            )
            return cls._wrap_model(model, config)

        if config.type == LLMModelTypeEnum.deepseek:
            extra = DeepSeekExtraConfig.from_dict(config.extra_config or {})
            model_cls = cls._load_class("langchain_openai", "ChatOpenAI")
            model = cls._instantiate_model(
                model_cls,
                {
                    "model": config.model_name,
                    "api_key": config.api_key,
                    "base_url": config.base_url,
                    "default_headers": cls._build_headers(extra),
                    "extra_body": cls._build_extra_body(extra),
                    **cls.get_model_kwargs(config),
                },
            )
            return cls._wrap_model(model, config)

        if config.type == LLMModelTypeEnum.azure_openai:
            extra = AzureOpenAIExtraConfig.from_dict(config.extra_config or {})
            model_cls = cls._load_class("langchain_openai", "AzureChatOpenAI")
            model = cls._instantiate_model(
                model_cls,
                {
                    "model": config.model_name,
                    "azure_deployment": extra.deployment_name or config.model_name,  # type: ignore
                    "api_key": config.api_key,
                    "azure_endpoint": config.base_url,
                    "api_version": extra.api_version,  # type: ignore
                    "default_headers": cls._build_headers(extra),
                    "extra_body": cls._build_extra_body(extra),
                    **cls.get_model_kwargs(config),
                },
            )
            return cls._wrap_model(model, config)

        if config.type == LLMModelTypeEnum.anthropic:
            extra = AnthropicExtraConfig.from_dict(config.extra_config or {})
            model_cls = cls._load_class("langchain_anthropic", "ChatAnthropic")
            model = cls._instantiate_model(
                model_cls,
                {
                    "model_name": config.model_name,
                    "api_key": config.api_key,
                    "base_url": config.base_url,
                    "default_headers": cls._build_headers(extra),
                    "model_kwargs": cls._build_extra_body(extra),
                    **cls.get_model_kwargs(config),
                },
            )
            return cls._wrap_model(model, config)

        raise ValueError(f"Unsupported model type: {config.type.value}")

    @classmethod
    def _build_headers(cls, extra: Any) -> dict[str, str]:
        headers = dict(extra.headers)

        if isinstance(extra, OpenAIExtraConfig):
            if extra.organization:
                headers["OpenAI-Organization"] = extra.organization
            if extra.project:
                headers["OpenAI-Project"] = extra.project

        return headers

    @classmethod
    def _build_extra_body(cls, extra: Any) -> dict[str, Any]:
        return dict(extra.extra_body)

    @classmethod
    def _instantiate_model(cls, model_cls: type[BaseChatModel], kwargs: dict[str, Any]) -> BaseChatModel:
        signature = inspect.signature(model_cls)
        has_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

        filtered_kwargs = {
            key: value
            for key, value in kwargs.items()
            if value is not None and value != {} and (has_var_kwargs or key in signature.parameters)
        }
        return model_cls(**filtered_kwargs)

    @classmethod
    def _wrap_model(
        cls,
        model: BaseChatModel,
        config: LLMModelConfig,
    ) -> CapabilityAwareChatModel:
        return CapabilityAwareChatModel(
            wrapped=model,
            capabilities=cls.get_declared_capabilities(config),
        )

    @classmethod
    def _load_class(cls, module_name: str, class_name: str) -> type[BaseChatModel]:
        try:
            module = import_module(module_name)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"Missing dependency `{module_name}`. "
                f"Install minimal LangChain provider packages such as "
                f"`langchain-openai` and `langchain-anthropic`.",
            ) from exc

        try:
            model_cls = getattr(module, class_name)
        except AttributeError as exc:
            raise ImportError(f"`{class_name}` not found in `{module_name}`") from exc

        return cast(type[BaseChatModel], model_cls)
