"""
LLM Model Factory

提供 LLM 模型的创建、缓存和注册功能
"""

from typing import Dict, Type
import asyncio
from loguru import logger

from ext.llm.base import BaseLLMModel
from ext.ext_tortoise.enums import LLMModelTypeEnum
from ext.ext_tortoise.models.knowledge_base import LLMModelConfig


class LLMModelFactory:
    """LLM 模型工厂类

    负责根据数据库配置动态创建 LLM 模型实例
    """

    _providers: Dict[LLMModelTypeEnum, Type[BaseLLMModel]] = {}

    _instances: Dict[int, BaseLLMModel] = {}

    _locks: Dict[int, asyncio.Lock] = {}

    @classmethod
    def register(
        cls,
        model_type: LLMModelTypeEnum,
        provider_class: Type[BaseLLMModel],
    ) -> None:
        """
        注册新的 LLM provider

        Args:
            model_type: 模型类型标识（如 "openai"）
            provider_class: 实现 BaseLLMModel 的类

        Example:
            >>> LLMModelFactory.register(LLMModelTypeEnum.openai, OpenAILLMModel)
        """
        if model_type in cls._providers:
            import warnings

            warnings.warn(f"模型类型 {model_type.value} 已注册，将被覆盖", stacklevel=2)
        cls._providers[model_type] = provider_class
        logger.info(f"Registered LLM provider: {model_type.value} -> {provider_class.__name__}")

    @classmethod
    async def create(cls, config: LLMModelConfig, use_cache: bool = True) -> BaseLLMModel:
        """
        创建 LLM 模型实例

        Args:
            config: LLMModelConfig 数据库实例（必须已保存到数据库）
            use_cache: 是否使用缓存

        Returns:
            BaseLLMModel 实例

        Raises:
            ValueError: 配置错误或模型未启用
            ValueError: 不支持的模型类型

        Example:
            >>> config = await LLMModelConfig.filter(name="gpt-4").first()
            >>> model = await LLMModelFactory.create(config)
        """
        if not config.is_enabled:
            raise ValueError(f"LLM model config is disabled: {config.name} (id={config.id})")

        provider_cls = cls._providers.get(config.type)
        if not provider_cls:
            available_types = ", ".join([t.value for t in cls._providers.keys()])
            raise ValueError(f"Unsupported model type: {config.type.value}, available: {available_types}")

        if not use_cache or not config._saved_in_db:
            return cls._create_instance(provider_cls, config)

        if config.id in cls._instances:
            return cls._instances[config.id]

        if config.id not in cls._locks:
            cls._locks[config.id] = asyncio.Lock()

        async with cls._locks[config.id]:
            if config.id in cls._instances:
                return cls._instances[config.id]

            model = cls._create_instance(provider_cls, config)
            cls._instances[config.id] = model

            return model

    @classmethod
    def _create_instance(
        cls,
        provider_cls: Type[BaseLLMModel],
        config: LLMModelConfig,
    ) -> BaseLLMModel:
        """
        创建模型实例的内部方法

        extra_config 以 dict 形式传入，provider 的 __init__ 内部会转换成对应的 dataclass 类型

        Args:
            provider_cls: provider类
            config: 配置对象

        Returns:
            BaseLLMModel 实例
        """
        return provider_cls(
            model_name=config.model_name,
            model_type=config.type.value,
            max_tokens=config.max_tokens,
            api_key=config.api_key,
            base_url=config.base_url,
            supports_chat=config.supports_chat,
            supports_completion=config.supports_completion,
            supports_streaming=config.supports_streaming,
            supports_function_calling=config.supports_function_calling,
            supports_vision=config.supports_vision,
            default_temperature=config.default_temperature,
            default_top_p=config.default_top_p,
            max_retries=config.max_retries,
            timeout=config.timeout,
            extra_config=config.extra_config or {},
        )

    @classmethod
    def clear_cache(cls, config_id: int | None = None) -> None:
        """
        清除模型实例缓存

        Args:
            config_id: 要清除的配置 ID，如果为 None 则清除所有缓存

        Example:
            >>> LLMModelFactory.clear_cache(config_id=1)
            >>> LLMModelFactory.clear_cache()
        """
        if config_id is None:
            cls._instances.clear()
            cls._locks.clear()
        else:
            cls._instances.pop(config_id, None)
            cls._locks.pop(config_id, None)

    @classmethod
    def has_provider(cls, model_type: LLMModelTypeEnum) -> bool:
        """
        检查 provider 类型是否已注册

        Args:
            model_type: 模型类型

        Returns:
            是否已注册

        Example:
            >>> LLMModelFactory.has_provider(LLMModelTypeEnum.openai)
            True
        """
        return model_type in cls._providers

    @classmethod
    def get_registered_model_types(cls) -> list[LLMModelTypeEnum]:
        """
        获取所有已注册的模型类型

        Returns:
            已注册模型类型列表

        Example:
            >>> types = LLMModelFactory.get_registered_model_types()
            >>> [t.value for t in types]
            ['openai', 'anthropic']
        """
        return list(cls._providers.keys())

    @classmethod
    async def create_by_name(cls, name: str, use_cache: bool = True) -> BaseLLMModel:
        """
        根据配置名称创建模型实例

        Args:
            name: 配置名称
            use_cache: 是否使用缓存

        Returns:
            BaseLLMModel 实例

        Raises:
            ValueError: 配置不存在

        Example:
            >>> model = await LLMModelFactory.create_by_name("gpt-4")
        """
        config = await LLMModelConfig.filter(name=name, is_enabled=True).first()
        if not config:
            raise ValueError(f"未找到名称为 '{name}' 的已启用 LLM 配置")

        return await cls.create(config, use_cache=use_cache)

    @classmethod
    async def create_default(cls, use_cache: bool = True) -> BaseLLMModel:
        """
        创建默认的 LLM 模型实例

        Args:
            use_cache: 是否使用缓存

        Returns:
            BaseLLMModel 实例

        Raises:
            ValueError: 没有配置默认模型

        Example:
            >>> model = await LLMModelFactory.create_default()
        """
        config = await LLMModelConfig.filter(is_enabled=True, is_default=True).first()

        if not config:
            config = await LLMModelConfig.filter(is_enabled=True).first()

        if not config:
            raise ValueError("未找到可用的 LLM 模型配置")

        return await cls.create(config, use_cache=use_cache)

    @classmethod
    def get_cache_info(cls) -> Dict[str, any]:
        """
        获取缓存信息

        Returns:
            缓存信息字典

        Example:
            >>> info = LLMModelFactory.get_cache_info()
            >>> info
            {'cached_count': 1, 'cached_ids': [1], 'registered_models': ['openai', 'anthropic']}
        """
        return {
            "cached_count": len(cls._instances),
            "cached_ids": list(cls._instances.keys()),
            "registered_models": [t.value for t in cls._providers.keys()],
        }
