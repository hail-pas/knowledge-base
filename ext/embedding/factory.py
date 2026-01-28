"""
Embedding Model Factory

提供embedding模型的创建、缓存和注册功能
"""

from typing import Dict, Type, Any
import asyncio
from loguru import logger

from ext.embedding.base import BaseEmbeddingModel
from ext.ext_tortoise.enums import EmbeddingModelTypeEnum
from ext.ext_tortoise.models.knowledge_base import EmbeddingModelConfig


class EmbeddingModelFactory:
    """Embedding 模型工厂类

    负责根据数据库配置动态创建 embedding 模型实例
    """

    # 模型类型到provider类的映射
    _providers: dict[EmbeddingModelTypeEnum, type[BaseEmbeddingModel]] = {}

    # 实例缓存
    _instances: dict[int, BaseEmbeddingModel] = {}

    # 锁，用于防止并发创建同一实例
    _locks: dict[int, asyncio.Lock] = {}

    @classmethod
    def register(
        cls,
        model_type: EmbeddingModelTypeEnum,
        provider_class: type[BaseEmbeddingModel],
    ) -> None:
        """
        注册新的 embedding provider

        Args:
            model_type: 模型类型标识（如 "openai"）
            provider_class: 实现 BaseEmbeddingModel 的类

        Example:
            >>> EmbeddingModelFactory.register(EmbeddingModelTypeEnum.openai, OpenAIEmbeddingModel)
        """
        if model_type in cls._providers:
            import warnings

            warnings.warn(f"模型类型 {model_type.value} 已注册，将被覆盖", stacklevel=2)
        cls._providers[model_type] = provider_class
        logger.info(f"Registered embedding provider: {model_type.value} -> {provider_class.__name__}")

    @classmethod
    async def create(cls, config: EmbeddingModelConfig, use_cache: bool = True) -> BaseEmbeddingModel:
        """
        创建 embedding 模型实例

        Args:
            config: EmbeddingModelConfig 数据库实例（必须已保存到数据库）
            use_cache: 是否使用缓存

        Returns:
            BaseEmbeddingModel 实例

        Raises:
            ValueError: 配置错误或模型未启用
            ValueError: 不支持的模型类型

        Example:
            >>> config = await EmbeddingModelConfig.filter(name="openai-embedding-small").first()
            >>> model = await EmbeddingModelFactory.create(config)
        """
        # 验证配置是否启用
        if not config.is_enabled:
            raise ValueError(f"Embedding model config is disabled: {config.name} (id={config.id})")

        # 获取provider类
        provider_cls = cls._providers.get(config.type)
        if not provider_cls:
            available_types = ", ".join([t.value for t in cls._providers.keys()])
            raise ValueError(f"Unsupported model type: {config.type.value}, available: {available_types}")

        # 如果不使用缓存，直接创建新实例
        # 临时对象（未保存到数据库）不使用缓存
        if not use_cache or not config._saved_in_db:
            return cls._create_instance(provider_cls, config)

        # 检查缓存
        if config.id in cls._instances:
            return cls._instances[config.id]

        # 获取或创建锁
        if config.id not in cls._locks:
            cls._locks[config.id] = asyncio.Lock()

        # 使用锁防止并发创建
        async with cls._locks[config.id]:
            # 再次检查缓存（可能在等待锁时已被其他协程创建）
            if config.id in cls._instances:
                return cls._instances[config.id]

            # 创建新实例并缓存
            model = cls._create_instance(provider_cls, config)
            cls._instances[config.id] = model

            return model

    @classmethod
    def _create_instance(
        cls,
        provider_cls: type[BaseEmbeddingModel],
        config: EmbeddingModelConfig,
    ) -> BaseEmbeddingModel:
        """
        创建模型实例的内部方法

        extra_config 以 dict 形式传入，provider 的 __init__ 内部会转换成对应的 dataclass 类型

        Args:
            provider_cls: provider类
            config: 配置对象

        Returns:
            BaseEmbeddingModel 实例
        """
        return provider_cls(
            model_name=config.model_name,
            model_type=config.type.value,
            dimension=config.dimension,
            api_key=config.api_key,
            base_url=config.base_url,
            max_chunk_length=config.max_chunk_length,
            batch_size=config.batch_size,
            max_retries=config.max_retries,
            timeout=config.timeout,
            rate_limit=config.rate_limit,
            extra_config=config.extra_config or {},  # 传入 dict
        )

    @classmethod
    def clear_cache(cls, config_id: int | None = None) -> None:
        """
        清除模型实例缓存

        Args:
            config_id: 要清除的配置 ID，如果为 None 则清除所有缓存

        Example:
            >>> # 清除单个配置的缓存
            >>> EmbeddingModelFactory.clear_cache(config_id=1)
            >>> # 清除所有缓存
            >>> EmbeddingModelFactory.clear_cache()
        """
        if config_id is None:
            cls._instances.clear()
            cls._locks.clear()
        else:
            cls._instances.pop(config_id, None)
            cls._locks.pop(config_id, None)

    @classmethod
    def has_provider(cls, model_type: EmbeddingModelTypeEnum) -> bool:
        """
        检查provider类型是否已注册

        Args:
            model_type: 模型类型

        Returns:
            是否已注册

        Example:
            >>> EmbeddingModelFactory.has_provider(EmbeddingModelTypeEnum.openai)
            True
        """
        return model_type in cls._providers

    @classmethod
    def get_registered_model_types(cls) -> list[EmbeddingModelTypeEnum]:
        """
        获取所有已注册的模型类型

        Returns:
            已注册模型类型列表

        Example:
            >>> types = EmbeddingModelFactory.get_registered_model_types()
            >>> [t.value for t in types]
            ['openai']
        """
        return list(cls._providers.keys())

    @classmethod
    async def create_by_name(cls, name: str, use_cache: bool = True) -> BaseEmbeddingModel:
        """
        根据配置名称创建模型实例

        Args:
            name: 配置名称
            use_cache: 是否使用缓存

        Returns:
            BaseEmbeddingModel 实例

        Raises:
            ValueError: 配置不存在

        Example:
            >>> model = await EmbeddingModelFactory.create_by_name("openai-embedding-small")
        """
        config = await EmbeddingModelConfig.filter(name=name, is_enabled=True).first()
        if not config:
            raise ValueError(f"未找到名称为 '{name}' 的已启用 embedding 配置")

        return await cls.create(config, use_cache=use_cache)

    @classmethod
    async def create_default(cls, use_cache: bool = True) -> BaseEmbeddingModel:
        """
        创建默认的 embedding 模型实例

        Args:
            use_cache: 是否使用缓存

        Returns:
            BaseEmbeddingModel 实例

        Raises:
            ValueError: 没有配置默认模型

        Example:
            >>> model = await EmbeddingModelFactory.create_default()
        """
        config = await EmbeddingModelConfig.filter(is_enabled=True, is_default=True).first()

        if not config:
            # 如果没有默认配置，尝试获取第一个启用的配置
            config = await EmbeddingModelConfig.filter(is_enabled=True).first()

        if not config:
            raise ValueError("未找到可用的 embedding 模型配置")

        return await cls.create(config, use_cache=use_cache)

    @classmethod
    def get_cache_info(cls) -> dict[str, Any]:
        """
        获取缓存信息

        Returns:
            缓存信息字典

        Example:
            >>> info = EmbeddingModelFactory.get_cache_info()
            >>> info
            {'cached_count': 1, 'cached_ids': [1], 'registered_models': ['openai']}
        """
        return {
            "cached_count": len(cls._instances),
            "cached_ids": list(cls._instances.keys()),
            "registered_models": [t.value for t in cls._providers.keys()],
        }
