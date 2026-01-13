from typing import Dict, Type, Optional
import asyncio

from ext.embedding.base import EmbeddingModel
from ext.embedding.exceptions import (
    EmbeddingConfigError,
    EmbeddingModelNotFoundError,
)

from ext.ext_tortoise.enums import EmbeddingModelTypeEnum
from ext.ext_tortoise.models.knowledge_base import EmbeddingModelConfig


class EmbeddingModelFactory:
    """Embedding 模型工厂类
    """

    # 模型类型到实现类的映射
    _models: dict[EmbeddingModelTypeEnum, type[EmbeddingModel]] = {}

    # 模型实例缓存
    _instances: dict[int, EmbeddingModel] = {}

    # 锁，用于防止并发创建同一实例
    _locks: dict[int, asyncio.Lock] = {}

    @classmethod
    def register(cls, model_type: EmbeddingModelTypeEnum, model_class: type[EmbeddingModel]) -> None:
        """注册新的 embedding 模型类型

        Args:
            model_type: 模型类型标识（如 "openai", "sentence_transformers"）
            model_class: 实现 EmbeddingModel 的类
        """
        cls._models[model_type] = model_class

    @classmethod
    async def create(cls, config: EmbeddingModelConfig, use_cache: bool = True) -> EmbeddingModel:
        """创建 embedding 模型实例

        Args:
            config: EmbeddingModelConfig 数据库实例（必须已保存到数据库）
            use_cache: 是否使用缓存

        Returns:
            EmbeddingModel 实例

        Raises:
            ValueError: 不支持的模型类型或配置无效
            EmbeddingConfigError: 配置错误
        """

        # 验证配置是否启用
        if not config.is_enabled:
            raise EmbeddingConfigError(
                f"Embedding model config is disabled. "
                f"Config: {config.name} (id={config.id})"
            )

        # 获取模型类
        model_cls = cls._models.get(config.type)
        if not model_cls:
            available_types = ", ".join(cls._models.keys())
            raise EmbeddingModelNotFoundError(
                f"不支持的模型类型: {config.type.value}, "
                f"可用类型: {available_types}"
            )

        # 如果不使用缓存，直接创建新实例
        # 临时对象（未保存到数据库）不使用缓存
        if not use_cache or not config._saved_in_db:
            return model_cls(
                model_name_or_path=config.model_name_or_path,
                dimension=config.dimension,
                max_batch_size=config.max_batch_size,
                max_token_per_request=config.max_token_per_request,
                max_token_per_text=config.max_token_per_text,
                config=config.config,
            )

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
            model = model_cls(
                model_name_or_path=config.model_name_or_path,
                dimension=config.dimension,
                max_batch_size=config.max_batch_size,
                max_token_per_request=config.max_token_per_request,
                max_token_per_text=config.max_token_per_text,
                config=config.config,
            )
            cls._instances[config.id] = model

            return model

    @classmethod
    def clear_cache(cls, config_id: Optional[int] = None) -> None:
        """清除模型实例缓存

        Args:
            config_id: 要清除的配置 ID，如果为 None 则清除所有缓存
        """
        if config_id is None:
            cls._instances.clear()
            cls._locks.clear()
        else:
            cls._instances.pop(config_id, None)
            cls._locks.pop(config_id, None)

    @classmethod
    def has_model(cls, model_type: EmbeddingModelTypeEnum) -> bool:
        """检查模型类型是否已注册

        Args:
            model_type: 模型类型

        Returns:
            是否已注册
        """
        return model_type in cls._models

    @classmethod
    def get_registered_model_types(cls) -> list[EmbeddingModelTypeEnum]:
        """获取所有已注册的模型类型

        Returns:
            已注册模型类型列表
        """
        return list(cls._models.keys())


from ext.embedding.providers.openai import OpenAIEmbedding
EmbeddingModelFactory.register(EmbeddingModelTypeEnum.openai, OpenAIEmbedding)
