"""Indexing Provider 工厂（管理连接池缓存）"""

import asyncio
import json
from typing import ClassVar

from ext.indexing.base import BaseProvider, BaseIndexModel
from ext.ext_tortoise.models.knowledge_base import (
    EmbeddingModelConfig,
    IndexingBackendConfig,
    IndexingBackendTypeEnum,
)


class IndexingProviderFactory:
    """Indexing Provider 工厂"""

    _providers: dict[str, type[BaseProvider]] = {}
    _instances: dict[int, tuple[str, BaseProvider]] = {}
    _locks: dict[int, asyncio.Lock] = {}

    @staticmethod
    def _config_signature(config: IndexingBackendConfig) -> str:
        """连接参数变化后必须重建连接，避免使用旧密码/旧地址。"""
        return json.dumps({
            "type": str(config.type), "host": config.host, "port": config.port,
            "username": config.username, "password": config.password,
            "use_ssl": config.use_ssl, "verify_ssl": config.verify_ssl,
            "timeout": config.timeout, "max_retries": config.max_retries,
            "max_connections": config.max_connections, "extra_config": config.extra_config,
        }, sort_keys=True, default=str)


    @classmethod
    def register(cls, backend_type: str, provider_class: type[BaseProvider]) -> None:
        """注册 provider"""
        cls._providers[backend_type] = provider_class

    @classmethod
    async def create(cls, config: IndexingBackendConfig, use_cache: bool = True) -> BaseProvider:
        """按配置 ID 缓存连接；配置变化自动刷新，失败时保留旧实例。"""
        provider_class = cls._providers.get(str(config.type))
        if provider_class is None:
            raise RuntimeError(f"Provider not found: {config.type}")
        if not use_cache:
            provider = provider_class(config)
            await provider.connect()
            return provider

        key = config.id
        signature = cls._config_signature(config)
        lock = cls._locks.setdefault(key, asyncio.Lock())
        async with lock:
            cached = cls._instances.get(key)
            if cached is not None and cached[0] == signature:
                return cached[1]
            provider = provider_class(config)
            await provider.connect()
            cls._instances[key] = (signature, provider)
            if cached is not None:
                await cached[1].disconnect()
            return provider

    @classmethod
    async def invalidate(cls, config_id: int) -> None:
        """配置删除/禁用后主动清除其连接。"""
        lock = cls._locks.setdefault(config_id, asyncio.Lock())
        async with lock:
            cached = cls._instances.pop(config_id, None)
            if cached:
                await cached[1].disconnect()

    @classmethod
    async def get_config(cls, backend_type: str) -> IndexingBackendConfig:
        """获取配置（从数据库）"""
        from ext.ext_tortoise.models.knowledge_base import IndexingBackendConfig

        config = await IndexingBackendConfig.filter(
            type=IndexingBackendTypeEnum(backend_type),
            is_enabled=True,
            is_default=True,
        ).first()

        if not config:
            raise RuntimeError(f"No active config found for backend: {backend_type}")

        return config

    @classmethod
    async def clear_cache(cls) -> None:
        """清空缓存"""
        for config_id in list(cls._instances):
            await cls.invalidate(config_id)
        cls._locks.clear()


from ext.indexing.providers.milvus import MilvusProvider
from ext.indexing.providers.elasticsearch import ElasticsearchProvider

IndexingProviderFactory.register(IndexingBackendTypeEnum.elasticsearch.value, ElasticsearchProvider)
IndexingProviderFactory.register(IndexingBackendTypeEnum.milvus.value, MilvusProvider)


class IndexModelFactory:
    """IndexModel 动态工厂

    根据 embedding 配置动态创建带正确维度的 IndexModel 类

    设计思路：
    1. 在 register.py 中定义基础 IndexModel 类并绑定 provider
    2. 运行时根据 embedding_config 的维度动态创建对应模型
    3. 动态模型复用基础模型的 provider 配置

    Example:
        # ext/register.py
        class ChunkIndex(BaseIndexModel):
            content: str
            class Meta:
                index_name = "chunks"
                provider = None  # 将在 register 时绑定

        async def register():
            provider = await IndexingProviderFactory.create(config)
            ChunkIndex.Meta.provider = provider

        # 运行时使用
        emb_config = await EmbeddingModelConfig.filter(name="model-1536").first()
        DynamicChunkIndex = await IndexModelFactory.create_for_embedding(
            base_model=ChunkIndex,
            embedding_config=emb_config
        )
        # 现在 DynamicChunkIndex.Meta.dense_vector_dimension == 1536
        # collection name 会自动变为
        # "chunks_1536"
    """

    _model_registry: dict[tuple[type[BaseIndexModel], int, int], type[BaseIndexModel]] = {}

    @classmethod
    async def create_for_embedding(
        cls,
        base_model: type[BaseIndexModel],
        embedding_config: "EmbeddingModelConfig",
        provider: BaseProvider | None = None,
    ) -> type[BaseIndexModel]:
        """根据 embedding 配置动态创建 IndexModel 类

        Args:
            base_model: 基础 IndexModel 类（已在 register 时绑定 provider）
            embedding_config: embedding 模型配置（包含维度信息）

        Returns:
            动态创建的 IndexModel 子类，Meta.dense_vector_dimension 已设置

        Raises:
            ValueError: 如果 embedding_config.dimension 无效
            RuntimeError: 如果 base_model.Meta.provider 未绑定
        """

        if not isinstance(embedding_config, EmbeddingModelConfig):
            raise TypeError(f"embedding_config must be EmbeddingModelConfig, got {type(embedding_config)}")

        dimension = embedding_config.dimension

        if not dimension or dimension <= 0:
            raise ValueError(f"Invalid dimension: {dimension}")

        selected_provider = provider or base_model.Meta.provider
        if not selected_provider:
            raise RuntimeError(
                f"base_model.Meta.provider is not bound. "
                "Please bind provider in register.py first: "
                f"{base_model.__name__}.Meta.provider = await IndexingProviderFactory.create(...)",
            )

        model_key = (base_model, dimension, id(selected_provider))

        if model_key not in cls._model_registry:
            class_name = cls._get_model_key(base_model.__name__, dimension)
            cls._model_registry[model_key] = cls._create_dynamic_class(base_model, dimension, class_name, selected_provider)

        _m = cls._model_registry[model_key]

        await _m.create_schema()

        return _m

    @classmethod
    def _get_model_key(cls, base_name: str, dimension: int) -> str:
        """生成唯一模型标识

        Args:
            base_name: 基础模型名称
            dimension: 向量维度

        Returns:
            唯一标识字符串，如 "ChunkIndex_1536"
        """
        return f"{base_name}_{dimension}"

    @classmethod
    def _create_dynamic_class(
        cls,
        base_model: type[BaseIndexModel],
        dimension: int,
        class_name: str,
        provider: BaseProvider,
    ) -> type[BaseIndexModel]:
        """动态创建 IndexModel 子类

        创建一个继承自 base_model 的新类，并覆盖 Meta.dense_vector_dimension

        Args:
            base_model: 基础模型类
            dimension: 向量维度
            class_name: 新类名称

        Returns:
            动态创建的 IndexModel 子类
        """

        class DynamicMeta(base_model.Meta):
            """动态 Meta 类

            继承 base_model.Meta 的所有配置，只覆盖 dense_vector_dimension
            """

            dense_vector_dimension = dimension
            provider = None  # set below to avoid class namespace name shadowing

        DynamicMeta.provider = provider
        namespace = {
            "Meta": DynamicMeta,
            "__module__": base_model.__module__,
            "__qualname__": f"{base_model.__qualname__}.{class_name}",
            "__annotations__": {"Meta": ClassVar},
        }

        return type(class_name, (base_model,), namespace)

    @classmethod
    def clear_cache(cls) -> None:
        """清除缓存的动态模型类

        通常在测试或需要重新加载模型时使用

        Example:
            IndexModelFactory.clear_cache()
        """
        cls._model_registry.clear()

    @classmethod
    def get_registered_models(cls) -> list[str]:
        """获取已注册的动态模型列表

        Returns:
            模型 key 列表，如 ["ChunkIndex_1536", "ChunkIndex_3072"]

        Example:
            >>> models = IndexModelFactory.get_registered_models()
            >>> models
            ['ChunkIndex_1536', 'ChunkIndex_3072']
        """
        return [f"{base.__name__}_{dimension}" for base, dimension, _ in cls._model_registry]

    @classmethod
    def get_model(cls, base_name: str, dimension: int) -> type[BaseIndexModel] | None:
        """根据基础模型名和维度获取已注册的动态模型

        Args:
            base_name: 基础模型名称，如 "ChunkIndex"
            dimension: 向量维度

        Returns:
            动态模型类，如果不存在返回 None

        Example:
            >>> model_cls = IndexModelFactory.get_model("ChunkIndex", 1536)
            >>> if model_cls:
            ...     print(model_cls.Meta.dense_vector_dimension)
        """
        matches = [model for (base, dim, _), model in cls._model_registry.items() if base.__name__ == base_name and dim == dimension]
        return matches[0] if len(matches) == 1 else None
