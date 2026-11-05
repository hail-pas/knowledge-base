"""Indexing Provider 工厂（管理连接池缓存）"""

import asyncio
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
    _instances: dict[tuple[int, str], BaseProvider] = {}
    _locks: dict[tuple[int, str], asyncio.Lock] = {}

    @classmethod
    def register(cls, backend_type: str, provider_class: type[BaseProvider]) -> None:
        """注册 provider"""
        cls._providers[backend_type] = provider_class

    @classmethod
    async def create(cls, config: IndexingBackendConfig, use_cache: bool = True) -> BaseProvider:
        """创建 provider 实例（带连接池缓存）"""
        provider_class = cls._providers.get(config.type)
        if not provider_class:
            raise RuntimeError(f"Provider not found: {config.type}")

        if not use_cache:
            provider = provider_class(config)
            await provider.connect()
            return provider

        config_id = int(config.id) if config.id is not None else id(config)
        version = str(getattr(config, "updated_at", None) or hash(
            (config.host, config.port, config.username, config.password, str(config.extra_config))
        ))
        key = (config_id, version)
        lock = cls._locks.setdefault(key, asyncio.Lock())
        async with lock:
            cached = cls._instances.get(key)
            if cached:
                return cached
            # Close stale instances for the same configuration before replacing them.
            for stale_key in [item for item in cls._instances if item[0] == config_id]:
                stale = cls._instances.pop(stale_key)
                await stale.disconnect()
            provider = provider_class(config)
            await provider.connect()
            cls._instances[key] = provider
            return provider

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
        providers = list(cls._instances.values())
        cls._instances.clear()
        cls._locks.clear()
        for provider in providers:
            await provider.disconnect()


from ext.indexing.providers.milvus import MilvusProvider
from ext.indexing.providers.elasticsearch import ElasticsearchProvider
from ext.indexing.providers.postgresql import PostgreSQLProvider

IndexingProviderFactory.register(IndexingBackendTypeEnum.postgresql.value, PostgreSQLProvider)
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

    _model_registry: dict[str, type[BaseIndexModel]] = {}

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
                f"{base_model.__name__}.Meta.provider is not bound; "
                "pass provider= or bind Meta.provider before creating the index model."
            )

        model_key = cls._get_model_key(base_model.__name__, dimension, selected_provider)

        if model_key not in cls._model_registry:
            cls._model_registry[model_key] = cls._create_dynamic_class(
                base_model, dimension, model_key, selected_provider
            )

        _m = cls._model_registry[model_key]

        await _m.create_schema()

        return _m

    @classmethod
    async def create_for_provider(
        cls,
        base_model: type[BaseIndexModel],
        provider: BaseProvider,
    ) -> type[BaseIndexModel]:
        """Create a provider-scoped model that does not require a dense vector dimension."""
        model_key = cls._get_model_key(base_model.__name__, None, provider)
        if model_key not in cls._model_registry:
            cls._model_registry[model_key] = cls._create_dynamic_class(
                base_model, None, model_key, provider
            )
        model = cls._model_registry[model_key]
        await model.create_schema()
        return model

    @classmethod
    def _get_model_key(cls, base_name: str, dimension: int | None, provider: BaseProvider | None = None) -> str:
        """生成唯一模型标识

        Args:
            base_name: 基础模型名称
            dimension: 向量维度

        Returns:
            唯一标识字符串，如 "ChunkIndex_1536"
        """
        if provider is None:
            return f"{base_name}_{dimension}" if dimension is not None else base_name
        config_id = getattr(provider.config, "id", None)
        provider_key = f"{type(provider).__name__}_{config_id or id(provider)}"
        return f"{base_name}_{dimension}_{provider_key}"

    @classmethod
    def _create_dynamic_class(
        cls,
        base_model: type[BaseIndexModel],
        dimension: int | None,
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

        DynamicMeta = type(
            "DynamicMeta",
            (base_model.Meta,),
            {"dense_vector_dimension": dimension, "provider": provider},
        )

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
        return list(cls._model_registry.keys())

    @classmethod
    def get_model(
        cls, base_name: str, dimension: int, provider: BaseProvider | None = None
    ) -> type[BaseIndexModel] | None:
        """Return a cached model, optionally scoped to a specific provider."""
        if provider is not None:
            return cls._model_registry.get(cls._get_model_key(base_name, dimension, provider))
        default_key = cls._get_model_key(base_name, dimension)
        if default_key in cls._model_registry:
            return cls._model_registry[default_key]
        prefix = f"{base_name}_{dimension}_"
        return next((model for key, model in cls._model_registry.items() if key.startswith(prefix)), None)
