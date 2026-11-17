"""Indexing Provider 工厂（管理连接池缓存）"""

import asyncio
import hashlib
import json
from importlib import import_module
from typing import ClassVar

from ext.indexing.base import BaseProvider, BaseIndexModel
from ext.ext_tortoise.models.knowledge_base import (
    EmbeddingModelConfig,
    IndexingBackendConfig,
    IndexingBackendTypeEnum,
)


class IndexingProviderFactory:
    """Indexing Provider 工厂"""

    _providers: dict[str, type[BaseProvider] | str] = {
    _instances: dict[tuple[int, str], BaseProvider] = {}
    _locks: dict[tuple[int, str], asyncio.Lock] = {}

    @staticmethod
    def _cache_key(config: IndexingBackendConfig) -> tuple[int, str]:
        """Identify a connection by backend settings, not mutable timestamps."""
        identity = int(config.id) if config.id is not None else 0
        settings = {
            "type": str(config.type),
            "host": config.host,
            "port": config.port,
            "username": config.username,
            "password": config.password,
            "use_ssl": config.use_ssl,
            "verify_ssl": config.verify_ssl,
            "timeout": config.timeout,
            "max_connections": config.max_connections,
            "max_retries": config.max_retries,
            "extra_config": config.extra_config,
        }
        serialized = json.dumps(settings, sort_keys=True, default=str)
        return identity, hashlib.sha256(serialized.encode()).hexdigest()

    @classmethod
    def register(cls, backend_type: str, provider_class: type[BaseProvider] | str) -> None:
        """注册 provider"""
        cls._providers[backend_type] = provider_class

    @classmethod
    async def create(cls, config: IndexingBackendConfig, use_cache: bool = True) -> BaseProvider:
        """创建 provider 实例（带连接池缓存）"""
        registered = cls._providers.get(config.type)
        if registered is None:
            raise RuntimeError(f"Provider not found: {config.type}")
        if isinstance(registered, str):
            module, _, name = registered.partition(":")
            provider_class = getattr(import_module(module), name)
            if not issubclass(provider_class, BaseProvider):
                raise TypeError(f"Invalid indexing provider: {registered}")
            cls._providers[config.type] = provider_class
        else:
            provider_class = registered

        if not use_cache:
            provider = provider_class(config)
            await provider.connect()
            return provider

        key = cls._cache_key(config)
        lock = cls._locks.setdefault(key, asyncio.Lock())
        async with lock:
            cached = cls._instances.get(key)
            if cached:
                return cached
            # Keep older provider instances alive for in-flight requests; all are closed on shutdown.
            provider = provider_class(config)
            try:
                await provider.connect()
            except BaseException:
                await provider.disconnect()
                raise
            cls._instances[key] = provider
            return provider

    @classmethod
    async def get_default(cls) -> BaseProvider:
        """Resolve default backend at call time so switching requires no restart."""
        configs = await IndexingBackendConfig.filter(
            is_enabled=True, is_default=True, deleted_at=0,
        ).limit(2)
        if len(configs) > 1:
            raise RuntimeError("Multiple default indexing backends configured")
        if configs:
            return await cls.create(configs[0])
        from ext.indexing.main import default_postgresql_config

        return await cls.create(default_postgresql_config())

    @classmethod
    async def clear_cache(cls) -> None:
        """清空缓存"""
        providers = list(cls._instances.values())
        cls._instances.clear()
        cls._locks.clear()
        IndexModelFactory.clear_cache()
        for provider in providers:
            await provider.disconnect()


# Import only the selected backend and its optional dependencies.
IndexingProviderFactory.register(
    IndexingBackendTypeEnum.postgresql.value, "ext.indexing.providers.postgresql:PostgreSQLProvider",
)
IndexingProviderFactory.register(
    IndexingBackendTypeEnum.elasticsearch.value, "ext.indexing.providers.elasticsearch:ElasticsearchProvider",
)
IndexingProviderFactory.register(
    IndexingBackendTypeEnum.milvus.value, "ext.indexing.providers.milvus:MilvusProvider",
)

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
    _schema_locks: dict[str, asyncio.Lock] = {}
    _schemas_ready: set[str] = set()

    @classmethod
    async def _ensure_schema(cls, key: str, model: type[BaseIndexModel]) -> None:
        """Create each provider-scoped schema once per process, including concurrent callers."""
        lock = cls._schema_locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key not in cls._schemas_ready:
                await model.create_schema()
                cls._schemas_ready.add(key)


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

        if provider is None:
            raise ValueError("provider= is required; model definitions must not own connection state")
        selected_provider = provider

        model_key = cls._get_model_key(base_model.__name__, dimension, selected_provider)

        if model_key not in cls._model_registry:
            cls._model_registry[model_key] = cls._create_dynamic_class(
                base_model, dimension, model_key, selected_provider
            )

        _m = cls._model_registry[model_key]

        await cls._ensure_schema(model_key, _m)

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
        await cls._ensure_schema(model_key, model)
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
        provider_key = f"{type(provider).__name__}_{config_id or 'transient'}_{id(provider)}"
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
        cls._schemas_ready.clear()
        cls._schema_locks.clear()

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
        matches = [model for key, model in cls._model_registry.items() if key.startswith(prefix)]
        return matches[0] if len(matches) == 1 else None
