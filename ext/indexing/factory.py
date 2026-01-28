"""Indexing Provider 工厂（管理连接池缓存）"""

from typing import Type, Dict
from ext.ext_tortoise.models.knowledge_base import IndexingBackendConfig, IndexingBackendTypeEnum
from ext.indexing.base import BaseProvider


class IndexingProviderFactory:
    """Indexing Provider 工厂"""

    _providers: dict[str, type[BaseProvider]] = {}
    _instances: dict[int, BaseProvider] = {}

    @classmethod
    def register(cls, backend_type: str, provider_class: type[BaseProvider]):
        """注册 provider"""
        cls._providers[backend_type] = provider_class

    @classmethod
    async def create(cls, config: IndexingBackendConfig, use_cache: bool = True) -> BaseProvider:
        """创建 provider 实例（带连接池缓存）"""
        if use_cache and config.id in cls._instances:
            return cls._instances[config.id]

        provider_class = cls._providers.get(config.type)
        if not provider_class:
            raise RuntimeError(f"Provider not found: {config.type}")

        provider = provider_class(config) # type: ignore
        await provider.connect()

        if use_cache:
            cls._instances[config.id] = provider

        return provider

    @classmethod
    async def get_config(cls, backend_type: str) -> IndexingBackendConfig:
        """获取配置（从数据库）"""
        from ext.ext_tortoise.models.knowledge_base import IndexingBackendConfig

        config = await IndexingBackendConfig.filter(
            type=IndexingBackendTypeEnum(backend_type), is_enabled=True, is_default=True,
        ).first()

        if not config:
            raise RuntimeError(f"No active config found for backend: {backend_type}")

        return config

    @classmethod
    async def clear_cache(cls):
        """清空缓存"""
        for provider in cls._instances.values():
            await provider.disconnect()
        cls._instances.clear()


from ext.indexing.providers.elasticsearch import ElasticsearchProvider
from ext.indexing.providers.milvus import MilvusProvider

IndexingProviderFactory.register(IndexingBackendTypeEnum.elasticsearch.value, ElasticsearchProvider)
IndexingProviderFactory.register("milvus", MilvusProvider)
