"""
File Source Factory

负责根据数据库配置动态创建 file source provider 实例
"""

from typing import Any
import asyncio
from loguru import logger

from ext.file_source.base import BaseFileSourceProvider, FileMetadata
from ext.ext_tortoise.enums import FileSourceTypeEnum
from ext.ext_tortoise.models.knowledge_base import FileSource


class FileSourceFactory:
    """文件源工厂类

    负责根据数据库配置动态创建 file source provider 实例
    """

    _providers: dict[FileSourceTypeEnum, type[BaseFileSourceProvider]] = {}
    _instances: dict[int, BaseFileSourceProvider] = {}
    _locks: dict[int, asyncio.Lock] = {}

    @classmethod
    def register(cls, source_type: FileSourceTypeEnum, provider_class: type[BaseFileSourceProvider]) -> None:
        """注册新的 file source provider"""
        if source_type in cls._providers:
            logger.warning(f"Provider type {source_type.value} 已注册，将被覆盖")
        cls._providers[source_type] = provider_class
        logger.info(f"Registered file source provider: {source_type.value} -> {provider_class.__name__}")

    @classmethod
    async def create(cls, config: FileSource, use_cache: bool = True) -> BaseFileSourceProvider:
        """创建 file source provider 实例"""
        if not config.is_enabled:
            raise ValueError(f"File source config is disabled: {config.name} (id={config.id})")

        provider_cls = cls._providers.get(config.type)
        if not provider_cls:
            available_types = ", ".join([t.value for t in cls._providers.keys()])
            raise ValueError(f"Unsupported file source type: {config.type.value}, available: {available_types}")

        if not use_cache or not config._saved_in_db:
            return cls._create_instance(provider_cls, config)

        if config.id in cls._instances:
            return cls._instances[config.id]

        if config.id not in cls._locks:
            cls._locks[config.id] = asyncio.Lock()

        async with cls._locks[config.id]:
            if config.id in cls._instances:
                return cls._instances[config.id]

            provider = cls._create_instance(provider_cls, config)
            cls._instances[config.id] = provider

            return provider

    @classmethod
    def _create_instance(cls, provider_cls: type[BaseFileSourceProvider], config: FileSource) -> BaseFileSourceProvider:
        """创建 provider 实例的内部方法"""
        return provider_cls(
            access_key=config.access_key,
            secret_key=config.secret_key,
            endpoint=config.endpoint,
            region=config.region,
            storage_location=config.storage_location,
            use_ssl=config.use_ssl,
            verify_ssl=config.verify_ssl,
            timeout=config.timeout,
            max_retries=config.max_retries,
            concurrent_limit=config.concurrent_limit,
            max_connections=config.max_connections,
            extra_config=config.extra_config or {},
        )

    @classmethod
    def clear_cache(cls, config_id: int | None = None) -> None:
        """清除实例缓存"""
        if config_id is None:
            cls._instances.clear()
            cls._locks.clear()
        else:
            cls._instances.pop(config_id, None)
            cls._locks.pop(config_id, None)

    @classmethod
    async def create_by_name(cls, name: str, use_cache: bool = True) -> BaseFileSourceProvider:
        """根据配置名称创建实例"""
        config = await FileSource.filter(name=name, is_enabled=True).first()
        if not config:
            raise ValueError(f"未找到名称为 '{name}' 的已启用 file source 配置")

        return await cls.create(config, use_cache=use_cache)

    @classmethod
    async def create_default(cls, use_cache: bool = True) -> BaseFileSourceProvider:
        """创建默认的 file source 实例"""
        config = await FileSource.filter(is_enabled=True, is_default=True).first()

        if not config:
            config = await FileSource.filter(is_enabled=True).first()

        if not config:
            raise ValueError("未找到可用的 file source 配置")

        return await cls.create(config, use_cache=use_cache)

    @classmethod
    def get_cache_info(cls) -> dict[str, Any]:
        """获取缓存信息"""
        return {
            "cached_count": len(cls._instances),
            "cached_ids": list(cls._instances.keys()),
            "registered_providers": [t.value for t in cls._providers.keys()],
        }
