"""Create and cache configured file-source providers."""

import asyncio
import hashlib
import inspect
import json
from typing import Any

from loguru import logger

from ext.ext_tortoise.enums import FileSourceTypeEnum
from ext.ext_tortoise.models.knowledge_base import FileSource
from ext.file_source.base import BaseFileSourceProvider
from ext.file_source.providers.aliyun_oss import AliyunOSSFileSourceProvider
from ext.file_source.providers.local import LocalFileSourceProvider
from ext.file_source.providers.minio import MinIOFileSourceProvider
from ext.file_source.providers.s3 import S3FileSourceProvider
from ext.file_source.secrets import decrypt_file_source_extra_config
from util.secret import decrypt_secret


class FileSourceFactory:
    """Registry and lifecycle manager for file-source provider instances."""

    _providers: dict[FileSourceTypeEnum, type[BaseFileSourceProvider]] = {}
    _instances: dict[int, tuple[str, BaseFileSourceProvider]] = {}
    _locks: dict[int, asyncio.Lock] = {}

    @classmethod
    def register(cls, source_type: FileSourceTypeEnum, provider_class: type[BaseFileSourceProvider]) -> None:
        if source_type in cls._providers:
            raise ValueError(f"File-source provider already registered: {source_type.value}")
        cls._providers[source_type] = provider_class

    @classmethod
    async def create(cls, config: FileSource, use_cache: bool = True) -> BaseFileSourceProvider:
        if not config.is_enabled:
            raise ValueError(f"File source config is disabled: {config.name} (id={config.id})")
        provider_cls = cls._providers.get(config.type)
        if provider_cls is None:
            supported = ", ".join(source_type.value for source_type in cls._providers)
            raise ValueError(f"Unsupported file source type: {config.type.value}; available: {supported}")

        if not use_cache or not config._saved_in_db:
            return cls._create_instance(provider_cls, config)

        config_id = config.id
        fingerprint = cls._config_fingerprint(config)
        lock = cls._locks.setdefault(config_id, asyncio.Lock())
        async with lock:
            cached = cls._instances.get(config_id)
            if cached and cached[0] == fingerprint:
                return cached[1]
            if cached:
                await cls._close_provider(cached[1])
            provider = cls._create_instance(provider_cls, config)
            cls._instances[config_id] = (fingerprint, provider)
            return provider

    @staticmethod
    def _config_fingerprint(config: FileSource) -> str:
        values = {
            field: getattr(config, field, None)
            for field in (
                "type",
                "access_key",
                "secret_key",
                "endpoint",
                "region",
                "storage_location",
                "use_ssl",
                "verify_ssl",
                "timeout",
                "max_retries",
                "concurrent_limit",
                "max_connections",
                "extra_config",
            )
        }
        values["type"] = getattr(values["type"], "value", values["type"])
        payload = json.dumps(values, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _create_instance(
        provider_cls: type[BaseFileSourceProvider],
        config: FileSource,
    ) -> BaseFileSourceProvider:
        persisted = config._saved_in_db
        source_type = getattr(config.type, "value", config.type)
        return provider_cls(
            access_key=(
                decrypt_secret(config.access_key, purpose="file_source.access_key")
                if persisted
                else config.access_key
            ),
            secret_key=(
                decrypt_secret(config.secret_key, purpose="file_source.secret_key")
                if persisted
                else config.secret_key
            ),
            endpoint=config.endpoint,
            region=config.region,
            storage_location=config.storage_location,
            use_ssl=config.use_ssl,
            verify_ssl=config.verify_ssl,
            timeout=config.timeout,
            max_retries=config.max_retries,
            concurrent_limit=config.concurrent_limit,
            max_connections=config.max_connections,
            extra_config=(
                decrypt_file_source_extra_config(source_type, config.extra_config)
                if persisted
                else config.extra_config or {}
            ),
        )

    @staticmethod
    async def _close_provider(provider: BaseFileSourceProvider) -> None:
        close = getattr(provider, "close", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result

    @classmethod
    async def close_all(cls) -> None:
        cached = list(cls._instances.values())
        cls._instances.clear()
        cls._locks.clear()
        for _, provider in cached:
            try:
                await cls._close_provider(provider)
            except Exception:
                # Continue closing remaining providers during application shutdown.
                logger.exception("Failed to close cached file-source provider")

    @classmethod
    def clear_cache(cls, config_id: int | None = None) -> None:
        """Drop cached references; use close_all during shutdown to release clients."""
        if config_id is None:
            cls._instances.clear()
            cls._locks.clear()
        else:
            cls._instances.pop(config_id, None)
            cls._locks.pop(config_id, None)

    @classmethod
    async def create_by_name(cls, name: str, use_cache: bool = True) -> BaseFileSourceProvider:
        config = await FileSource.filter(name=name, is_enabled=True, deleted_at=0).first()
        if config is None:
            raise ValueError(f"未找到名称为 '{name}' 的已启用 file source 配置")
        return await cls.create(config, use_cache=use_cache)

    @classmethod
    async def create_default(cls, use_cache: bool = True) -> BaseFileSourceProvider:
        config = await FileSource.filter(is_enabled=True, is_default=True, deleted_at=0).first()
        if config is None:
            config = await FileSource.filter(is_enabled=True, deleted_at=0).first()
        if config is None:
            raise ValueError("未找到可用的 file source 配置")
        return await cls.create(config, use_cache=use_cache)

    @classmethod
    def get_cache_info(cls) -> dict[str, Any]:
        return {
            "cached_count": len(cls._instances),
            "cached_ids": list(cls._instances),
            "registered_providers": [source_type.value for source_type in cls._providers],
        }


FileSourceFactory.register(FileSourceTypeEnum.local_file, LocalFileSourceProvider)
FileSourceFactory.register(FileSourceTypeEnum.s3, S3FileSourceProvider)
FileSourceFactory.register(FileSourceTypeEnum.minio, MinIOFileSourceProvider)
FileSourceFactory.register(FileSourceTypeEnum.aliyun_oss, AliyunOSSFileSourceProvider)
