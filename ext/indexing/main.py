"""Indexing extension lifecycle and PostgreSQL fallback configuration."""
from typing import override
from urllib.parse import parse_qs, unquote, urlsplit

from loguru import logger

from config.default import RegisterExtensionConfig
from ext.ext_tortoise.enums import IndexingBackendTypeEnum
from ext.ext_tortoise.models.knowledge_base import IndexingBackendConfig
from ext.indexing.factory import IndexingProviderFactory


def default_postgresql_config() -> IndexingBackendConfig:
    """Use the application PostgreSQL DSN if no indexing backend is configured."""
    from config.main import local_configs

    dsn = urlsplit(str(local_configs.extensions.tortoise.knowledge_base.url))
    query = parse_qs(dsn.query)
    sslmode = query.get("sslmode", ["disable"])[0]
    return IndexingBackendConfig(
        name="default-postgresql",
        type=IndexingBackendTypeEnum.postgresql,
        host=dsn.hostname or "127.0.0.1",
        port=dsn.port or 5432,
        username=unquote(dsn.username or ""),
        password=unquote(dsn.password or ""),
        use_ssl=sslmode not in {"disable", "allow"},
        verify_ssl=sslmode in {"verify-ca", "verify-full"},
        timeout=30,
        max_retries=3,
        max_connections=10,
        extra_config={"database": (dsn.path or "/postgres").lstrip("/")},
        is_enabled=True,
        is_default=True,
    )


class ModelProviderConfig(RegisterExtensionConfig):
    """Initialize registration; backend connections are lazily resolved per collection."""

    @override
    async def register(self) -> None:
        logger.info("Indexing registry ready; providers are selected per collection")

    @override
    async def unregister(self) -> None:
        await IndexingProviderFactory.clear_cache()
        logger.info("Indexing providers disconnected")
