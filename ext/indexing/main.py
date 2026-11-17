from typing import override
from urllib.parse import parse_qs, unquote, urlsplit

from loguru import logger
from pydantic import ConfigDict

from config.default import RegisterExtensionConfig
from ext.indexing.base import BaseProvider
from ext.indexing.models import (
    DocumentFAQDenseIndex,
    DocumentContentDenseIndex,
    DocumentContentSparseIndex,
)
from ext.indexing.factory import IndexingProviderFactory
from ext.ext_tortoise.enums import IndexingBackendTypeEnum
from ext.ext_tortoise.models.knowledge_base import IndexingBackendConfig


class ModelProviderConfig(RegisterExtensionConfig):
    """Own the shared indexing provider lifecycle."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    postgresql_provider: BaseProvider | None = None

    @staticmethod
    def _default_postgresql_config() -> IndexingBackendConfig:
        """Reuse the knowledge-base PostgreSQL DSN when no explicit backend is configured."""
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

    @override
    async def register(self) -> None:
        logger.info("Registering PostgreSQL indexing provider")
        config = await IndexingBackendConfig.filter(
            type=IndexingBackendTypeEnum.postgresql,
            is_enabled=True,
            is_default=True,
        ).first()
        if config is None:
            config = self._default_postgresql_config()
            logger.info("No explicit PostgreSQL indexing config; reusing knowledge-base database")
        provider = await IndexingProviderFactory.create(config)
        self.postgresql_provider = provider
        DocumentContentDenseIndex.Meta.provider = provider
        DocumentContentSparseIndex.Meta.provider = provider
        DocumentFAQDenseIndex.Meta.provider = provider

    @override
    async def unregister(self) -> None:
        logger.info("Unregistering PostgreSQL indexing provider")
        await IndexingProviderFactory.clear_cache()
        self.postgresql_provider = None
