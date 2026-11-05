from typing import override

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

    @override
    async def register(self) -> None:
        logger.info("Registering PostgreSQL indexing provider")
        config = await IndexingBackendConfig.filter(
            type=IndexingBackendTypeEnum.postgresql,
            is_enabled=True,
            is_default=True,
        ).first()
        if config is None:
            raise RuntimeError(
                "No enabled default PostgreSQL indexing backend configured. "
                "Create an IndexingBackendConfig with type=postgresql and is_default=True."
            )
        provider = await IndexingProviderFactory.create(config)
        self.postgresql_provider = provider
        DocumentContentDenseIndex.Meta.provider = provider
        DocumentContentSparseIndex.Meta.provider = provider
        DocumentFAQDenseIndex.Meta.provider = provider

    @override
    async def unregister(self) -> None:
        logger.info("Unregistering PostgreSQL indexing provider")
        if self.postgresql_provider:
            await self.postgresql_provider.disconnect()
            self.postgresql_provider = None
