from typing import override
from pydantic import ConfigDict
from config.default import RegisterExtensionConfig
from loguru import logger
from ext.indexing.base import BaseProvider
from ext.indexing.factory import IndexingProviderFactory
from ext.ext_tortoise.models.knowledge_base import IndexingBackendConfig
from ext.ext_tortoise.enums import IndexingBackendTypeEnum
from ext.indexing.models import DocumentContentDenseIndex, DocumentContentSparseIndex, DocumentFAQDenseIndex


class ModelProviderConfig(RegisterExtensionConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    es_provider: BaseProvider | None = None
    milvus_provider: BaseProvider | None = None

    @override
    async def register(self) -> None:
        """初始化 httpx.AsyncClient"""

        logger.info("Registering model provider")

        es_config = await IndexingBackendConfig.filter(
            type=IndexingBackendTypeEnum.elasticsearch, is_enabled=True, is_default=True
        ).first()
        milvus_config = await IndexingBackendConfig.filter(
            type=IndexingBackendTypeEnum.milvus, is_enabled=True, is_default=True
        ).first()

        assert es_config
        assert milvus_config
        es_provider = await IndexingProviderFactory.create(es_config, use_cache=False)
        milvus_provider = await IndexingProviderFactory.create(milvus_config, use_cache=False)
        self.es_provider = es_provider
        self.milvus_provider = milvus_provider
        DocumentContentDenseIndex.Meta.provider = self.milvus_provider  # type: ignore
        DocumentContentSparseIndex.Meta.provider = self.es_provider  # type: ignore
        DocumentFAQDenseIndex.Meta.provider = self.milvus_provider  # type: ignore

    @override
    async def unregister(self) -> None:
        """关闭 httpx.AsyncClient"""
        logger.info("Unregistering model provider")

        if self.es_provider:
            await self.es_provider.disconnect()
        if self.milvus_provider:
            await self.milvus_provider.disconnect()
