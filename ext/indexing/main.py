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
    model_config = ConfigDict(arbitrary_types_allowed=True)

    es_provider: BaseProvider | None = None
    milvus_provider: BaseProvider | None = None

    @override
    async def register(self) -> None:
        """连接已启用的默认索引后端；缺失的后端在使用时明确报错。"""

        logger.info("Registering model provider")

        es_config = await IndexingBackendConfig.filter(
            type=IndexingBackendTypeEnum.elasticsearch,
            is_enabled=True,
            is_default=True,
        ).first()
        milvus_config = await IndexingBackendConfig.filter(
            type=IndexingBackendTypeEnum.milvus,
            is_enabled=True,
            is_default=True,
        ).first()

        self.es_provider = await IndexingProviderFactory.create(es_config) if es_config else None
        self.milvus_provider = await IndexingProviderFactory.create(milvus_config) if milvus_config else None

        # 不改动公共基类 Meta；独立模型绑定，默认行为兼容既有调用。
        DocumentContentDenseIndex.Meta.provider = self.milvus_provider
        DocumentFAQDenseIndex.Meta.provider = self.milvus_provider
        DocumentContentSparseIndex.Meta.provider = self.es_provider

        if self.es_provider is None:
            logger.warning("No default Elasticsearch backend; sparse indexing unavailable")
        if self.milvus_provider is None:
            logger.warning("No default Milvus backend; dense indexing unavailable")

    @override
    async def unregister(self) -> None:
        """关闭 httpx.AsyncClient"""
        logger.info("Unregistering model provider")

        DocumentContentDenseIndex.Meta.provider = None
        DocumentFAQDenseIndex.Meta.provider = None
        DocumentContentSparseIndex.Meta.provider = None
        await IndexingProviderFactory.clear_cache()
        self.es_provider = None
        self.milvus_provider = None
