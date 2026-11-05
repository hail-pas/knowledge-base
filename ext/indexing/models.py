from typing import cast

from pydantic import Field

from core.types import StrEnum
from ext.indexing.base import FilterClause, BaseIndexModel
from ext.indexing.factory import IndexModelFactory
from ext.ext_tortoise.models.knowledge_base import (
    Document,
    Collection,
    EmbeddingModelConfig,
    IndexingBackendConfig,
)


class SourceTypeEnum(StrEnum):
    document_content = "document_content"
    document_faq = "document_faq"


class _DocumentBaseIndexModel(BaseIndexModel):

    collection_id: int
    file_id: int
    file_name: str
    tags: list[str]
    source_type: str

    extras: dict  # = Field(index_metadata={"json_support": False}) # type: ignore
    # tenant_id: str


class DocumentContentSparseIndex(_DocumentBaseIndexModel):
    content: str

    # chunk position
    db_chunk_id: int
    start_page: int
    start_char_index: int
    end_page: int
    end_char_index: int

    class Meta:  # type: ignore
        index_name: str = "document_content"
        dense_vector_field: str | None = None
        # partition_key = "tenant_id"


class DocumentContentDenseIndex(DocumentContentSparseIndex):

    id: int = Field(default_factory=lambda: DocumentContentDenseIndex._get_id_default(), index_metadata={})  # type: ignore
    dense_vector: list[float]

    class Meta:  # type: ignore
        index_name: str = "document_content"
        # partition_key = "tenant_id"


class DocumentFAQDenseIndex(_DocumentBaseIndexModel):

    id: int = Field(default_factory=lambda: DocumentContentDenseIndex._get_id_default(), index_metadata={})  # type: ignore
    question: str
    dense_vector: list[float]
    answer: str
    db_faq_id: str  # gfaq:{id} / faq:{id}

    class Meta:  # type: ignore
        index_name: str = "document_gfaq"
        # partition_key = "tenant_id"


class CollectionIndexModelHelper:
    """Collection 索引模型访问代理

    提供 collection 关联的三种 index model 的访问属性：
    - sparse_model: DocumentContentSparseIndex (BM25 搜索)
    - dense_model: DocumentContentDenseIndex (动态维度向量搜索)
    - faq_model: DocumentGenerateFAQDenseIndex (FAQ 向量搜索)

    Usage:
        collection = await Collection.get(id=1).prefetch_related("embedding_model_config")
        helper = CollectionIndexModelHelper(collection)

        # 直接使用 model 的方法
        await helper.sparse_model.bulk_insert([...])
        dense_model = await helper.get_dense_model()
        results = await dense_model.search(query_clause, ...)
    """

    def __init__(self, collection: Collection) -> None:
        self.collection = collection
        assert self.collection.embedding_model_config
        assert isinstance(self.collection.embedding_model_config, EmbeddingModelConfig)

    async def get_provider(self):
        """Resolve this collection's backend without mutating shared model metadata."""
        from ext.indexing.factory import IndexingProviderFactory

        backend_config_id = self.collection.indexing_backend_config_id
        if backend_config_id:
            config = getattr(self.collection, "indexing_backend_config", None)
            if config is None:
                await self.collection.fetch_related("indexing_backend_config")
                config = self.collection.indexing_backend_config
            if config is None or not config.is_enabled or config.deleted_at:
                raise ValueError(
                    f"Indexing backend config {backend_config_id} is missing or disabled"
                )
            return await IndexingProviderFactory.create(config)

        provider = DocumentContentDenseIndex.Meta.provider
        if provider is None:
            raise RuntimeError("No collection indexing backend or default provider is configured")
        return provider

    async def get_sparse_model(self) -> type[DocumentContentSparseIndex]:
        provider = await self.get_provider()
        return cast(
            type[DocumentContentSparseIndex],
            await IndexModelFactory.create_for_provider(DocumentContentSparseIndex, provider),
        )

    async def get_dense_model(self) -> type[DocumentContentDenseIndex]:
        """获取稠密索引模型（自动根据 collection embedding config 创建正确维度的模型）"""
        emb_config = self.collection.embedding_model_config
        if not emb_config:
            raise ValueError("Collection embedding model config is not set")
        provider = await self.get_provider()
        return cast(
            type[DocumentContentDenseIndex],
            await IndexModelFactory.create_for_embedding(
                DocumentContentDenseIndex,
                emb_config,
                provider=provider,
            ),
        )

    async def get_faq_model(self) -> type[DocumentFAQDenseIndex]:
        """获取 FAQ 稠密索引模型（自动根据 collection embedding config 创建正确维度的模型）"""
        emb_config = self.collection.embedding_model_config
        if not emb_config:
            raise ValueError("Collection embedding model config is not set")
        provider = await self.get_provider()
        return cast(
            type[DocumentFAQDenseIndex],
            await IndexModelFactory.create_for_embedding(
                DocumentFAQDenseIndex,
                emb_config,
                provider=provider,
            ),
        )

    async def delete_by_collection(self) -> None:
        equal_clause = FilterClause(equals={"collection_id": self.collection.id})
        dense_model = await self.get_dense_model()
        sparse_model = await self.get_sparse_model()
        faq_model = await self.get_faq_model()

        await dense_model.delete_by_query(equal_clause)
        await sparse_model.delete_by_query(equal_clause)
        await faq_model.delete_by_query(equal_clause)

    async def delete_by_documents(self, documents: list[Document]) -> None:
        if not documents:
            return
        assert all(self.collection.id == doc.collection_id for doc in documents)  # type: ignore

        equal_clause = FilterClause(in_list={"file_id": [doc.id for doc in documents]})
        dense_model = await self.get_dense_model()
        faq_model = await self.get_faq_model()

        await dense_model.delete_by_query(equal_clause)
        await self.sparse_model.delete_by_query(equal_clause)
        await faq_model.delete_by_query(equal_clause)
