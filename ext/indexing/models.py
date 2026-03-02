from typing import cast
from pydantic import Field

from core.types import StrEnum
from ext.indexing.base import BaseIndexModel, FilterClause
from ext.indexing.factory import IndexModelFactory
from ext.ext_tortoise.models.knowledge_base import Collection, Document, EmbeddingModelConfig

class SourceTypeEnum(StrEnum):
    document_content = "document_content"
    document_faq = "document_faq"


class _DocumentBaseIndexModel(BaseIndexModel):

    collection_id: int
    file_id: int
    file_name: str
    tags: list[str]
    source_type: str

    extras: dict
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

    id: int = Field(default_factory=lambda: DocumentContentDenseIndex._get_id_default(), index_metadata={}) # type: ignore
    dense_vector: list[float]

    class Meta:  # type: ignore
        index_name: str = "document_content"
        # partition_key = "tenant_id"


class DocumentGenerateFAQDenseIndex(_DocumentBaseIndexModel):

    id: int = Field(default_factory=lambda: DocumentContentDenseIndex._get_id_default(), index_metadata={}) # type: ignore
    question: str
    dense_vector: list[float]
    answer: str
    db_faq_id: int # 当为0的时候则表示由文件本身内容切分而来的

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
        results = await helper.dense_model.search(query_clause, ...)
    """

    def __init__(self, collection: Collection):
        self.collection = collection
        assert self.collection.embedding_model_config
        assert isinstance(self.collection.embedding_model_config, EmbeddingModelConfig)

    @property
    def sparse_model(self) -> type[DocumentContentSparseIndex]:
        """获取稀疏索引模型（无需 embedding 维度）"""
        return DocumentContentSparseIndex

    @property
    def dense_model(self) -> type[DocumentContentDenseIndex]:
        """获取稠密索引模型（自动根据 collection embedding config 创建正确维度的模型）"""
        emb_config = self.collection.embedding_model_config
        if not emb_config:
            raise ValueError("Collection embedding model config is not set")
        return cast(
            type[DocumentContentDenseIndex],
            IndexModelFactory.create_for_embedding(
                DocumentContentDenseIndex,
                emb_config,
            ),
        )

    @property
    def faq_model(self) -> type[DocumentGenerateFAQDenseIndex]:
        """获取 FAQ 稠密索引模型（自动根据 collection embedding config 创建正确维度的模型）"""
        emb_config = self.collection.embedding_model_config
        if not emb_config:
            raise ValueError("Collection embedding model config is not set")
        return cast(
            type[DocumentGenerateFAQDenseIndex],
            IndexModelFactory.create_for_embedding(
                DocumentGenerateFAQDenseIndex,
                emb_config,
            ),
        )

    async def delete_by_collection(self):
        equal_clause = FilterClause(equals={"collection_id": self.collection.id})

        await self.dense_model.delete_by_query(equal_clause)
        await self.sparse_model.delete_by_query(equal_clause)
        await self.faq_model.delete_by_query(equal_clause)

    async def delete_by_documents(self, documents: list[Document]):
        assert all([self.collection.id == doc.collection_id for doc in documents])  # type: ignore

        equal_clause = FilterClause(in_list={"document_id": [doc.id for doc in documents]})
        await self.dense_model.delete_by_query(equal_clause)
        await self.sparse_model.delete_by_query(equal_clause)
        await self.faq_model.delete_by_query(equal_clause)
