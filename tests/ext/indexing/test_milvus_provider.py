"""Milvus Provider 单元测试"""
from typing import List, Dict, Any
from datetime import datetime
from pydantic import Field

import pytest
import numpy as np

from ext.indexing.providers.milvus import MilvusProvider
from ext.indexing.base import (
    FieldDefinition,
    FieldType,
    SearchQuery,
    QueryCondition,
    MatchType,
    BoolQuery,
    DenseIndexModel,
    VectorSearchParam,
)
from ext.ext_tortoise.enums import IndexingBackendTypeEnum

# 从 conftest 导入共享的 fixtures
from tests.ext.indexing.conftest import (
    milvus_config,
    sample_embedding,
    should_skip_milvus_test,
)


# ============================================================================
# 测试用的索引模型类
# ============================================================================

class TestDocumentModel(DenseIndexModel):
    """Milvus 测试文档模型"""

    # Pydantic 字段定义
    id: str = Field(default="", description="文档 ID")
    title: str = Field(default="", description="文档标题")
    content: str = Field(default="", description="文档内容")
    author: str = Field(default="", description="作者")
    category: str = Field(default="", description="分类")
    tags: str = Field(default="", description="标签")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    embedding: List[float] = Field(default_factory=list, description="向量嵌入")
    extras: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")

    @classmethod
    def get_index_name(cls) -> str:
        return "test_documents_milvus"

    @classmethod
    def get_index_type(cls) -> str:
        return "dense"

    @classmethod
    def get_backend_type(cls):
        return IndexingBackendTypeEnum.milvus

    @classmethod
    def get_field_definitions(cls) -> List[FieldDefinition]:
        return [
            FieldDefinition(name="id", type=FieldType.keyword, is_primary_key=True, auto_id=False, max_length=255),
            FieldDefinition(name="title", type=FieldType.text, max_length=65535),
            FieldDefinition(name="content", type=FieldType.text, max_length=65535),
            FieldDefinition(name="author", type=FieldType.keyword, max_length=255),
            FieldDefinition(name="category", type=FieldType.keyword, max_length=255),
            FieldDefinition(name="tags", type=FieldType.keyword, max_length=255),
            FieldDefinition(name="created_at", type=FieldType.datetime),
            FieldDefinition(
                name="embedding",
                type=FieldType.dense_vector,
                dimension=768,
                metric_type="L2"
            ),
            FieldDefinition(
                name="extras",
                type=FieldType.json,
                description="动态 JSON 字段，用于存储额外的元数据"
            ),
        ]


class PartitionKeyDocumentModel(DenseIndexModel):
    """使用 Partition Key 的测试文档模型"""

    # Pydantic 字段定义
    id: str = Field(default="", description="文档 ID")
    title: str = Field(default="", description="文档标题")
    tenant_id: str = Field(default="", description="租户 ID")
    embedding: List[float] = Field(default_factory=list, description="向量嵌入")

    @classmethod
    def get_index_name(cls) -> str:
        return "test_partition_key_milvus"

    @classmethod
    def get_index_type(cls) -> str:
        return "dense"

    @classmethod
    def get_backend_type(cls):
        return IndexingBackendTypeEnum.milvus

    @classmethod
    def get_field_definitions(cls) -> List[FieldDefinition]:
        return [
            FieldDefinition(name="id", type=FieldType.keyword, is_primary_key=True, auto_id=False, max_length=255),
            FieldDefinition(name="title", type=FieldType.text, max_length=65535),
            FieldDefinition(name="tenant_id", type=FieldType.keyword, max_length=255, is_partition_key=True, index=False),
            FieldDefinition(
                name="embedding",
                type=FieldType.dense_vector,
                dimension=768,
                metric_type="L2"
            ),
        ]



# ============================================================================
# 辅助函数
# ============================================================================

def create_sample_document(doc_id: str, embedding: List[float], extras: dict = None) -> TestDocumentModel:
    """创建示例文档"""
    return TestDocumentModel(
        id=doc_id,
        title=f"Test Document {doc_id}",
        content=f"This is test content for {doc_id}",
        author="test_user",
        category="test",
        tags="test,milvus",
        created_at=datetime.now(),
        embedding=embedding,
        extras=extras or {"version": "1.0"},
    )


def create_multiple_documents(count: int, embedding_dim: int = 768) -> List[TestDocumentModel]:
    """创建多个测试文档"""
    titles = [
        "Machine Learning",
        "Deep Learning",
        "Python Programming",
        "Data Science",
        "AI Applications",
    ]
    contents = [
        "Introduction to machine learning",
        "Deep learning neural networks",
        "Python programming for data",
        "Data science techniques",
        "Artificial intelligence in industry",

    ]
    authors = ["author_a", "author_b", "author_a", "author_b", "author_a"]
    categories = ["ai", "ai", "programming", "data", "ai"]
    tags_list = [
        "ml,ai",
        "dl,neural",
        "python,data",
        "data,science",
        "ai,industry",
    ]

    documents = []
    num_templates = len(titles)
    for i in range(count):
        idx = i % num_templates
        embedding = np.random.rand(embedding_dim).tolist()
        extras = {
            "version": "1.0",
            "priority": i % 3,
            "special_attr": f"value_{i % 5}",
        }
        doc = TestDocumentModel(
            id=f"doc_{i+1:03d}",
            title=titles[idx],
            content=contents[idx],
            author=authors[idx],
            category=categories[idx],
            tags=tags_list[idx],
            created_at=datetime.now(),
            embedding=embedding,
            extras=extras,
        )
        documents.append(doc)

    return documents


# ============================================================================
# 测试类
# ============================================================================

@pytest.mark.asyncio
class TestMilvusProvider:
    """Milvus Provider 测试类"""

    # ========================================================================
    # 初始化测试
    # ========================================================================

    async def test_init_success(self, milvus_config):
        """测试成功初始化 MilvusProvider"""
        provider = MilvusProvider(
            backend_type=IndexingBackendTypeEnum.milvus,
            config=milvus_config.copy(),
        )

        assert provider.host == milvus_config["host"]
        assert provider.port == milvus_config["port"]
        assert provider.secure == milvus_config["secure"]
        assert provider.db_name == milvus_config["db_name"]

        await provider.disconnect()

    async def test_init_missing_host(self):
        """测试缺少 host 参数时抛出异常"""
        config = {}  # 缺少必需的 host

        with pytest.raises(KeyError):
            MilvusProvider(
                backend_type=IndexingBackendTypeEnum.milvus,
                config=config,
            )

    # ========================================================================
    # 连接测试
    # ========================================================================

    async def test_connect_success(self, milvus_config):
        """测试成功连接到 Milvus"""
        if should_skip_milvus_test(milvus_config):
            pytest.skip("MILVUS_HOST not configured or is localhost")

        provider = MilvusProvider(
            backend_type=IndexingBackendTypeEnum.milvus,
            config=milvus_config.copy(),
        )

        await provider.connect()
        assert provider._client is not None

        await provider.disconnect()

    async def test_connect_invalid_host(self):
        """测试连接到无效的主机"""
        config = {
            "host": "invalid-host-that-does-not-exist.example.com",
            "port": 19530,
            "timeout": 5,  # 短超时
        }

        provider = MilvusProvider(
            backend_type=IndexingBackendTypeEnum.milvus,
            config=config,
        )

        try:
            with pytest.raises(Exception, match="Failed to connect"):
                await provider.connect()
        finally:
            await provider.disconnect()

    async def test_ping(self, milvus_config):
        """测试 ping 方法"""
        if should_skip_milvus_test(milvus_config):
            pytest.skip("MILVUS_HOST not configured or is localhost")

        provider = MilvusProvider(
            backend_type=IndexingBackendTypeEnum.milvus,
            config=milvus_config.copy(),
        )

        try:
            # 未连接时应该返回 False
            assert await provider.ping() is False

            # 连接后应该返回 True
            await provider.connect()
            assert await provider.ping() is True
        finally:
            await provider.disconnect()

    # ========================================================================
    # 索引操作测试
    # ========================================================================

    async def test_create_and_drop_index(self, milvus_config):
        """测试创建和删除索引"""
        if should_skip_milvus_test(milvus_config):
            pytest.skip("MILVUS_HOST not configured or is localhost")

        provider = MilvusProvider(
            backend_type=IndexingBackendTypeEnum.milvus,
            config=milvus_config.copy(),
        )

        try:
            await provider.connect()

            # 确保索引不存在
            if await provider.index_exists(TestDocumentModel):
                await provider.drop_index(TestDocumentModel)

            # 创建索引
            assert await provider.create_index(TestDocumentModel) is True
            assert await provider.index_exists(TestDocumentModel) is True

            # 再次创建应该返回 False（已存在）
            assert await provider.create_index(TestDocumentModel) is False

            # 删除索引
            assert await provider.drop_index(TestDocumentModel) is True
            assert await provider.index_exists(TestDocumentModel) is False
        finally:
            try:
                if await provider.index_exists(TestDocumentModel):
                    await provider.drop_index(TestDocumentModel)
            except Exception:
                pass
            await provider.disconnect()

    # ========================================================================
    # 文档操作测试（CRUD 合并）
    # ========================================================================

    async def test_crud_operations(self, milvus_config):
        """测试完整的 CRUD 操作（合并 insert/update/upsert/bulk/delete）"""
        if should_skip_milvus_test(milvus_config):
            pytest.skip("MILVUS_HOST not configured or is localhost")

        provider = MilvusProvider(
            backend_type=IndexingBackendTypeEnum.milvus,
            config=milvus_config.copy(),
        )

        try:
            await provider.connect()

            # 创建测试索引
            if await provider.index_exists(TestDocumentModel):
                await provider.drop_index(TestDocumentModel)
            await provider.create_index(TestDocumentModel)

            # ========================================
            # 场景 1: 插入和获取单条文档
            # ========================================
            doc = create_sample_document(
                "doc_001",
                np.random.rand(768).tolist(),
                extras={"special_attr": "special", "priority": 1}
            )
            assert await provider.insert(doc) is True

            # 显式 flush 确保数据持久化
            await TestDocumentModel.flush(provider)

            retrieved_doc = await provider.get_by_id(TestDocumentModel, "doc_001")
            assert retrieved_doc is not None
            assert retrieved_doc.id == "doc_001"
            assert retrieved_doc.extras.get("special_attr") == "special"

            # ========================================
            # 场景 2: 更新文档
            # ========================================
            doc.title = "Updated Title"
            doc.extras = {"priority": 2, "new_field": "value"}
            assert await provider.update(doc) is True

            # 显式 flush 确保数据持久化
            await TestDocumentModel.flush(provider)

            updated_doc = await provider.get_by_id(TestDocumentModel, "doc_001")
            assert updated_doc.title == "Updated Title"
            assert updated_doc.extras.get("priority") == 2

            # ========================================
            # 场景 3: Upsert 新文档和更新
            # ========================================
            upsert_doc = create_sample_document(
                "doc_002",
                np.random.rand(768).tolist(),
                extras={"status": "new"}
            )
            assert await provider.upsert(upsert_doc) is True

            # 显式 flush 确保数据持久化
            await TestDocumentModel.flush(provider)

            upsert_doc.title = "Upserted Title"
            upsert_doc.extras = {"status": "updated"}
            assert await provider.upsert(upsert_doc) is True

            # 显式 flush 确保数据持久化
            await TestDocumentModel.flush(provider)

            retrieved = await provider.get_by_id(TestDocumentModel, "doc_002")
            assert retrieved.title == "Upserted Title"

            # ========================================
            # 场景 4: 批量插入
            # ========================================
            bulk_docs = create_multiple_documents(5)
            count = await provider.bulk_insert(TestDocumentModel, bulk_docs)
            assert count == 5

            # 显式 flush 确保数据持久化
            await TestDocumentModel.flush(provider)

            # 验证批量插入的文档
            for bulk_doc in bulk_docs:
                retrieved = await provider.get_by_id(TestDocumentModel, bulk_doc.id)
                assert retrieved is not None
                assert retrieved.extras.get("version") == "1.0"

            # ========================================
            # 场景 5: 删除文档
            # ========================================
            assert await provider.delete(TestDocumentModel, "doc_001") == 1

            # 显式 flush 确保删除生效
            await TestDocumentModel.flush(provider)

            deleted_doc = await provider.get_by_id(TestDocumentModel, "doc_001")
            assert deleted_doc is None

        finally:
            try:
                if await provider.index_exists(TestDocumentModel):
                    await provider.drop_index(TestDocumentModel)
            except Exception:
                pass
            await provider.disconnect()

    # ========================================================================
    # 搜索操作测试（合并用例）
    # ========================================================================

    async def test_search_operations(self, milvus_config):
        """测试搜索操作：向量检索 + 过滤查询（复用数据）"""
        if should_skip_milvus_test(milvus_config):
            pytest.skip("MILVUS_HOST not configured or is localhost")

        provider = MilvusProvider(
            backend_type=IndexingBackendTypeEnum.milvus,
            config=milvus_config.copy(),
        )

        try:
            await provider.connect()

            # 创建测试索引
            if await provider.index_exists(TestDocumentModel):
                await provider.drop_index(TestDocumentModel)
            await provider.create_index(TestDocumentModel)

            # 插入测试文档（复用数据）
            docs = [
                create_sample_document(
                    "doc_001",
                    np.random.rand(768).tolist(),
                    extras={"special_attr": "special", "priority": 1}
                ),
                create_sample_document(
                    "doc_002",
                    np.random.rand(768).tolist(),
                    extras={"special_attr": "normal", "priority": 2}
                ),
                create_sample_document(
                    "doc_003",
                    np.random.rand(768).tolist(),
                    extras={"special_attr": "special", "priority": 3}
                ),
                create_sample_document(
                    "doc_004",
                    np.random.rand(768).tolist(),
                    extras={"special_attr": "normal", "priority": 4}
                ),
                create_sample_document(
                    "doc_005",
                    np.random.rand(768).tolist(),
                    extras={"special_attr": "special", "priority": 5}
                ),
            ]
            await provider.bulk_insert(TestDocumentModel, docs)

            # 显式 flush 确保数据持久化
            await TestDocumentModel.flush(provider)

            # ========================================
            # 场景 1: 基本向量检索（无过滤）
            # ========================================
            query_vector = np.random.rand(768).tolist()
            vector_param = VectorSearchParam(vector=query_vector, k=10)
            search_query = SearchQuery(vector_param=vector_param, include_scores=True)

            result = await provider.search(TestDocumentModel, search_query)

            assert len(result.documents) > 0
            assert result.total > 0
            assert len(result.scores) == len(result.documents)
            # 验证分数类型
            for score in result.scores:
                assert isinstance(score, float)

            # ========================================
            # 场景 2: 使用 extras 字段过滤（special_attr == "special"）
            # ========================================
            vector_param = VectorSearchParam(
                vector=query_vector,
                k=10,
                filter=BoolQuery(must=[
                    QueryCondition(
                        field='extras["special_attr"]',
                        value="special",
                        match_type=MatchType.term
                    )
                ])
            )
            result = await provider.search(TestDocumentModel, SearchQuery(vector_param=vector_param))

            # 应该返回 3 个 special_attr == "special" 的文档
            assert len(result.documents) == 3
            for doc in result.documents:
                assert doc.extras.get("special_attr") == "special"

            # ========================================
            # 场景 3: 复杂过滤条件（priority >= 3 AND special_attr == "special"）
            # ========================================
            vector_param = VectorSearchParam(
                vector=query_vector,
                k=10,
                filter=BoolQuery(must=[
                    QueryCondition(
                        field='extras["special_attr"]',
                        value="special",
                        match_type=MatchType.term
                    ),
                    QueryCondition(
                        field='extras["priority"]',
                        value=3,
                        match_type=MatchType.range,
                        range_gte=3
                    )
                ])
            )
            result = await provider.search(TestDocumentModel, SearchQuery(vector_param=vector_param))

            # 应该返回 2 个文档（doc_003 和 doc_005）
            assert len(result.documents) == 2
            for doc in result.documents:
                assert doc.extras.get("special_attr") == "special"
                assert doc.extras.get("priority") >= 3
        finally:
            try:
                if await provider.index_exists(TestDocumentModel):
                    await provider.drop_index(TestDocumentModel)
            except Exception:
                pass
            await provider.disconnect()

    async def test_count_documents(self, milvus_config):
        """测试统计文档数量"""
        if should_skip_milvus_test(milvus_config):
            pytest.skip("MILVUS_HOST not configured or is localhost")

        provider = MilvusProvider(
            backend_type=IndexingBackendTypeEnum.milvus,
            config=milvus_config.copy(),
        )

        try:
            await provider.connect()

            # 创建测试索引
            if await provider.index_exists(TestDocumentModel):
                await provider.drop_index(TestDocumentModel)
            await provider.create_index(TestDocumentModel)

            # 批量插入文档
            documents = create_multiple_documents(10)
            await provider.bulk_insert(TestDocumentModel, documents)

            # 显式 flush 确保数据持久化
            await TestDocumentModel.flush(provider)

            # 统计所有文档
            total = await provider.count(TestDocumentModel)
            assert total == 10
        finally:
            try:
                if await provider.index_exists(TestDocumentModel):
                    await provider.drop_index(TestDocumentModel)
            except Exception:
                pass
            await provider.disconnect()

    # ========================================================================
    # 分区操作测试
    # ========================================================================

    async def test_partition_key_and_manual_partitions(self, milvus_config):
        """测试 Partition Key 和手动分区的使用"""
        if should_skip_milvus_test(milvus_config):
            pytest.skip("MILVUS_HOST not configured or is localhost")

        provider = MilvusProvider(
            backend_type=IndexingBackendTypeEnum.milvus,
            config=milvus_config.copy(),
        )

        try:
            await provider.connect()

            # ========================================
            # 测试 1: Partition Key 自动分区
            # ========================================
            if await provider.index_exists(PartitionKeyDocumentModel):
                await provider.drop_index(PartitionKeyDocumentModel)
            await provider.create_index(PartitionKeyDocumentModel)

            # 插入不同租户的数据（会被自动分配到不同分区）
            docs_tenant_a = [
                PartitionKeyDocumentModel(
                    id=f"pk_doc_{i}",
                    title=f"Tenant A Doc {i}",
                    tenant_id="tenant_a",
                    embedding=np.random.rand(768).tolist()
                )
                for i in range(3)
            ]
            docs_tenant_b = [
                PartitionKeyDocumentModel(
                    id=f"pk_doc_{i+3}",
                    title=f"Tenant B Doc {i}",
                    tenant_id="tenant_b",
                    embedding=np.random.rand(768).tolist()
                )
                for i in range(3)
            ]
            await provider.bulk_insert(PartitionKeyDocumentModel, docs_tenant_a + docs_tenant_b)

            # 显式 flush 确保数据持久化
            await PartitionKeyDocumentModel.flush(provider)

            query_vector = np.random.rand(768).tolist()
            vector_param = VectorSearchParam(vector=query_vector, k=10)
            search_query = SearchQuery(vector_param=vector_param)
            result = await provider.search(PartitionKeyDocumentModel, search_query)

            # 验证所有文档都被插入了
            assert len(result.documents) == 6

            # ========================================
            # Partition Key 过滤查询测试
            # ========================================
            # 注意：对于 Partition Key，不能使用 partition_name 参数直接查询分区
            # 而是应该使用过滤表达式，Milvus 会自动优化查询到相关分区

            # 测试 1: 查询 tenant_a 的数据
            filter_tenant_a = BoolQuery(must=[
                QueryCondition(field='tenant_id', value="tenant_a", match_type=MatchType.term),
            ])
            vector_param_tenant_a = VectorSearchParam(
                vector=query_vector,
                k=10,
                filter=filter_tenant_a
            )
            search_query_tenant_a = SearchQuery(vector_param=vector_param_tenant_a)
            result_tenant_a = await provider.search(PartitionKeyDocumentModel, search_query_tenant_a)

            # 验证只返回了 tenant_a 的数据
            assert len(result_tenant_a.documents) == 3
            for doc in result_tenant_a.documents:
                assert doc.tenant_id == "tenant_a"
                assert "Tenant A" in doc.title

            # 测试 2: 查询 tenant_b 的数据
            filter_tenant_b = BoolQuery(must=[
                QueryCondition(field='tenant_id', value="tenant_b", match_type=MatchType.term),
            ])
            vector_param_tenant_b = VectorSearchParam(
                vector=query_vector,
                k=10,
                filter=filter_tenant_b
            )
            search_query_tenant_b = SearchQuery(vector_param=vector_param_tenant_b)
            result_tenant_b = await provider.search(PartitionKeyDocumentModel, search_query_tenant_b)

            # 验证只返回了 tenant_b 的数据
            assert len(result_tenant_b.documents) == 3
            for doc in result_tenant_b.documents:
                assert doc.tenant_id == "tenant_b"
                assert "Tenant B" in doc.title

            # 测试 3: 不使用过滤条件，验证返回所有数据
            vector_param_all = VectorSearchParam(vector=query_vector, k=10)
            search_query_all = SearchQuery(vector_param=vector_param_all)
            result_all = await provider.search(PartitionKeyDocumentModel, search_query_all)

            # 验证返回了所有租户的数据
            assert len(result_all.documents) == 6
            tenant_ids = {doc.tenant_id for doc in result_all.documents}
            assert tenant_ids == {"tenant_a", "tenant_b"}

            await provider.drop_index(PartitionKeyDocumentModel)

            # ========================================
            # 测试 2: 手动创建和管理分区
            # ========================================
            # 手动分区适用于需要完全隔离数据的场景，如不同租户、不同数据类型等
            # 与 Partition Key 不同，手动分区需要显式创建分区并指定插入到哪个分区

            if await provider.index_exists(TestDocumentModel):
                await provider.drop_index(TestDocumentModel)
            await provider.create_index(TestDocumentModel)


            # 向集合中插入一些测试数据
            docs = [
                create_sample_document(
                    f"manual_doc_{i}",
                    np.random.rand(768).tolist(),
                    extras={"partition_test": True}
                )
                for i in range(3)
            ]
            await provider.bulk_insert(TestDocumentModel, docs)

            # Flush 确保数据持久化
            await TestDocumentModel.flush(provider)

            # 在默认分区搜索（不指定 partition_name）
            query_vector = np.random.rand(768).tolist()
            vector_param = VectorSearchParam(vector=query_vector, k=10)
            search_query = SearchQuery(
                vector_param=vector_param,
                include_scores=True
            )
            result = await provider.search(TestDocumentModel, search_query)

            # 验证可以搜索到数据
            assert len(result.documents) > 0
            # 验证返回了分数
            assert len(result.scores) == len(result.documents)

            await provider.drop_index(TestDocumentModel)

        finally:
            try:
                if await provider.index_exists(PartitionKeyDocumentModel):
                    await provider.drop_index(PartitionKeyDocumentModel)
                if await provider.index_exists(TestDocumentModel):
                    await provider.drop_index(TestDocumentModel)
            except Exception:
                pass
            await provider.disconnect()

    async def test_extras_dynamic_and_query(self, milvus_config):
        """测试 extras 字段的动态性和复杂查询（合并用例复用数据）"""
        if should_skip_milvus_test(milvus_config):
            pytest.skip("MILVUS_HOST not configured or is localhost")

        provider = MilvusProvider(
            backend_type=IndexingBackendTypeEnum.milvus,
            config=milvus_config.copy(),
        )

        try:
            await provider.connect()

            if await provider.index_exists(TestDocumentModel):
                await provider.drop_index(TestDocumentModel)
            await provider.create_index(TestDocumentModel)

            # ========================================
            # 场景 1: 测试动态字段的增减和更新
            # ========================================
            doc1 = create_sample_document(
                "doc_001",
                np.random.rand(768).tolist(),
                extras={
                    "field1": "value1",
                    "field2": 100,
                    "field3": True,
                    "nested": {"inner": "data"}
                }
            )
            await provider.insert(doc1)

            doc2 = create_sample_document(
                "doc_002",
                np.random.rand(768).tolist(),
                extras={
                    "field1": "value2",
                    "field4": ["item1", "item2"],
                    "field5": None,
                }
            )
            await provider.insert(doc2)

            # 显式 flush 确保数据持久化
            await TestDocumentModel.flush(provider)

            # 验证动态字段读取
            retrieved_doc1 = await provider.get_by_id(TestDocumentModel, "doc_001")
            assert retrieved_doc1.extras.get("field1") == "value1"
            assert retrieved_doc1.extras.get("field2") == 100

            # 验证动态字段更新
            retrieved_doc1.extras = {"field1": "updated", "new_field": "new_value"}
            await provider.update(retrieved_doc1)

            # 显式 flush 确保数据持久化
            await TestDocumentModel.flush(provider)

            updated_doc = await provider.get_by_id(TestDocumentModel, "doc_001")
            assert updated_doc.extras.get("field1") == "updated"
            assert updated_doc.extras.get("new_field") == "new_value"
            assert updated_doc.extras.get("field2") is None

            # ========================================
            # 场景 2: 复用数据测试复杂查询
            # ========================================
            docs = [
                create_sample_document(
                    "doc_003",
                    np.random.rand(768).tolist(),
                    extras={"category": "tech", "score": 85, "featured": True}
                ),
                create_sample_document(
                    "doc_004",
                    np.random.rand(768).tolist(),
                    extras={"category": "tech", "score": 92, "featured": False}
                ),
                create_sample_document(
                    "doc_005",
                    np.random.rand(768).tolist(),
                    extras={"category": "business", "score": 78, "featured": True}
                ),
                create_sample_document(
                    "doc_006",
                    np.random.rand(768).tolist(),
                    extras={"category": "tech", "score": 88, "featured": True}
                ),
            ]
            await provider.bulk_insert(TestDocumentModel, docs)

            # 显式 flush 确保数据持久化
            await TestDocumentModel.flush(provider)

            # 复杂查询：category == "tech" AND score >= 85 AND featured == True
            query_vector = np.random.rand(768).tolist()
            vector_param = VectorSearchParam(
                vector=query_vector,
                k=10,
                filter=BoolQuery(must=[
                    QueryCondition(field='extras["category"]', value="tech", match_type=MatchType.term),
                    QueryCondition(field='extras["score"]', value=85, match_type=MatchType.range, range_gte=85),
                    QueryCondition(field='extras["featured"]', value=True, match_type=MatchType.term),
                ])
            )
            search_query = SearchQuery(vector_param=vector_param)

            result = await provider.search(TestDocumentModel, search_query)

            # 应该匹配 doc_003 和 doc_006
            assert len(result.documents) == 2
            for doc in result.documents:
                assert doc.extras.get("category") == "tech"
                assert doc.extras.get("score") >= 85
                assert doc.extras.get("featured") is True

        finally:
            try:
                if await provider.index_exists(TestDocumentModel):
                    await provider.drop_index(TestDocumentModel)
            except Exception:
                pass
            await provider.disconnect()

    # ========================================================================
    # 上下文管理器测试
    # ========================================================================

    async def test_context_manager(self, milvus_config):
        """测试异步上下文管理器"""
        if should_skip_milvus_test(milvus_config):
            pytest.skip("MILVUS_HOST not configured or is localhost")

        config = milvus_config.copy()

        async with MilvusProvider(
            backend_type=IndexingBackendTypeEnum.milvus,
            config=config,
        ) as provider:
            assert provider._client is not None
            assert await provider.ping() is True

        # 退出上下文后，连接应该关闭
        assert provider._client is None
