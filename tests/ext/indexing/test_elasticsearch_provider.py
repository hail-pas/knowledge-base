"""Elasticsearch Provider 单元测试"""
from typing import List
from datetime import datetime

import pytest
import numpy as np

from ext.indexing.providers.elasticsearch import ElasticsearchProvider
from ext.indexing.base import (
    FieldDefinition,
    FieldType,
    SearchQuery,
    QueryCondition,
    MatchType,
    BoolQuery,
    DenseIndexModel,
)
from ext.ext_tortoise.enums import IndexingBackendTypeEnum

# 从 conftest 导入共享的 fixtures
from tests.ext.indexing.conftest import (
    es_config,
    sample_embedding,
    should_skip_es_test,
)


# ============================================================================
# 测试用的索引模型类
# ============================================================================

class TestDocumentModel(DenseIndexModel):
    """Elasticsearch 测试文档模型"""
    id: str
    title: str
    content: str = ""
    author: str = ""
    tags: List[str] = []
    category: str = ""
    created_at: datetime | None = None
    embedding: List[float] = []
    extras: dict = {}

    @classmethod
    def get_index_name(cls) -> str:
        return "test_documents_es"

    @classmethod
    def get_index_type(cls) -> str:
        return "dense"

    @classmethod
    def get_backend_type(cls):
        return IndexingBackendTypeEnum.elasticsearch

    @classmethod
    def get_field_definitions(cls) -> List[FieldDefinition]:
        return [
            FieldDefinition(name="id", type=FieldType.keyword, is_primary_key=True),
            FieldDefinition(name="title", type=FieldType.text, analyzer="standard"),
            FieldDefinition(name="content", type=FieldType.text, analyzer="standard"),
            FieldDefinition(name="author", type=FieldType.keyword),
            FieldDefinition(name="category", type=FieldType.keyword),
            FieldDefinition(name="tags", type=FieldType.keyword),
            FieldDefinition(name="created_at", type=FieldType.datetime),
            FieldDefinition(
                name="embedding",
                type=FieldType.dense_vector,
                dimension=768,
                metric_type="cosine"
            ),
            FieldDefinition(
                name="extras",
                type=FieldType.json,
                description="动态 JSON 字段，用于存储额外的元数据"
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
        tags=["test", "elasticsearch"],
        created_at=datetime.now(),
        embedding=embedding,
        extras=extras or {"version": "1.0"},
    )


def create_multiple_documents(count: int, embedding_dim: int = 768) -> List[TestDocumentModel]:
    """创建多个测试文档"""
    titles = [
        "Machine Learning Basics",
        "Deep Learning Tutorial",
        "Python Programming",
        "Data Science Guide",
        "AI Applications",
    ]
    contents = [
        "Introduction to machine learning algorithms and concepts",
        "Deep learning neural networks explained with examples",
        "Python programming for data science and analysis",
        "Complete guide to data science techniques and tools",
        "Artificial intelligence applications in various industries",
    ]
    authors = ["author_a", "author_b", "author_a", "author_b", "author_a"]
    categories = ["ai", "ai", "programming", "data", "ai"]
    tags_list = [
        ["ml", "ai", "tutorial"],
        ["dl", "neural", "networks"],
        ["python", "programming", "data"],
        ["data", "science", "analysis"],
        ["ai", "applications", "industry"],
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
class TestElasticsearchProvider:
    """Elasticsearch Provider 测试类"""

    # ========================================================================
    # 初始化测试
    # ========================================================================

    async def test_init_success(self, es_config):
        """测试成功初始化 ElasticsearchProvider"""
        provider = ElasticsearchProvider(
            backend_type=IndexingBackendTypeEnum.elasticsearch,
            config=es_config.copy(),
        )

        assert provider.host == es_config["host"]
        assert provider.port == es_config["port"]
        assert provider.secure == es_config["secure"]

        await provider.disconnect()

    async def test_init_missing_host(self):
        """测试缺少 host 参数时抛出异常"""
        config = {}  # 缺少必需的 host

        with pytest.raises(KeyError):
            ElasticsearchProvider(
                backend_type=IndexingBackendTypeEnum.elasticsearch,
                config=config,
            )

    # ========================================================================
    # 连接测试
    # ========================================================================

    async def test_connect_success(self, es_config):
        """测试成功连接到 Elasticsearch"""
        if should_skip_es_test(es_config):
            pytest.skip("ES_HOST not configured or is localhost")

        provider = ElasticsearchProvider(
            backend_type=IndexingBackendTypeEnum.elasticsearch,
            config=es_config.copy(),
        )

        await provider.connect()
        assert provider._client is not None

        await provider.disconnect()

    async def test_connect_invalid_host(self):
        """测试连接到无效的主机"""
        config = {
            "host": "invalid-host-that-does-not-exist.example.com",
            "port": 9200,
            "timeout": 5,  # 短超时
        }

        provider = ElasticsearchProvider(
            backend_type=IndexingBackendTypeEnum.elasticsearch,
            config=config,
        )

        try:
            with pytest.raises(Exception, match="Failed to connect"):
                await provider.connect()
        finally:
            await provider.disconnect()

    async def test_ping(self, es_config):
        """测试 ping 方法"""
        if should_skip_es_test(es_config):
            pytest.skip("ES_HOST not configured or is localhost")

        provider = ElasticsearchProvider(
            backend_type=IndexingBackendTypeEnum.elasticsearch,
            config=es_config.copy(),
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

    async def test_create_and_drop_index(self, es_config):
        """测试创建和删除索引"""
        if should_skip_es_test(es_config):
            pytest.skip("ES_HOST not configured or is localhost")

        provider = ElasticsearchProvider(
            backend_type=IndexingBackendTypeEnum.elasticsearch,
            config=es_config.copy(),
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

            # 删除不存在的索引应该也返回 True（ignore_unavailable）
            assert await provider.drop_index(TestDocumentModel) is True
        finally:
            try:
                if await provider.index_exists(TestDocumentModel):
                    await provider.drop_index(TestDocumentModel)
            except Exception:
                pass
            await provider.disconnect()

    # ========================================================================
    # 文档操作测试
    # ========================================================================

    async def test_insert_and_get_document(self, es_config, sample_embedding):
        """测试插入和获取文档"""
        if should_skip_es_test(es_config):
            pytest.skip("ES_HOST not configured or is localhost")

        provider = ElasticsearchProvider(
            backend_type=IndexingBackendTypeEnum.elasticsearch,
            config=es_config.copy(),
        )

        try:
            await provider.connect()

            # 创建测试索引
            if await provider.index_exists(TestDocumentModel):
                await provider.drop_index(TestDocumentModel)
            await provider.create_index(TestDocumentModel)

            # 插入文档
            doc = create_sample_document(
                "doc_001",
                sample_embedding,
                extras={"special_attr": "special", "priority": 1}
            )
            assert await provider.insert(doc) is True

            # 获取文档
            retrieved_doc = await provider.get_by_id(TestDocumentModel, "doc_001")
            assert retrieved_doc is not None
            assert retrieved_doc.id == "doc_001"
            assert retrieved_doc.title == doc.title
            assert retrieved_doc.content == doc.content
            assert retrieved_doc.author == doc.author

            # 验证 extras 字段
            assert retrieved_doc.extras is not None
            assert retrieved_doc.extras.get("special_attr") == "special"
            assert retrieved_doc.extras.get("priority") == 1
        finally:
            try:
                if await provider.index_exists(TestDocumentModel):
                    await provider.drop_index(TestDocumentModel)
            except Exception:
                pass
            await provider.disconnect()

    async def test_update_document(self, es_config, sample_embedding):
        """测试更新文档"""
        if should_skip_es_test(es_config):
            pytest.skip("ES_HOST not configured or is localhost")

        provider = ElasticsearchProvider(
            backend_type=IndexingBackendTypeEnum.elasticsearch,
            config=es_config.copy(),
        )

        try:
            await provider.connect()

            # 创建测试索引
            if await provider.index_exists(TestDocumentModel):
                await provider.drop_index(TestDocumentModel)
            await provider.create_index(TestDocumentModel)

            # 插入文档
            doc = create_sample_document(
                "doc_001",
                sample_embedding,
                extras={"priority": 1}
            )
            await provider.insert(doc)

            # 更新文档
            doc.title = "Updated Title"
            doc.content = "Updated content"
            # 更新 extras 字段，添加新字段
            doc.extras = {
                "priority": 2,
                "special_attr": "updated",
                "new_field": "new_value",
            }
            assert await provider.update(doc) is True

            # 验证更新
            retrieved_doc = await provider.get_by_id(TestDocumentModel, "doc_001")
            assert retrieved_doc.title == "Updated Title"
            assert retrieved_doc.content == "Updated content"
            # 验证 extras 更新
            assert retrieved_doc.extras.get("priority") == 2
            assert retrieved_doc.extras.get("special_attr") == "updated"
            assert retrieved_doc.extras.get("new_field") == "new_value"
        finally:
            try:
                if await provider.index_exists(TestDocumentModel):
                    await provider.drop_index(TestDocumentModel)
            except Exception:
                pass
            await provider.disconnect()

    async def test_upsert_document(self, es_config, sample_embedding):
        """测试 upsert（插入或更新）文档"""
        if should_skip_es_test(es_config):
            pytest.skip("ES_HOST not configured or is localhost")

        provider = ElasticsearchProvider(
            backend_type=IndexingBackendTypeEnum.elasticsearch,
            config=es_config.copy(),
        )

        try:
            await provider.connect()

            # 创建测试索引
            if await provider.index_exists(TestDocumentModel):
                await provider.drop_index(TestDocumentModel)
            await provider.create_index(TestDocumentModel)

            # Upsert 新文档
            doc = create_sample_document(
                "doc_001",
                sample_embedding,
                extras={"special_attr": "test"}
            )
            assert await provider.upsert(doc) is True
            retrieved_doc = await provider.get_by_id(TestDocumentModel, "doc_001")
            assert retrieved_doc.title == f"Test Document doc_001"

            # Upsert 更新现有文档
            doc.title = "Upserted Title"
            doc.extras = {"special_attr": "updated", "new_field": "value"}
            assert await provider.upsert(doc) is True
            retrieved_doc = await provider.get_by_id(TestDocumentModel, "doc_001")
            assert retrieved_doc.title == "Upserted Title"
            assert retrieved_doc.extras.get("special_attr") == "updated"
        finally:
            try:
                if await provider.index_exists(TestDocumentModel):
                    await provider.drop_index(TestDocumentModel)
            except Exception:
                pass
            await provider.disconnect()

    async def test_bulk_insert(self, es_config):
        """测试批量插入文档"""
        if should_skip_es_test(es_config):
            pytest.skip("ES_HOST not configured or is localhost")

        provider = ElasticsearchProvider(
            backend_type=IndexingBackendTypeEnum.elasticsearch,
            config=es_config.copy(),
        )

        try:
            await provider.connect()

            # 创建测试索引
            if await provider.index_exists(TestDocumentModel):
                await provider.drop_index(TestDocumentModel)
            await provider.create_index(TestDocumentModel)

            # 创建批量文档
            documents = create_multiple_documents(10)

            # 批量插入
            count = await provider.bulk_insert(TestDocumentModel, documents)
            assert count == 10

            # 验证插入的文档
            for i, doc in enumerate(documents):
                retrieved_doc = await provider.get_by_id(TestDocumentModel, doc.id)
                assert retrieved_doc is not None
                assert retrieved_doc.title == doc.title
                # 验证 extras 字段
                assert retrieved_doc.extras is not None
                assert "version" in retrieved_doc.extras
                assert "priority" in retrieved_doc.extras
        finally:
            try:
                if await provider.index_exists(TestDocumentModel):
                    await provider.drop_index(TestDocumentModel)
            except Exception:
                pass
            await provider.disconnect()

    async def test_delete_document(self, es_config, sample_embedding):
        """测试删除文档"""
        if should_skip_es_test(es_config):
            pytest.skip("ES_HOST not configured or is localhost")

        provider = ElasticsearchProvider(
            backend_type=IndexingBackendTypeEnum.elasticsearch,
            config=es_config.copy(),
        )

        try:
            await provider.connect()

            # 创建测试索引
            if await provider.index_exists(TestDocumentModel):
                await provider.drop_index(TestDocumentModel)
            await provider.create_index(TestDocumentModel)

            # 插入文档
            doc = create_sample_document("doc_001", sample_embedding)
            await provider.insert(doc)

            # 删除文档
            count = await provider.delete(TestDocumentModel, doc_id="doc_001")
            assert count == 1

            # 验证删除
            retrieved_doc = await provider.get_by_id(TestDocumentModel, "doc_001")
            assert retrieved_doc is None

            # 删除不存在的文档
            count = await provider.delete(TestDocumentModel, doc_id="non_existent")
            assert count == 0
        finally:
            try:
                if await provider.index_exists(TestDocumentModel):
                    await provider.drop_index(TestDocumentModel)
            except Exception:
                pass
            await provider.disconnect()

    # ========================================================================
    # 搜索操作测试
    # ========================================================================

    async def test_search_text_query(self, es_config):
        """测试文本搜索"""
        if should_skip_es_test(es_config):
            pytest.skip("ES_HOST not configured or is localhost")

        provider = ElasticsearchProvider(
            backend_type=IndexingBackendTypeEnum.elasticsearch,
            config=es_config.copy(),
        )

        try:
            await provider.connect()

            # 创建测试索引
            if await provider.index_exists(TestDocumentModel):
                await provider.drop_index(TestDocumentModel)
            await provider.create_index(TestDocumentModel)

            # 批量插入文档
            documents = create_multiple_documents(5)
            await provider.bulk_insert(TestDocumentModel, documents)

            # 搜索 "machine"
            query = SearchQuery(query="machine", limit=10)
            result = await provider.search(TestDocumentModel, query)

            assert len(result.documents) > 0
            assert result.total > 0
            assert "machine" in result.documents[0].title.lower() or "machine" in result.documents[0].content.lower()

            # 搜索 "python"
            query = SearchQuery(query="python", limit=10)
            result = await provider.search(TestDocumentModel, query)

            assert len(result.documents) > 0
            assert "python" in result.documents[0].title.lower() or "python" in result.documents[0].content.lower()
        finally:
            try:
                if await provider.index_exists(TestDocumentModel):
                    await provider.drop_index(TestDocumentModel)
            except Exception:
                pass
            await provider.disconnect()

    async def test_search_bool_query(self, es_config):
        """测试布尔查询"""
        if should_skip_es_test(es_config):
            pytest.skip("ES_HOST not configured or is localhost")

        provider = ElasticsearchProvider(
            backend_type=IndexingBackendTypeEnum.elasticsearch,
            config=es_config.copy(),
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

            # 布尔查询：author = author_a AND category = ai
            bool_query = BoolQuery(
                must=[
                    QueryCondition(field="author", value="author_a", match_type=MatchType.term),
                    QueryCondition(field="category", value="ai", match_type=MatchType.term),
                ]
            )

            query = SearchQuery(bool_query=bool_query, limit=10)
            result = await provider.search(TestDocumentModel, query)

            assert len(result.documents) > 0
            for doc in result.documents:
                assert doc.author == "author_a"
                assert doc.category == "ai"
        finally:
            try:
                if await provider.index_exists(TestDocumentModel):
                    await provider.drop_index(TestDocumentModel)
            except Exception:
                pass
            await provider.disconnect()

    async def test_count_documents(self, es_config):
        """测试统计文档数量"""
        if should_skip_es_test(es_config):
            pytest.skip("ES_HOST not configured or is localhost")

        provider = ElasticsearchProvider(
            backend_type=IndexingBackendTypeEnum.elasticsearch,
            config=es_config.copy(),
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

            # 统计所有文档
            total = await provider.count(TestDocumentModel)
            assert total == 10

            # 统计特定作者的文档
            bool_query = BoolQuery(
                must=[QueryCondition(field="author", value="author_a", match_type=MatchType.term)]
            )
            count = await provider.count(TestDocumentModel, query=bool_query)
            assert count > 0
        finally:
            try:
                if await provider.index_exists(TestDocumentModel):
                    await provider.drop_index(TestDocumentModel)
            except Exception:
                pass
            await provider.disconnect()

    # ========================================================================
    # extras 字段测试
    # ========================================================================

    async def test_extras_field_search(self, es_config):
        """测试基于 extras JSON 字段的搜索"""
        if should_skip_es_test(es_config):
            pytest.skip("ES_HOST not configured or is localhost")

        provider = ElasticsearchProvider(
            backend_type=IndexingBackendTypeEnum.elasticsearch,
            config=es_config.copy(),
        )

        try:
            await provider.connect()

            # 创建测试索引
            if await provider.index_exists(TestDocumentModel):
                await provider.drop_index(TestDocumentModel)
            await provider.create_index(TestDocumentModel)

            # 插入不同 special_attr 值的文档
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
            ]
            await provider.bulk_insert(TestDocumentModel, docs)

            # 搜索 extras.special_attr == "special"
            # 注意：在 Elasticsearch 中，JSON 字段使用点号语法访问
            bool_query = BoolQuery(
                must=[
                    QueryCondition(
                        field="special_attr",
                        value="special",
                        match_type=MatchType.term
                    )
                ]
            )
            query = SearchQuery(bool_query=bool_query, limit=10)
            result = await provider.search(TestDocumentModel, query)

            # 应该返回 special_attr == "special" 的文档
            assert len(result.documents) == 2
            for doc in result.documents:
                assert doc.extras.get("special_attr") == "special"
        finally:
            try:
                if await provider.index_exists(TestDocumentModel):
                    await provider.drop_index(TestDocumentModel)
            except Exception:
                pass
            await provider.disconnect()


    async def test_extras_dynamic_fields(self, es_config, sample_embedding):
        """测试 extras 字段的动态增减"""
        if should_skip_es_test(es_config):
            pytest.skip("ES_HOST not configured or is localhost")

        provider = ElasticsearchProvider(
            backend_type=IndexingBackendTypeEnum.elasticsearch,
            config=es_config.copy(),
        )

        try:
            await provider.connect()

            # 创建测试索引
            if await provider.index_exists(TestDocumentModel):
                await provider.drop_index(TestDocumentModel)
            await provider.create_index(TestDocumentModel)

            # 插入文档，extras 包含多个字段
            doc1 = create_sample_document(
                "doc_001",
                sample_embedding,
                extras={
                    "field1": "value1",
                    "field2": 100,
                    "field3": True,
                    "nested": {"inner": "data"}
                }
            )
            await provider.insert(doc1)

            # 插入另一个文档，extras 包含不同的字段
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

            # 获取文档并验证 extras
            retrieved_doc1 = await provider.get_by_id(TestDocumentModel, "doc_001")
            assert retrieved_doc1.extras is not None
            assert retrieved_doc1.extras.get("field1") == "value1"
            assert retrieved_doc1.extras.get("field2") == 100
            assert retrieved_doc1.extras.get("field3") is True
            assert retrieved_doc1.extras.get("nested") == {"inner": "data"}

            retrieved_doc2 = await provider.get_by_id(TestDocumentModel, "doc_002")
            assert retrieved_doc2.extras is not None
            assert retrieved_doc2.extras.get("field1") == "value2"
            assert retrieved_doc2.extras.get("field4") == ["item1", "item2"]
            assert retrieved_doc2.extras.get("field5") is None

            # 更新文档，修改 extras 字段
            retrieved_doc1.extras = {
                "field1": "updated",
                "new_field": "new_value"
            }
            await provider.update(retrieved_doc1)

            updated_doc = await provider.get_by_id(TestDocumentModel, "doc_001")
            assert updated_doc.extras.get("field1") == "updated"
            assert updated_doc.extras.get("new_field") == "new_value"
            # 原来的字段应该不存在了（更新是替换整个对象）
            assert updated_doc.extras.get("field2") is None
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

    async def test_context_manager(self, es_config):
        """测试异步上下文管理器"""
        if should_skip_es_test(es_config):
            pytest.skip("ES_HOST not configured or is localhost")

        config = es_config.copy()
        provider = None

        try:
            async with ElasticsearchProvider(
                backend_type=IndexingBackendTypeEnum.elasticsearch,
                config=config,
            ) as p:
                provider = p
                assert provider._client is not None
                assert await provider.ping() is True

            # 退出上下文后，连接应该关闭
            assert provider._client is None
        finally:
            # 确保清理资源
            if provider and provider._client is not None:
                try:
                    await provider.disconnect()
                except Exception:
                    pass
