import pytest
from ext.indexing.factory import IndexingProviderFactory


# Skip condition
skip_if_no_milvus = pytest.mark.skipif(
    not pytest.importorskip("os").getenv("MILVUS_HOST"), reason="MILVUS_HOST not set in environment"
)


@pytest.fixture
async def test_milvus_index_model(test_index_model, milvus_config):
    index_model = test_index_model
    index_model.Meta.provider = await IndexingProviderFactory.create(milvus_config)
    return index_model


@pytest.fixture
async def test_milvus_index_model_with_partition(test_index_model_with_partition, milvus_config):
    index_model = test_index_model_with_partition
    index_model.Meta.provider = await IndexingProviderFactory.create(milvus_config)
    return index_model


@skip_if_no_milvus
class TestMilvusBaseIndexModel:
    """测试 Milvus 的 BaseIndexModel 方法（需要真实后端连接）"""

    @pytest.mark.asyncio
    async def test_create_schema(self, test_milvus_index_model):
        """测试创建 schema"""
        await test_milvus_index_model.create_schema(drop_existing=False)

    @pytest.mark.asyncio
    async def test_create_schema_with_drop(self, test_milvus_index_model):
        """测试创建 schema（drop_existing=True）"""
        await test_milvus_index_model.create_schema(drop_existing=True)

    @pytest.mark.asyncio
    async def test_drop_schema(self, test_milvus_index_model):
        """测试删除 schema"""
        await test_milvus_index_model.create_schema(drop_existing=True)
        await test_milvus_index_model.drop_schema()

    @pytest.mark.asyncio
    async def test_save(self, test_milvus_index_model):
        """测试保存单个文档"""
        doc = test_milvus_index_model(
            id="test_doc_1", title="Test Document", content="Test content", category="test", embedding=[0.1] * 1536
        )
        await doc.save()
        assert doc.id == "test_doc_1"

    @pytest.mark.asyncio
    async def test_save_with_auto_generate_id(self, test_milvus_index_model):
        """测试保存文档（自动生成 ID）"""
        test_milvus_index_model.Meta.auto_generate_id = True
        doc = test_milvus_index_model(
            id="", title="Auto ID Document", content="Auto ID content", category="test", embedding=[0.1] * 1536
        )
        await doc.save()
        assert doc.id != ""
        assert len(doc.id) > 0
        test_milvus_index_model.Meta.auto_generate_id = False

    @pytest.mark.asyncio
    async def test_get(self, test_milvus_index_model):
        """测试通过 ID 获取文档"""
        doc = test_milvus_index_model(
            id="get_test_1", title="Get Test", content="Get test content", category="test", embedding=[0.2] * 1536
        )
        await doc.save()

        retrieved_doc = await test_milvus_index_model.get("get_test_1")
        assert retrieved_doc is not None
        assert retrieved_doc.id == "get_test_1"
        assert retrieved_doc.title == "Get Test"

    @pytest.mark.asyncio
    async def test_get_not_found(self, test_milvus_index_model):
        """测试获取不存在的文档"""
        result = await test_milvus_index_model.get("nonexistent_id")
        assert result is None

    @pytest.mark.asyncio
    async def test_filter_simple(self, test_milvus_index_model):
        """测试简单过滤查询"""
        docs = [
            test_milvus_index_model(
                id=f"filter_{i}",
                title=f"Doc {i}",
                content=f"Content {i}",
                category="tech" if i < 2 else "other",
                embedding=[0.1 * i] * 1536,
            )
            for i in range(5)
        ]
        for doc in docs:
            await doc.save()

        from ext.indexing.types import FilterClause

        results = await test_milvus_index_model.filter(
            filter_clause=FilterClause(equals={"category": "tech"}), limit=10
        )
        assert len(results) >= 2

    @pytest.mark.asyncio
    async def test_filter_with_pagination(self, test_milvus_index_model):
        """测试过滤查询（带分页）"""
        docs = [
            test_milvus_index_model(
                id=f"page_{i}",
                title=f"Page Doc {i}",
                content=f"Content {i}",
                category="page_test",
                embedding=[0.1 * i] * 1536,
            )
            for i in range(10)
        ]
        for doc in docs:
            await doc.save()

        from ext.indexing.types import FilterClause

        page1 = await test_milvus_index_model.filter(
            filter_clause=FilterClause(equals={"category": "page_test"}), limit=3, offset=0
        )
        page2 = await test_milvus_index_model.filter(
            filter_clause=FilterClause(equals={"category": "page_test"}), limit=3, offset=3
        )
        assert len(page1) == 3
        assert len(page2) == 3

    @pytest.mark.asyncio
    async def test_filter_with_sort(self, test_milvus_index_model):
        """测试过滤查询（带排序）"""
        docs = [
            test_milvus_index_model(
                id=f"sort_{i}",
                title=f"Sort {10 - i}",
                content=f"Content {i}",
                category="sort_test",
                embedding=[0.1 * i] * 1536,
            )
            for i in range(5)
        ]
        for doc in docs:
            await doc.save()

        from ext.indexing.types import FilterClause

        results = await test_milvus_index_model.filter(
            filter_clause=FilterClause(equals={"category": "sort_test"}), limit=10, sort="title:asc"
        )
        assert len(results) >= 5
        assert results[0].title.startswith("Sort 1")

    @pytest.mark.asyncio
    async def test_search_dense(self, test_milvus_index_model, sample_query_vector):
        """测试稠密向量搜索"""
        from ext.indexing.types import DenseSearchClause

        query_clause = DenseSearchClause(vector=sample_query_vector, top_k=5, metric="cosine")
        results = await test_milvus_index_model.search(query_clause=query_clause, limit=5)

        assert isinstance(results, list)
        assert all(isinstance(result, tuple) and len(result) == 2 for result in results)
        assert all(isinstance(doc, test_milvus_index_model) and isinstance(score, float) for doc, score in results)

    @pytest.mark.asyncio
    async def test_search_with_filter(self, test_milvus_index_model, sample_query_vector):
        """测试搜索（带过滤条件）"""
        from ext.indexing.types import DenseSearchClause, FilterClause

        query_clause = DenseSearchClause(vector=sample_query_vector, top_k=5)
        filter_clause = FilterClause(equals={"category": "tech"})
        results = await test_milvus_index_model.search(query_clause=query_clause, filter_clause=filter_clause, limit=5)

        assert isinstance(results, list)
        for doc, score in results:
            assert doc.category == "tech"

    @pytest.mark.asyncio
    async def test_search_cursor(self, test_milvus_index_model, sample_query_vector):
        """测试搜索（游标分页）"""
        from ext.indexing.types import DenseSearchClause

        query_clause = DenseSearchClause(vector=sample_query_vector, top_k=10)
        cursor = await test_milvus_index_model.search_cursor(query_clause=query_clause, page_size=3)

        assert cursor.results is not None
        assert len(cursor.results) <= 3
        assert isinstance(cursor.next_cursor, (str, type(None)))

    @pytest.mark.asyncio
    async def test_bulk_insert(self, test_milvus_index_model):
        """测试批量插入"""
        docs = [
            test_milvus_index_model(
                id=f"bulk_{i}", title=f"Bulk {i}", content=f"Content {i}", category="bulk", embedding=[0.1 * i] * 1536
            )
            for i in range(20)
        ]
        await test_milvus_index_model.bulk_insert(docs, batch_size=5, concurrent_batches=2)

    @pytest.mark.asyncio
    async def test_bulk_update(self, test_milvus_index_model):
        """测试批量更新"""
        docs = [
            test_milvus_index_model(
                id=f"update_{i}",
                title=f"Update {i}",
                content=f"Content {i}",
                category="update",
                embedding=[0.1 * i] * 1536,
            )
            for i in range(5)
        ]
        await test_milvus_index_model.bulk_insert(docs)

        for doc in docs:
            doc.title = f"Updated {doc.id}"
        await test_milvus_index_model.bulk_update(docs)

        updated_doc = await test_milvus_index_model.get("update_0")
        assert updated_doc.title == "Updated update_0"

    @pytest.mark.asyncio
    async def test_bulk_upsert(self, test_milvus_index_model):
        """测试批量 upsert"""
        docs = [
            test_milvus_index_model(
                id=f"upsert_{i}",
                title=f"Upsert {i}",
                content=f"Content {i}",
                category="upsert",
                embedding=[0.1 * i] * 1536,
            )
            for i in range(5)
        ]

        await test_milvus_index_model.bulk_upsert(docs)

        docs[0].title = "Updated Upsert 0"
        await test_milvus_index_model.bulk_upsert(docs)

        doc = await test_milvus_index_model.get("upsert_0")
        assert doc.title == "Updated Upsert 0"

    @pytest.mark.asyncio
    async def test_bulk_delete(self, test_milvus_index_model):
        """测试批量删除"""
        docs = [
            test_milvus_index_model(
                id=f"delete_{i}",
                title=f"Delete {i}",
                content=f"Content {i}",
                category="delete",
                embedding=[0.1 * i] * 1536,
            )
            for i in range(5)
        ]
        await test_milvus_index_model.bulk_insert(docs)

        ids_to_delete = [f"delete_{i}" for i in range(3)]
        await test_milvus_index_model.bulk_delete(ids_to_delete)

        doc = await test_milvus_index_model.get("delete_0")
        assert doc is None
        doc = await test_milvus_index_model.get("delete_3")
        assert doc is not None

    @pytest.mark.asyncio
    async def test_delete_by_query(self, test_milvus_index_model):
        """测试根据条件删除"""
        docs = [
            test_milvus_index_model(
                id=f"del_query_{i}",
                title=f"DelQuery {i}",
                content=f"Content {i}",
                category="del_target" if i < 3 else "keep",
                embedding=[0.1 * i] * 1536,
            )
            for i in range(6)
        ]
        await test_milvus_index_model.bulk_insert(docs)

        from ext.indexing.types import FilterClause

        filter_clause = FilterClause(equals={"category": "del_target"})
        await test_milvus_index_model.delete_by_query(filter_clause)

        doc = await test_milvus_index_model.get("del_query_0")
        assert doc is None
        doc = await test_milvus_index_model.get("del_query_3")
        assert doc is not None

    @pytest.mark.asyncio
    async def test_count(self, test_milvus_index_model):
        """测试统计文档数量"""
        docs = [
            test_milvus_index_model(
                id=f"count_{i}",
                title=f"Count {i}",
                content=f"Content {i}",
                category="count",
                embedding=[0.1 * i] * 1536,
            )
            for i in range(5)
        ]
        await test_milvus_index_model.bulk_insert(docs)

        count = await test_milvus_index_model.count()
        assert count >= 5

        from ext.indexing.types import FilterClause

        filter_clause = FilterClause(equals={"category": "count"})
        filtered_count = await test_milvus_index_model.count(filter_clause=filter_clause)
        assert filtered_count >= 5

    @pytest.mark.asyncio
    async def test_exists(self, test_milvus_index_model):
        """测试检查文档是否存在"""
        doc = test_milvus_index_model(
            id="exists_test",
            title="Exists Test",
            content="Exists test content",
            category="exists",
            embedding=[0.5] * 1536,
        )
        await doc.save()

        exists = await test_milvus_index_model.exists("exists_test")
        assert exists is True

        not_exists = await test_milvus_index_model.exists("nonexistent_exists")
        assert not_exists is False
