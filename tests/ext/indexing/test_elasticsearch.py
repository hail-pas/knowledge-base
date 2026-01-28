import pytest
from ext.indexing.factory import IndexingProviderFactory


# Skip condition
skip_if_no_es = pytest.mark.skipif(
    not pytest.importorskip("os").getenv("ES_HOST"), reason="ES_HOST not set in environment"
)


@pytest.fixture
async def test_es_index_model(test_index_model, elasticsearch_config):
    index_model = test_index_model
    index_model.Meta.provider = await IndexingProviderFactory.create(elasticsearch_config, use_cache=False)
    yield index_model
    await index_model.Meta.provider.disconnect()


@pytest.fixture
async def test_es_index_model_with_partition(test_index_model_with_partition, elasticsearch_config):
    index_model = test_index_model_with_partition
    index_model.Meta.provider = await IndexingProviderFactory.create(elasticsearch_config, use_cache=False)
    yield index_model
    await index_model.Meta.provider.disconnect()


@skip_if_no_es
class TestElasticSearchBaseIndexModel:
    """测试 Elasticsearch 的 BaseIndexModel 方法（需要真实后端连接）"""

    @pytest.mark.asyncio
    async def test_create_schema(self, test_es_index_model):
        await test_es_index_model.drop_schema()
        await test_es_index_model.create_schema(drop_existing=False)
        await test_es_index_model.drop_schema()
        await test_es_index_model.create_schema(drop_existing=True)

    @pytest.mark.asyncio
    async def test_save_and_get(self, test_es_index_model):
        """测试保存单个文档"""
        doc = test_es_index_model(
            title="Test Document", content="Test content", category=["test"], embedding=[0.1] * 1536
        )
        assert not doc.id
        await doc.save()
        assert doc.id

        docs = await test_es_index_model.get([doc.id])

        assert len(docs) == 1
        assert docs[0].id == doc.id
        assert docs[0].title == doc.title
        assert docs[0].content == doc.content
        assert docs[0].category == doc.category

        result = await test_es_index_model.get(["nonexistent_id"])
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_filter_simple(self, test_es_index_model):
        """测试简单过滤查询"""
        docs = [
            test_es_index_model(
                title=f"Doc {i}",
                content=f"Content {i}",
                category=["tech"] if i <= 2 else ["other"],
                embedding=[0.1 * i] * 1536,
            )
            for i in range(1, 5)
        ]
        for doc in docs:
            await doc.save()

        from ext.indexing.types import FilterClause

        results = await test_es_index_model.filter(filter_clause=FilterClause(equals={"category": "tech"}), limit=10)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_filter_with_pagination(self, test_es_index_model):
        """测试过滤查询（带分页）"""
        docs = [
            test_es_index_model(
                id=f"page_{i}",
                title=f"Page Doc {i}",
                content=f"Content {i}",
                category=["page_test"],
                embedding=[0.1 * i] * 1536,
            )
            for i in range(1, 11)
        ]
        for doc in docs:
            await doc.save()

        from ext.indexing.types import FilterClause

        page1 = await test_es_index_model.filter(
            filter_clause=FilterClause(equals={"category": "page_test"}), limit=3, offset=0
        )
        page2 = await test_es_index_model.filter(
            filter_clause=FilterClause(equals={"category": "page_test"}), limit=3, offset=3
        )
        assert len(page1) == 3
        assert len(page2) == 3

    @pytest.mark.asyncio
    async def test_filter_with_sort(self, test_es_index_model):
        """测试过滤查询（带排序）"""
        docs = [
            test_es_index_model(
                title="Sort",
                content=f"Content {i}",
                category=[f"sort_test {10 - i}"],
                embedding=[0.1 * i] * 1536,
            )
            for i in range(1, 6)
        ]
        for doc in docs:
            await doc.save()

        from ext.indexing.types import FilterClause

        results = await test_es_index_model.filter(
            filter_clause=FilterClause(equals={"title.keyword": "Sort"}), limit=10, sort="category:asc"
        )
        assert len(results) >= 5  # i=0 has zero vector, fails to insert
        assert results[0].category[0].startswith("sort_test 5")

    @pytest.mark.asyncio
    async def test_search_dense(self, test_es_index_model, sample_query_vector):
        """测试稠密向量搜索"""
        from ext.indexing.types import DenseSearchClause

        query_clause = DenseSearchClause(vector=sample_query_vector, top_k=5, metric="cosine")
        results = await test_es_index_model.search(query_clause=query_clause, limit=5)

        assert len(results) != 0
        assert isinstance(results, list)
        assert all(isinstance(result, tuple) and len(result) == 2 for result in results)
        assert all(isinstance(doc, test_es_index_model) and isinstance(score, float) for doc, score in results)

    @pytest.mark.asyncio
    async def test_search_with_filter(self, test_es_index_model, sample_query_vector):
        """测试搜索（带过滤条件）"""
        from ext.indexing.types import DenseSearchClause, FilterClause

        query_clause = DenseSearchClause(vector=sample_query_vector, top_k=5)
        filter_clause = FilterClause(equals={"category": "tech"})
        results = await test_es_index_model.search(query_clause=query_clause, filter_clause=filter_clause, limit=5)

        assert isinstance(results, list)
        for doc, score in results:
            assert doc.category == ["tech"]

    @pytest.mark.asyncio
    async def test_search_cursor(self, test_es_index_model, sample_query_vector):
        """测试搜索（游标分页）"""
        from ext.indexing.types import DenseSearchClause

        query_clause = DenseSearchClause(vector=sample_query_vector, top_k=10)
        cursor = await test_es_index_model.search_cursor(query_clause=query_clause, page_size=3)

        assert cursor.results is not None
        assert len(cursor.results) <= 3
        assert isinstance(cursor.next_cursor, (str, type(None)))

    @pytest.mark.asyncio
    async def test_bulk_insert(self, test_es_index_model):
        """测试批量插入"""
        docs = [
            test_es_index_model(
                title=f"Bulk {i}", content=f"Content {i}", category=["bulk"], embedding=[0.1 * i] * 1536
            )
            for i in range(1, 21)
        ]

        from ext.indexing.types import FilterClause

        await test_es_index_model.bulk_insert(docs, batch_size=5, concurrent_batches=2)
        assert (
            len(await test_es_index_model.filter(filter_clause=FilterClause(equals={"category": "bulk"}), limit=30))
            == 20
        )

    @pytest.mark.asyncio
    async def test_bulk_update(self, test_es_index_model):
        """测试批量更新"""
        docs = [
            test_es_index_model(
                title=f"title {i}",
                content=f"Content {i}",
                category=["update"],
                embedding=[0.1 * i] * 1536,
            )
            for i in range(1, 6)
        ]
        await test_es_index_model.bulk_insert(docs)

        for doc in docs:
            doc.title = f"Updated {doc.id}"
        await test_es_index_model.bulk_update(docs)

        assert (await test_es_index_model.get([docs[0].id]))[0].title.startswith("Updated")

    @pytest.mark.asyncio
    async def test_bulk_upsert(self, test_es_index_model):
        """测试批量 upsert"""
        docs = [
            test_es_index_model(
                title=f"Upsert {i}",
                content=f"Content {i}",
                category=["upsert"],
                embedding=[0.1 * i] * 1536,
            )
            for i in range(1, 6)
        ]

        await test_es_index_model.bulk_upsert(docs)

        docs[0].title = "Updated Upsert 0"
        await test_es_index_model.bulk_upsert(docs)

        docs = await test_es_index_model.get([docs[0].id])
        assert docs[0].title == "Updated Upsert 0"

    @pytest.mark.asyncio
    async def test_bulk_delete(self, test_es_index_model):
        """测试批量删除"""
        docs = [
            test_es_index_model(
                title=f"Delete {i}",
                content=f"Content {i}",
                category=["delete"],
                embedding=[0.1 * i] * 1536,
            )
            for i in range(1, 6)
        ]
        await test_es_index_model.bulk_insert(docs)

        ids_to_delete = [doc.id for doc in docs[:3]]
        await test_es_index_model.bulk_delete(ids_to_delete)

        assert len(await test_es_index_model.get([docs[0].id])) == 0
        assert len(await test_es_index_model.get([docs[3].id])) != 0

    @pytest.mark.asyncio
    async def test_delete_by_query(self, test_es_index_model):
        """测试根据条件删除"""
        docs = [
            test_es_index_model(
                title=f"DelQuery {i}",
                content=f"Content {i}",
                category=["del_target"] if i <= 3 else ["keep"],
                embedding=[0.1 * i] * 1536,
            )
            for i in range(1, 6)
        ]
        await test_es_index_model.bulk_insert(docs)

        from ext.indexing.types import FilterClause

        filter_clause = FilterClause(equals={"category": "del_target"})
        await test_es_index_model.delete_by_query(filter_clause)

        assert len(await test_es_index_model.get([docs[0].id])) == 0
        assert len(await test_es_index_model.get([docs[3].id])) != 0

    @pytest.mark.asyncio
    async def test_count(self, test_es_index_model):
        """测试统计文档数量"""
        docs = [
            test_es_index_model(
                title=f"Count {i}",
                content=f"Content {i}",
                category=["count"],
                embedding=[0.1 * i] * 1536,
            )
            for i in range(1, 6)
        ]
        await test_es_index_model.bulk_insert(docs)

        count = await test_es_index_model.count()
        assert count >= 5

        from ext.indexing.types import FilterClause

        filter_clause = FilterClause(equals={"category": "count"})
        filtered_count = await test_es_index_model.count(filter_clause=filter_clause)
        assert filtered_count >= 5

    @pytest.mark.asyncio
    async def test_exists(self, test_es_index_model):
        """测试检查文档是否存在"""
        doc = test_es_index_model(
            title="Exists Test",
            content="Exists test content",
            category=["exists"],
            embedding=[0.5] * 1536,
        )
        await doc.save()

        exists = await test_es_index_model.exists(doc.id)
        assert exists is True

        not_exists = await test_es_index_model.exists("nonexistent_exists")
        assert not_exists is False


@skip_if_no_es
class TestElasticSearchBaseIndexModelWithPartitionKey:
    """测试 Elasticsearch 的 BaseIndexModel 方法（带分区键，需要真实后端连接）"""

    @pytest.mark.asyncio
    async def test_create_schema(self, test_es_index_model_with_partition):
        await test_es_index_model_with_partition.drop_schema()
        await test_es_index_model_with_partition.create_schema(drop_existing=False)
        await test_es_index_model_with_partition.drop_schema()
        await test_es_index_model_with_partition.create_schema(drop_existing=True)

    @pytest.mark.asyncio
    async def test_save_and_get(self, test_es_index_model_with_partition):
        """测试保存单个文档"""
        doc = test_es_index_model_with_partition(
            tenant_id="tenant_1",
            title="Test Document",
            content="Test content",
            category=["test"],
            embedding=[0.1] * 1536,
        )
        assert not doc.id
        await doc.save()
        assert doc.id

        docs = await test_es_index_model_with_partition.get([doc.id])

        assert len(docs) == 1
        assert docs[0].id == doc.id
        assert docs[0].title == doc.title
        assert docs[0].content == doc.content
        assert docs[0].category == doc.category
        assert docs[0].tenant_id == doc.tenant_id

        result = await test_es_index_model_with_partition.get(["nonexistent_id"])
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_filter_simple(self, test_es_index_model_with_partition):
        """测试简单过滤查询"""
        docs = [
            test_es_index_model_with_partition(
                tenant_id="tenant_1",
                title=f"Doc {i}",
                content=f"Content {i}",
                category=["tech"] if i <= 2 else ["other"],
                embedding=[0.1 * i] * 1536,
            )
            for i in range(1, 5)
        ]
        for doc in docs:
            await doc.save()

        from ext.indexing.types import FilterClause

        results = await test_es_index_model_with_partition.filter(
            filter_clause=FilterClause(equals={"category": "tech"}), limit=10
        )
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_filter_with_pagination(self, test_es_index_model_with_partition):
        """测试过滤查询（带分页）"""
        docs = [
            test_es_index_model_with_partition(
                tenant_id="tenant_1",
                id=f"page_{i}",
                title=f"Page Doc {i}",
                content=f"Content {i}",
                category=["page_test"],
                embedding=[0.1 * i] * 1536,
            )
            for i in range(1, 11)
        ]
        for doc in docs:
            await doc.save()

        from ext.indexing.types import FilterClause

        page1 = await test_es_index_model_with_partition.filter(
            filter_clause=FilterClause(equals={"category": "page_test"}), limit=3, offset=0
        )
        page2 = await test_es_index_model_with_partition.filter(
            filter_clause=FilterClause(equals={"category": "page_test"}), limit=3, offset=3
        )
        assert len(page1) == 3
        assert len(page2) == 3

    @pytest.mark.asyncio
    async def test_filter_with_sort(self, test_es_index_model_with_partition):
        """测试过滤查询（带排序）"""
        docs = [
            test_es_index_model_with_partition(
                tenant_id="tenant_1",
                title="Sort",
                content=f"Content {i}",
                category=[f"sort_test {10 - i}"],
                embedding=[0.1 * i] * 1536,
            )
            for i in range(1, 6)
        ]
        for doc in docs:
            await doc.save()

        from ext.indexing.types import FilterClause

        results = await test_es_index_model_with_partition.filter(
            filter_clause=FilterClause(equals={"title.keyword": "Sort"}), limit=10, sort="category:asc"
        )
        assert len(results) >= 5
        assert results[0].category[0].startswith("sort_test 5")

    @pytest.mark.asyncio
    async def test_search_dense(self, test_es_index_model_with_partition, sample_query_vector):
        """测试稠密向量搜索"""
        from ext.indexing.types import DenseSearchClause

        query_clause = DenseSearchClause(vector=sample_query_vector, top_k=5, metric="cosine")
        results = await test_es_index_model_with_partition.search(query_clause=query_clause, limit=5)

        assert len(results) != 0
        assert isinstance(results, list)
        assert all(isinstance(result, tuple) and len(result) == 2 for result in results)
        assert all(
            isinstance(doc, test_es_index_model_with_partition) and isinstance(score, float) for doc, score in results
        )

    @pytest.mark.asyncio
    async def test_search_with_filter(self, test_es_index_model_with_partition, sample_query_vector):
        """测试搜索（带过滤条件）"""
        from ext.indexing.types import DenseSearchClause, FilterClause

        query_clause = DenseSearchClause(vector=sample_query_vector, top_k=5)
        filter_clause = FilterClause(equals={"category": "tech"})
        results = await test_es_index_model_with_partition.search(
            query_clause=query_clause, filter_clause=filter_clause, limit=5
        )

        assert isinstance(results, list)
        for doc, score in results:
            assert doc.category == ["tech"]

    @pytest.mark.asyncio
    async def test_search_cursor(self, test_es_index_model_with_partition, sample_query_vector):
        """测试搜索（游标分页）"""
        from ext.indexing.types import DenseSearchClause

        query_clause = DenseSearchClause(vector=sample_query_vector, top_k=10)
        cursor = await test_es_index_model_with_partition.search_cursor(query_clause=query_clause, page_size=3)

        assert cursor.results is not None
        assert len(cursor.results) <= 3
        assert isinstance(cursor.next_cursor, (str, type(None)))

    @pytest.mark.asyncio
    async def test_bulk_insert(self, test_es_index_model_with_partition):
        """测试批量插入"""
        docs = [
            test_es_index_model_with_partition(
                tenant_id="tenant_1",
                title=f"Bulk {i}",
                content=f"Content {i}",
                category=["bulk"],
                embedding=[0.1 * i] * 1536,
            )
            for i in range(1, 21)
        ]

        from ext.indexing.types import FilterClause

        await test_es_index_model_with_partition.bulk_insert(docs, batch_size=5, concurrent_batches=2)
        assert (
            len(
                await test_es_index_model_with_partition.filter(
                    filter_clause=FilterClause(equals={"category": "bulk"}), limit=30
                )
            )
            == 20
        )

    @pytest.mark.asyncio
    async def test_bulk_update(self, test_es_index_model_with_partition):
        """测试批量更新"""
        docs = [
            test_es_index_model_with_partition(
                tenant_id="tenant_1",
                title=f"Title {i}",
                content=f"Content {i}",
                category=["update"],
                embedding=[0.1 * i] * 1536,
            )
            for i in range(1, 6)
        ]
        await test_es_index_model_with_partition.bulk_insert(docs)

        for doc in docs:
            doc.title = f"Updated {doc.id}"
        await test_es_index_model_with_partition.bulk_update(docs)

        assert (await test_es_index_model_with_partition.get([docs[0].id]))[0].title.startswith("Updated")

    @pytest.mark.asyncio
    async def test_bulk_upsert(self, test_es_index_model_with_partition):
        """测试批量 upsert"""
        docs = [
            test_es_index_model_with_partition(
                tenant_id="tenant_1",
                title=f"Upsert {i}",
                content=f"Content {i}",
                category=["upsert"],
                embedding=[0.1 * i] * 1536,
            )
            for i in range(1, 6)
        ]

        await test_es_index_model_with_partition.bulk_upsert(docs)

        docs[0].title = "Updated Upsert 0"
        await test_es_index_model_with_partition.bulk_upsert(docs)

        docs = await test_es_index_model_with_partition.get([docs[0].id])
        assert docs[0].title == "Updated Upsert 0"

    @pytest.mark.asyncio
    async def test_bulk_delete(self, test_es_index_model_with_partition):
        """测试批量删除"""
        docs = [
            test_es_index_model_with_partition(
                tenant_id="tenant_1",
                title=f"Delete {i}",
                content=f"Content {i}",
                category=["delete"],
                embedding=[0.1 * i] * 1536,
            )
            for i in range(1, 6)
        ]
        await test_es_index_model_with_partition.bulk_insert(docs)

        ids_to_delete = [doc.id for doc in docs[:3]]
        await test_es_index_model_with_partition.bulk_delete(ids_to_delete)

        assert len(await test_es_index_model_with_partition.get([docs[0].id])) == 0
        assert len(await test_es_index_model_with_partition.get([docs[3].id])) != 0

    @pytest.mark.asyncio
    async def test_delete_by_query(self, test_es_index_model_with_partition):
        """测试根据条件删除"""
        docs = [
            test_es_index_model_with_partition(
                tenant_id="tenant_1",
                title=f"DelQuery {i}",
                content=f"Content {i}",
                category=["del_target"] if i <= 3 else ["keep"],
                embedding=[0.1 * i] * 1536,
            )
            for i in range(1, 6)
        ]
        await test_es_index_model_with_partition.bulk_insert(docs)

        from ext.indexing.types import FilterClause

        filter_clause = FilterClause(equals={"category": "del_target", "tenant_id": "tenant_1"})
        await test_es_index_model_with_partition.delete_by_query(filter_clause)

        assert len(await test_es_index_model_with_partition.get([docs[0].id])) == 0
        assert len(await test_es_index_model_with_partition.get([docs[3].id])) != 0

    @pytest.mark.asyncio
    async def test_count(self, test_es_index_model_with_partition):
        """测试统计文档数量"""
        docs = [
            test_es_index_model_with_partition(
                tenant_id="tenant_1",
                title=f"Count {i}",
                content=f"Content {i}",
                category=["count"],
                embedding=[0.1 * i] * 1536,
            )
            for i in range(1, 6)
        ]
        await test_es_index_model_with_partition.bulk_insert(docs)

        count = await test_es_index_model_with_partition.count()
        assert count >= 5

        from ext.indexing.types import FilterClause

        filter_clause = FilterClause(equals={"category": "count"})
        filtered_count = await test_es_index_model_with_partition.count(filter_clause=filter_clause)
        assert filtered_count >= 5

    @pytest.mark.asyncio
    async def test_exists(self, test_es_index_model_with_partition):
        """测试检查文档是否存在"""
        doc = test_es_index_model_with_partition(
            tenant_id="tenant_1",
            title="Exists Test",
            content="Exists test content",
            category=["exists"],
            embedding=[0.5] * 1536,
        )
        await doc.save()

        exists = await test_es_index_model_with_partition.exists(doc.id)
        assert exists is True

        not_exists = await test_es_index_model_with_partition.exists("nonexistent_exists")
        assert not_exists is False
