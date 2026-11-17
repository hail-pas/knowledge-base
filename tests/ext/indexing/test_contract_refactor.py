"""Unit-level indexing contracts; no database or search engine required."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from ext.indexing.base import BaseIndexModel
from ext.indexing.search import IndexSearch
from ext.indexing.types import DenseSearchClause, HybridSearchClause, SparseSearchClause


class Note(BaseIndexModel):
    content: str

    class Meta:
        index_name = "notes"
        dense_vector_field = None


class FakeSearch:
    def __init__(self):
        self.calls = []
        self.dense = [({"id": "a", "content": "A"}, 0.1), ({"id": "b", "content": "B"}, 0.8)]
        self.sparse = [({"id": "b", "content": "B"}, 10.0), ({"id": "c", "content": "C"}, 0.2)]

    async def search_native(self, model, query, filters, limit, offset):
        self.calls.append((type(query).__name__, limit, offset))
        source = self.dense if isinstance(query, DenseSearchClause) else self.sparse
        return source[offset : offset + limit]


@pytest.mark.asyncio
async def test_weighted_rrf_has_backend_independent_order_and_no_duplicate_hits():
    provider = FakeSearch()
    query = HybridSearchClause(
        dense=DenseSearchClause(vector=[1.0]),
        sparse=SparseSearchClause(query_text="keyword"),
        weight_dense=0.7,
        weight_sparse=0.3,
    )
    rows = await IndexSearch.execute(provider, Note, query, None, 5, 0)
    assert [row[0]["id"] for row in rows] == ["b", "a", "c"]
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_portable_cursor_advances_without_repeat():
    provider = FakeSearch()
    query = DenseSearchClause(vector=[0.5])
    first, cursor = await IndexSearch.execute_cursor(provider, Note, query, page_size=1)
    second, final_cursor = await IndexSearch.execute_cursor(
        provider, Note, query, page_size=1, cursor=cursor,
    )
    assert [row[0]["id"] for row in first] == ["a"]
    assert [row[0]["id"] for row in second] == ["b"]
    assert cursor == "1"
    assert final_cursor is None


@pytest.mark.asyncio
async def test_invalid_pagination_rejected():
    with pytest.raises(ValueError):
        await IndexSearch.execute(FakeSearch(), Note, DenseSearchClause(vector=[1]), None, 0, 0)
    with pytest.raises(ValueError):
        await IndexSearch.execute_cursor(
            FakeSearch(), Note, DenseSearchClause(vector=[1]), page_size=2, cursor="-1",
        )


def test_invalid_hybrid_weights_rejected():
    with pytest.raises(ValidationError):
        HybridSearchClause(
            dense=DenseSearchClause(vector=[1]),
            sparse=SparseSearchClause(query_text="x"),
            weight_dense=0.8,
            weight_sparse=0.8,
        )


@pytest.mark.asyncio
async def test_save_inserts_new_and_upserts_existing_identity():
    provider = SimpleNamespace(
        insert=AsyncMock(return_value=[{"id": "new-1"}]),
        bulk_upsert=AsyncMock(return_value=[{"id": "new-1"}]),
    )
    class SavedNote(Note):
        pass

    SavedNote.Meta.provider = provider
    note = SavedNote(content="text")
    await note.save()
    assert note.id == "new-1"
    provider.insert.assert_awaited_once()
    note.content = "updated"
    await note.save()
    provider.bulk_upsert.assert_awaited_once()
    assert provider.bulk_upsert.await_args.args[1][0]["id"] == "new-1"


@pytest.mark.asyncio
async def test_bulk_insert_and_upsert_use_distinct_provider_operations():
    provider = SimpleNamespace(
        insert=AsyncMock(side_effect=lambda model, rows: [{"id": f"i{i}"} for i in range(len(rows))]),
        bulk_upsert=AsyncMock(side_effect=lambda model, rows: rows),
    )
    class BatchNote(Note):
        pass

    BatchNote.Meta.provider = provider
    docs = [BatchNote(content="a"), BatchNote(content="b")]
    await BatchNote.bulk_insert(docs, batch_size=1, concurrent_batches=2)
    assert provider.insert.await_count == 2
    assert provider.bulk_upsert.await_count == 0
    await BatchNote.bulk_upsert(docs)
    assert provider.bulk_upsert.await_count == 1
    with pytest.raises(ValueError):
        await BatchNote.bulk_insert(docs, batch_size=0)


def test_cache_key_reuses_equivalent_unsaved_configs():
    from ext.indexing.factory import IndexingProviderFactory

    def config():
        return SimpleNamespace(
            id=None, type="postgresql", host="localhost", port=5432,
            username="alice", password="test", use_ssl=False,
            verify_ssl=False, timeout=30, max_connections=5,
            max_retries=2, extra_config={"database": "knowledge"},
        )

    assert IndexingProviderFactory._cache_key(config()) == IndexingProviderFactory._cache_key(config())
