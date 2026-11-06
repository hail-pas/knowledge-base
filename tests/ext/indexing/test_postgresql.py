from types import SimpleNamespace

import pytest

from ext.indexing.base import BaseIndexModel
from ext.indexing.providers.postgresql import PostgreSQLProvider
from ext.indexing.types import FilterClause, SparseSearchClause


class SparseDocumentIndex(BaseIndexModel):
    content: str
    status: str

    class Meta(BaseIndexModel.Meta):
        index_name = "sparse_documents"
        dense_vector_field = None


class _Connection:
    def __init__(self):
        self.query = None
        self.args = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def fetch(self, query, *args):
        self.query = query
        self.args = args
        return [{"id": "1", "content": "matching content", "status": "active", "score": 0.8}]


class _Pool:
    def __init__(self):
        self.connection = _Connection()

    def acquire(self):
        return self.connection


@pytest.mark.asyncio
async def test_sparse_min_score_is_applied_before_pagination():
    provider = PostgreSQLProvider.__new__(PostgreSQLProvider)
    provider.extra_config = SimpleNamespace(text_search_config="simple")
    provider._pool = _Pool()
    provider._table = lambda model: '"test"."sparse_documents"'

    query = SparseSearchClause(query_text="matching", min_score=0.5)
    filters = FilterClause(equals={"status": "active"})

    results = await provider._sparse(SparseDocumentIndex, query, filters, limit=10, offset=0)

    connection = provider._pool.connection
    assert ">= $2" in connection.query
    assert "LIMIT 10 OFFSET 0" in connection.query
    assert connection.args == ("matching", 0.5, "active")
    assert len(results) == 1
    assert results[0][1] == 0.8
