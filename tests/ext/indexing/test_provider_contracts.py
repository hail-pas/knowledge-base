"""Provider adapter regression tests without running external services."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ext.indexing.base import BaseIndexModel
from ext.indexing.providers.elasticsearch import ElasticsearchProvider
from ext.indexing.providers.milvus import MilvusProvider
from ext.indexing.types import FilterClause, SparseSearchClause, parse_sort


class Note(BaseIndexModel):
    content: str

    class Meta:
        dense_vector_field = None
        index_name = "adapter_notes"


def test_sparse_dsl_uses_top_level_min_score():
    provider = object.__new__(ElasticsearchProvider)
    provider._get_default_text_field = lambda model: "content"
    result = provider._build_sparse_search_body(Note, SparseSearchClause(query_text="example", min_score=0.4))
    assert result["min_score"] == 0.4
    assert result["query"]["match"]["content"]["query"] == "example"
    assert "min_score" not in result["query"]["match"]["content"]


def test_sparse_filters_do_not_create_self_referential_json():
    provider = object.__new__(ElasticsearchProvider)
    provider._convert_filter = lambda model, filters: {"term": {"collection_id": 2}}
    result = provider._apply_filter_to_search_body(
        Note, {"query": {"match": {"content": "term"}}},
        SparseSearchClause(query_text="term"),
        FilterClause(equals={"collection_id": 2}),
    )
    assert json.loads(json.dumps(result)) == result
    assert len(result["query"]["bool"]["must"]) == 2


def test_sort_contract_and_invalid_sort():
    assert parse_sort("created_at:desc") == ("created_at", "desc")
    assert parse_sort("-created_at") == ("created_at", "desc")
    assert parse_sort("content") == ("content", "asc")
    assert parse_sort("meta.keyword:asc") == ("meta.keyword", "asc")
    with pytest.raises(ValueError):
        parse_sort("id;DROP TABLE notes")
    with pytest.raises(ValueError):
        parse_sort("content:reverse")


@pytest.mark.asyncio
async def test_milvus_bulk_upsert_splits_new_and_existing_ids():
    provider = object.__new__(MilvusProvider)
    provider._client = SimpleNamespace(upsert=AsyncMock(return_value={"upsert_count": 1, "ids": [7]}))
    provider.extra_config = SimpleNamespace(auto_flush=False)
    provider.insert = AsyncMock(return_value=[{"id": 20, "content": "new"}])
    provider._convert_datetime_for_write = lambda rows, model: rows
    provider.build_collection_name = lambda model: "adapter_notes"
    result = await provider.bulk_upsert(
        Note, [{"content": "new"}, {"id": 7, "content": "old"}],
    )
    assert [item["id"] for item in result] == [20, 7]
    provider.insert.assert_awaited_once()
    provider._client.upsert.assert_awaited_once()
    assert provider._client.upsert.await_args.kwargs["data"] == [{"id": 7, "content": "old"}]


@pytest.mark.asyncio
async def test_delete_filter_requires_a_predicate():
    assert not FilterClause().has_conditions()
    assert not FilterClause(and_conditions=[FilterClause()]).has_conditions()
    assert FilterClause(or_conditions=[FilterClause(equals={"collection_id": 1})]).has_conditions()
