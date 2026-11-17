"""Regression tests for backend switching and model lifecycle."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ext.indexing.base import BaseIndexModel
from ext.indexing.factory import IndexModelFactory, IndexingProviderFactory


class BaseDocument(BaseIndexModel):
    content: str

    class Meta:
        index_name = "isolated_document"
        dense_vector_field = None


def test_meta_is_isolated_between_subclasses():
    class First(BaseDocument):
        pass

    class Second(BaseDocument):
        pass

    First.Meta.provider = object()
    First.Meta.index_name = "first_document"
    assert Second.Meta.provider is None
    assert Second.Meta.index_name == "isolated_document"
    assert BaseDocument.Meta.index_name == "isolated_document"


@pytest.mark.asyncio
async def test_provider_scoped_model_initializes_schema_once():
    IndexModelFactory.clear_cache()
    provider = SimpleNamespace(config=SimpleNamespace(id=42))
    create_schema = AsyncMock()

    class TestDocument(BaseDocument):
        @classmethod
        async def create_schema(cls, drop_existing=False):
            await create_schema()

    models = await asyncio.gather(
        *(IndexModelFactory.create_for_provider(TestDocument, provider) for _ in range(8))
    )
    assert all(model is models[0] for model in models)
    create_schema.assert_awaited_once()
    IndexModelFactory.clear_cache()


def test_connection_cache_key_changes_when_settings_change():
    config = SimpleNamespace(
        id=7, type="postgresql", host="localhost", port=5432,
        username="user", password="pw", use_ssl=False, verify_ssl=False,
        timeout=30, max_connections=10, max_retries=3,
        extra_config={"database": "one"},
    )
    first = IndexingProviderFactory._cache_key(config)
    config.extra_config = {"database": "two"}
    second = IndexingProviderFactory._cache_key(config)
    assert first != second
    assert first[0] == second[0]
