"""Indexing routing regression tests; no external search service required."""

from unittest.mock import AsyncMock

import pytest

from ext.indexing.base import BaseIndexModel
from ext.indexing.factory import IndexModelFactory, IndexingProviderFactory
from ext.indexing.models import DocumentContentSparseIndex
from ext.ext_tortoise.enums import IndexingBackendTypeEnum, EmbeddingModelTypeEnum
from ext.ext_tortoise.models.knowledge_base import EmbeddingModelConfig, IndexingBackendConfig


class SampleIndex(BaseIndexModel):
    content: str

    class Meta:
        index_name = "isolated"
        dense_vector_field = None


def test_meta_isolation():
    class First(SampleIndex):
        class Meta:
            index_name = "first"

    class Second(SampleIndex):
        class Meta:
            index_name = "second"

    assert SampleIndex.Meta.index_name == "isolated"
    assert First.Meta.index_name == "first"
    assert Second.Meta.index_name == "second"
    First.Meta.provider = object()
    assert Second.Meta.provider is None


def test_scoped_provider_binding_does_not_mutate_global_model():
    IndexModelFactory.clear_cache()
    provider_a = type("Provider", (), {"config": type("Config", (), {"id": 101})()})()
    provider_b = type("Provider", (), {"config": type("Config", (), {"id": 102})()})()
    a = IndexModelFactory.bind_provider(DocumentContentSparseIndex, provider_a)
    b = IndexModelFactory.bind_provider(DocumentContentSparseIndex, provider_b)
    assert a is not b
    assert a.Meta.provider is provider_a
    assert b.Meta.provider is provider_b
    assert DocumentContentSparseIndex.Meta.provider is not provider_a
    assert IndexModelFactory.bind_provider(DocumentContentSparseIndex, provider_a) is a


@pytest.mark.asyncio
async def test_dynamic_model_distinguishes_providers():
    class VectorIndex(BaseIndexModel):
        vector: list[float]

        class Meta:
            index_name = "vector_index"
            dense_vector_field = "vector"

    emb = EmbeddingModelConfig(
        name="embedding-test", type=EmbeddingModelTypeEnum.openai, model_name="test", dimension=3,
    )
    provider_a = type("Provider", (), {"create_collection": AsyncMock()})()
    provider_b = type("Provider", (), {"create_collection": AsyncMock()})()
    IndexModelFactory.clear_cache()
    a = await IndexModelFactory.create_for_embedding(VectorIndex, emb, provider=provider_a)
    b = await IndexModelFactory.create_for_embedding(VectorIndex, emb, provider=provider_b)
    assert a is not b
    assert a.Meta.provider is provider_a
    assert b.Meta.provider is provider_b
    assert VectorIndex.Meta.provider is None
    await IndexModelFactory.create_for_embedding(VectorIndex, emb, provider=provider_a)
    provider_a.create_collection.assert_awaited_once()
    provider_b.create_collection.assert_awaited_once()


@pytest.mark.asyncio
async def test_provider_factory_reuses_and_refreshes_connections():
    class Provider:
        def __init__(self, config):
            self.config = config
            self.connect = AsyncMock()
            self.disconnect = AsyncMock()

    config = IndexingBackendConfig(
        id=99991, name="test", type=IndexingBackendTypeEnum.elasticsearch,
        host="localhost", port=9200,
    )
    original = IndexingProviderFactory._providers.get(str(config.type))
    IndexingProviderFactory.register(str(config.type), Provider)
    try:
        await IndexingProviderFactory.invalidate(config.id)
        first = await IndexingProviderFactory.create(config)
        assert await IndexingProviderFactory.create(config) is first
        config.host = "other-host"
        second = await IndexingProviderFactory.create(config)
        assert second is not first
        first.disconnect.assert_awaited_once()
        await IndexingProviderFactory.invalidate(config.id)
        second.disconnect.assert_awaited_once()
    finally:
        if original is not None:
            IndexingProviderFactory.register(str(config.type), original)
        await IndexingProviderFactory.invalidate(config.id)
