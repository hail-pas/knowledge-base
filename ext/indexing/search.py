"""Backend-neutral retrieval strategies.

Providers execute native backend queries; strategies decide how query modes are composed.
"""
from __future__ import annotations

import asyncio
from typing import Any, Protocol, cast

from ext.indexing.base import BaseIndexModel, BaseProvider
from ext.indexing.types import DenseSearchClause, FilterClause, HybridSearchClause, SparseSearchClause


class NativeSearchProvider(Protocol):
    """Optional provider capability for executing one backend-native query."""

    async def search_native(
        self,
        model: type[BaseIndexModel],
        query: Any,
        filters: FilterClause | None,
        limit: int,
        offset: int,
    ) -> list[tuple[dict[str, Any], float]]: ...


class SearchStrategy:
    """Base contract for one retrieval mode."""

    async def execute(
        self,
        provider: NativeSearchProvider,
        model: type[BaseIndexModel],
        query: Any,
        filters: FilterClause | None,
        limit: int,
        offset: int,
    ) -> list[tuple[dict[str, Any], float]]:
        raise NotImplementedError


class DenseSearchStrategy(SearchStrategy):
    async def execute(self, provider, model, query, filters, limit, offset):
        return await provider.search_native(model, query, filters, limit, offset)


class SparseSearchStrategy(SearchStrategy):
    async def execute(self, provider, model, query, filters, limit, offset):
        return await provider.search_native(model, query, filters, limit, offset)


class HybridSearchStrategy(SearchStrategy):
    """Use native hybrid search when available; otherwise weighted RRF."""

    async def execute(self, provider, model, query, filters, limit, offset):
        if getattr(provider, "supports_native_hybrid", False):
            return await provider.search_native(model, query, filters, limit, offset)

        candidate_limit = max(limit + offset, query.dense.top_k, query.sparse.top_k, limit * 3)
        dense, sparse = await asyncio.gather(
            provider.search_native(model, query.dense, filters, candidate_limit, 0),
            provider.search_native(model, query.sparse, filters, candidate_limit, 0),
        )
        fused: dict[str, tuple[dict[str, Any], float]] = {}
        for hits, weight in (
            (dense, query.weight_dense),
            (sparse, query.weight_sparse),
        ):
            for rank, (document, _) in enumerate(hits, 1):
                key = str(document["id"])
                previous = fused.get(key, (document, 0.0))
                fused[key] = (document, previous[1] + weight / (60 + rank))
        ranked = sorted(fused.values(), key=lambda item: item[1], reverse=True)
        return ranked[offset : offset + limit]


class SearchStrategyFactory:
    """Resolve and execute a retrieval strategy by query type."""

    _strategies: dict[type, SearchStrategy] = {
        DenseSearchClause: DenseSearchStrategy(),
        SparseSearchClause: SparseSearchStrategy(),
        HybridSearchClause: HybridSearchStrategy(),
    }

    @classmethod
    def register(cls, query_type: type, strategy: SearchStrategy) -> None:
        cls._strategies[query_type] = strategy

    @classmethod
    async def execute(cls, provider, model, query, filters, limit, offset):
        strategy = cls._strategies.get(type(query))
        if strategy is None:
            raise TypeError(f"Unsupported search clause: {type(query).__name__}")
        return await strategy.execute(cast(NativeSearchProvider, provider), model, query, filters, limit, offset)

    @classmethod
    async def execute_cursor(cls, provider, model, query, filters, page_size, cursor):
        native_cursor = getattr(provider, "search_cursor_native", None)
        if type(query) in {DenseSearchClause, SparseSearchClause, HybridSearchClause} and native_cursor:
            return await native_cursor(model, query, filters, page_size, cursor)
        offset = int(cursor or 0)
        if offset < 0:
            raise ValueError("Cursor offset cannot be negative")
        rows = await cls.execute(provider, model, query, filters, page_size + 1, offset)
        next_cursor = str(offset + page_size) if len(rows) > page_size else None
        return rows[:page_size], next_cursor
