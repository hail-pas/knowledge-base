"""Backend-independent search orchestration.

Providers own backend query compilation; this module owns dispatch, fallback
fusion, validation and deterministic pagination semantics.
"""
from __future__ import annotations

import asyncio
from typing import Any

from ext.indexing.types import DenseSearchClause, FilterClause, HybridSearchClause, SparseSearchClause

SearchHit = tuple[dict[str, Any], float]


def _validate_page(limit: int, offset: int) -> None:
    if limit <= 0 or offset < 0:
        raise ValueError("limit must be positive and offset cannot be negative")


def _fuse_rrf(
    dense: list[SearchHit],
    sparse: list[SearchHit],
    weight_dense: float,
    weight_sparse: float,
    *,
    k: int = 60,
) -> list[SearchHit]:
    """Weighted reciprocal-rank fusion avoids mixing incompatible raw scores."""
    fused: dict[str, SearchHit] = {}
    for rows, weight in ((dense, weight_dense), (sparse, weight_sparse)):
        if not weight:
            continue
        for rank, (document, _) in enumerate(rows, start=1):
            if "id" not in document:
                raise ValueError("Search result must include an id for hybrid fusion")
            key = str(document["id"])
            original, score = fused.get(key, (document, 0.0))
            fused[key] = (original, score + weight / (k + rank))
    return sorted(fused.values(), key=lambda hit: (-hit[1], str(hit[0]["id"])))


class SearchStrategyFactory:
    """One search entrypoint; every backend uses the same weighted RRF for hybrid queries."""

    @classmethod
    async def execute(
        cls, provider: Any, model: Any,
        query: DenseSearchClause | SparseSearchClause | HybridSearchClause,
        filters: FilterClause | None = None,
        limit: int = 10, offset: int = 0,
    ) -> list[SearchHit]:
        _validate_page(limit, offset)
        if isinstance(query, (DenseSearchClause, SparseSearchClause)):
            return await provider.search_native(model, query, filters, limit, offset)
        if not isinstance(query, HybridSearchClause):
            raise TypeError(f"Unsupported search clause: {type(query).__name__}")
        requested = offset + limit
        candidate_limit = max(requested, query.dense.top_k, query.sparse.top_k)
        dense, sparse = await asyncio.gather(
            provider.search_native(model, query.dense, filters, candidate_limit, 0),
            provider.search_native(model, query.sparse, filters, candidate_limit, 0),
        )
        merged = _fuse_rrf(dense, sparse, query.weight_dense, query.weight_sparse)
        return merged[offset:requested]

    @classmethod
    async def execute_cursor(
        cls, provider: Any, model: Any,
        query: DenseSearchClause | SparseSearchClause | HybridSearchClause,
        filters: FilterClause | None = None,
        page_size: int = 100, cursor: str | None = None,
    ) -> tuple[list[SearchHit], str | None]:
        """Portable offset cursor; deterministic only for stable underlying indexes.

        Backends may implement their own search-after/PIT API separately. In
        particular, do not invoke backend-specific cursors when composing RRF.
        """
        offset = int(cursor) if cursor is not None else 0
        _validate_page(page_size, offset)
        rows = await cls.execute(provider, model, query, filters, page_size + 1, offset)
        more = len(rows) > page_size
        return rows[:page_size], str(offset + page_size) if more else None
