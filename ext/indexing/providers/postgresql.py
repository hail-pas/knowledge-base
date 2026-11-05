"""PostgreSQL indexing backend using pgvector and PostgreSQL full-text search."""
from __future__ import annotations

import json
import re
from typing import Any, get_args, get_origin

import asyncpg
from loguru import logger

from ext.indexing.base import BaseIndexModel, BaseProvider
from ext.indexing.providers.types import ProviderConfig
from ext.indexing.types import DenseSearchClause, FilterClause, HybridSearchClause, SparseSearchClause
from ext.ext_tortoise.models.knowledge_base import IndexingBackendConfig


class PostgreSQLConfig(ProviderConfig):
    database: str = "postgres"
    schema: str = "public"
    vector_metric: str = "cosine"
    vector_index: str = "hnsw"
    text_search_config: str = "simple"
    pool_min_size: int = 1
    pool_max_size: int = 10


class PostgreSQLProvider(BaseProvider[PostgreSQLConfig]):
    """One logical index maps to a typed PostgreSQL table; dimensions are part of its physical name."""

    def __init__(self, config: IndexingBackendConfig) -> None:
        super().__init__(config, PostgreSQLConfig)
        if self.extra_config.vector_metric not in {"cosine", "l2", "ip"}:
            raise ValueError("vector_metric must be cosine, l2 or ip")
        if self.extra_config.vector_index not in {"hnsw", "ivfflat"}:
            raise ValueError("vector_index must be hnsw or ivfflat")
        if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", self.extra_config.schema):
            raise ValueError("Invalid schema")
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(
            host=self.config.host, port=self.config.port or 5432, user=self.config.username,
            password=self.password, database=self.extra_config.database,
            min_size=self.extra_config.pool_min_size, max_size=self.extra_config.pool_max_size,
            timeout=self.config.timeout, command_timeout=self.config.timeout,
            ssl=self.config.verify_ssl if self.config.use_ssl else False,
        )
        async with self.pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.extra_config.schema}"')

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PostgreSQL provider is not connected")
        return self._pool

    async def disconnect(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    @staticmethod
    def _q(name: str) -> str:
        if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", name):
            raise ValueError(f"Invalid SQL identifier: {name}")
        return f'"{name}"'

    def _table(self, model: type[BaseIndexModel]) -> str:
        return f'{self._q(self.extra_config.schema)}.{self._q(self.build_collection_name(model))}'

    def _columns(self, model: type[BaseIndexModel]) -> list[str]:
        columns = []
        for name, field in model.model_fields.items():
            if name == "id":
                kind = "BIGSERIAL PRIMARY KEY" if model.Meta.auto_generate_id else "TEXT PRIMARY KEY"
            elif name == model.Meta.dense_vector_field:
                dimension = model.Meta.dense_vector_dimension
                if not dimension or dimension > 2000:
                    raise ValueError("pgvector vector requires a dimension between 1 and 2000")
                kind = f"vector({dimension})"
            else:
                annotation = field.annotation
                origin, args = get_origin(annotation), get_args(annotation)
                if origin is not None and type(None) in args:
                    annotation = next((arg for arg in args if arg is not type(None)), str)
                    origin = get_origin(annotation)
                if annotation is str:
                    kind = "TEXT"
                elif annotation is bool:
                    kind = "BOOLEAN"
                elif annotation is int:
                    kind = "BIGINT"
                elif annotation is float:
                    kind = "DOUBLE PRECISION"
                elif origin is list or origin is dict or annotation in (list, dict, Any):
                    kind = "JSONB"
                elif getattr(annotation, "__name__", "") == "datetime":
                    kind = "TIMESTAMPTZ"
                else:
                    kind = "JSONB"
            columns.append(f"{self._q(name)} {kind}")
        return columns

    async def create_collection(self, model_class: type[BaseIndexModel], drop_existing: bool = False) -> None:
        self._validate_model_config(model_class)
        table = self._table(model_class)
        async with self.pool.acquire() as conn:
            if drop_existing:
                await conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
            await conn.execute(f"CREATE TABLE IF NOT EXISTS {table} ({', '.join(self._columns(model_class))})")
            vector_field = model_class.Meta.dense_vector_field
            if vector_field and model_class.Meta.dense_vector_dimension:
                ops = {"cosine": "vector_cosine_ops", "l2": "vector_l2_ops", "ip": "vector_ip_ops"}
                idx = self._q(f"{model_class.Meta.index_name}_{model_class.Meta.dense_vector_dimension}_hnsw")
                await conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {idx} ON {table} USING hnsw "
                    f"({self._q(vector_field)} {ops[self.extra_config.vector_metric]})"
                )
            for name, field in model_class.model_fields.items():
                extra = field.json_schema_extra or {}
                metadata = extra.get("index_metadata", {}) if isinstance(extra, dict) else {}
                if field.annotation is str and metadata.get("enable_keyword", name == "content"):
                    idx = self._q(f"{model_class.Meta.index_name}_{name}_fts")
                    cfg = self.extra_config.text_search_config
                    await conn.execute(
                        f"CREATE INDEX IF NOT EXISTS {idx} ON {table} USING GIN "
                        f"(to_tsvector('{cfg}', coalesce({self._q(name)}, '')))"
                    )

    async def drop_collection(self, model_class: type[BaseIndexModel]) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(f"DROP TABLE IF EXISTS {self._table(model_class)} CASCADE")

    def _where(self, clause: FilterClause | None, start: int = 1) -> tuple[str, list[Any]]:
        if clause is None:
            return "", []
        terms: list[str] = []
        values: list[Any] = []
        for field, value in (clause.equals or {}).items():
            values.append(value)
            terms.append(f"{self._q(field)} = ${len(values) + start - 1}")
        for field, items in (clause.in_list or {}).items():
            values.append(items)
            terms.append(f"{self._q(field)} = ANY(${len(values) + start - 1})")
        for field, bounds in (clause.range or {}).items():
            for op, value in bounds.items():
                values.append(value)
                sql_op = {"gte": ">=", "gt": ">", "lte": "<=", "lt": "<"}[op]
                terms.append(f"{self._q(field)} {sql_op} ${len(values) + start - 1}")
        for nested_clause in clause.and_conditions or []:
            sql, params = self._where(nested_clause, start + len(values))
            if sql:
                terms.append(f"({sql})")
                values.extend(params)
        if clause.or_conditions:
            nested = [self._where(item, start + len(values)) for item in clause.or_conditions]
            nested = [(sql, args) for sql, args in nested if sql]
            if nested:
                terms.append("(" + " OR ".join(f"({sql})" for sql, _ in nested) + ")")
                for _, args in nested:
                    values.extend(args)
        return " AND ".join(terms), values

    async def get(self, model_class: type[BaseIndexModel], ids: list) -> list[dict]:
        if not ids:
            return []
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(f"SELECT * FROM {self._table(model_class)} WHERE id = ANY($1)", ids)
            return [dict(row) for row in rows]

    async def filter(self, model_class: type[BaseIndexModel], filter_clause: FilterClause | None,
                     limit: int = 10, offset: int = 0, sort: str | None = None) -> list[dict[str, Any]]:
        where, args = self._where(filter_clause)
        order = ""
        if sort:
            order = f" ORDER BY {self._q(sort.lstrip('-'))} {'DESC' if sort.startswith('-') else 'ASC'}"
        sql = f"SELECT * FROM {self._table(model_class)}" + (f" WHERE {where}" if where else "")
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql + order + f" LIMIT {int(limit)} OFFSET {int(offset)}", *args)
            return [dict(row) for row in rows]

    def _encode(self, model: type[BaseIndexModel], doc: dict[str, Any]) -> tuple[list[str], list[Any], list[str]]:
        names, values, casts = [], [], []
        for key, value in doc.items():
            names.append(self._q(key))
            if key == model.Meta.dense_vector_field and value is not None:
                if len(value) != model.Meta.dense_vector_dimension:
                    raise ValueError(f"Vector dimension mismatch for {key}")
                values.append("[" + ",".join(str(float(v)) for v in value) + "]")
                casts.append("::vector")
            elif isinstance(value, (dict, list)):
                values.append(json.dumps(value, ensure_ascii=False))
                casts.append("::jsonb")
            else:
                values.append(value)
                casts.append("")
        return names, values, casts

    async def _write(self, model: type[BaseIndexModel], documents: list[dict[str, Any]], upsert: bool) -> list[dict]:
        if not documents:
            return []
        output = []
        async with self.pool.acquire() as conn, conn.transaction():
            for raw in documents:
                doc = dict(raw)
                if model.Meta.auto_generate_id and not doc.get("id"):
                    doc.pop("id", None)
                names, values, casts = self._encode(model, doc)
                placeholders = ", ".join(f"${i}{cast}" for i, cast in enumerate(casts, 1))
                updates = ", ".join(f"{name} = EXCLUDED.{name}" for name in names if name != '"id"')
                conflict = f" ON CONFLICT (id) DO UPDATE SET {updates}" if upsert and updates else ""
                row = await conn.fetchrow(
                    f"INSERT INTO {self._table(model)} ({', '.join(names)}) VALUES ({placeholders})"
                    f"{conflict} RETURNING *", *values
                )
                output.append(dict(row))
        return output

    async def insert(self, model_class: type[BaseIndexModel], documents: list[dict[str, Any]]) -> list[dict]:
        return await self._write(model_class, documents, False)

    async def update(self, model_class: type[BaseIndexModel], documents: list[dict[str, Any]]) -> list:
        rows = await self._write(model_class, documents, True)
        return [row["id"] for row in rows]

    async def bulk_upsert(self, model_class: type[BaseIndexModel], documents: list[dict[str, Any]]) -> list[dict]:
        return await self._write(model_class, documents, True)

    async def delete(self, model_class: type[BaseIndexModel], ids: list[str]) -> None:
        if ids:
            async with self.pool.acquire() as conn:
                await conn.execute(f"DELETE FROM {self._table(model_class)} WHERE id = ANY($1)", ids)

    async def delete_by_query(self, model_class: type[BaseIndexModel], filter_clause: FilterClause) -> None:
        where, args = self._where(filter_clause)
        if not where:
            raise ValueError("delete_by_query requires a non-empty filter")
        async with self.pool.acquire() as conn:
            await conn.execute(f"DELETE FROM {self._table(model_class)} WHERE {where}", *args)

    async def count(self, model_class: type[BaseIndexModel], filter_clause: FilterClause | None) -> int:
        where, args = self._where(filter_clause)
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                f"SELECT count(*) FROM {self._table(model_class)}" + (f" WHERE {where}" if where else ""), *args
            )

    async def _dense(self, model: type[BaseIndexModel], query: DenseSearchClause, filters: FilterClause | None,
                     limit: int, offset: int) -> list[tuple[dict, float]]:
        field = model.Meta.dense_vector_field
        if not field or not model.Meta.dense_vector_dimension or len(query.vector) != model.Meta.dense_vector_dimension:
            raise ValueError("Dense vector is missing or its dimension does not match the index")
        op = {"cosine": "<=>", "l2": "<->", "ip": "<#>"}[self.extra_config.vector_metric]
        vector = "[" + ",".join(str(float(v)) for v in query.vector) + "]"
        where, args = self._where(filters, 2)
        sql = f"SELECT *, {self._q(field)} {op} $1::vector AS distance FROM {self._table(model)}"
        sql += f" WHERE {where}" if where else ""
        sql += f" ORDER BY {self._q(field)} {op} $1::vector LIMIT {int(limit)} OFFSET {int(offset)}"
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, vector, *args)
        return [(dict(row), float(1 - row["distance"]) if self.extra_config.vector_metric == "cosine"
                 else float(-row["distance"] if self.extra_config.vector_metric == "ip" else row["distance"])) for row in rows]

    async def _sparse(self, model: type[BaseIndexModel], query: SparseSearchClause, filters: FilterClause | None,
                      limit: int, offset: int) -> list[tuple[dict, float]]:
        field = query.field_name or ("content" if "content" in model.model_fields else None)
        if not field or field not in model.model_fields or model.model_fields[field].annotation is not str:
            raise ValueError("Sparse search requires a text field")
        cfg = self.extra_config.text_search_config
        where, args = self._where(filters, 3)
        sql = (f"SELECT *, ts_rank_cd(to_tsvector('{cfg}', coalesce({self._q(field)}, '')), "
               f"plainto_tsquery('{cfg}', $1)) AS score FROM {self._table(model)} "
               f"WHERE to_tsvector('{cfg}', coalesce({self._q(field)}, '')) @@ plainto_tsquery('{cfg}', $1)")
        sql += f" AND {where}" if where else ""
        sql += f" ORDER BY score DESC LIMIT {int(limit)} OFFSET {int(offset)}"
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, query.query_text, *args)
        return [(dict(row), float(row["score"])) for row in rows if row["score"] >= query.min_score]

    async def search(self, model_class: type[BaseIndexModel],
                     query_clause: DenseSearchClause | SparseSearchClause | HybridSearchClause,
                     filter_clause: FilterClause | None = None, limit: int = 10, offset: int = 0) -> list[tuple[dict, float]]:
        if isinstance(query_clause, DenseSearchClause):
            return await self._dense(model_class, query_clause, filter_clause, limit, offset)
        if isinstance(query_clause, SparseSearchClause):
            return await self._sparse(model_class, query_clause, filter_clause, limit, offset)
        dense = await self._dense(model_class, query_clause.dense, filter_clause, limit * 3, 0)
        sparse = await self._sparse(model_class, query_clause.sparse, filter_clause, limit * 3, 0)
        fused: dict[str, tuple[dict, float]] = {}
        for hits, weight in ((dense, query_clause.weight_dense), (sparse, query_clause.weight_sparse)):
            for rank, (doc, _) in enumerate(hits, 1):
                key = str(doc["id"])
                old = fused.get(key, (doc, 0.0))
                fused[key] = (doc, old[1] + weight / (60 + rank))
        return sorted(fused.values(), key=lambda item: item[1], reverse=True)[offset:offset + limit]

    async def search_cursor(self, model_class: type[BaseIndexModel],
                            query_clause: DenseSearchClause | SparseSearchClause | HybridSearchClause,
                            filter_clause: FilterClause | None = None, page_size: int = 100,
                            cursor: str | None = None) -> tuple[list[tuple[dict, float]], str | None]:
        offset = int(cursor or 0)
        rows = await self.search(model_class, query_clause, filter_clause, page_size + 1, offset)
        return rows[:page_size], str(offset + page_size) if len(rows) > page_size else None

    async def health_check(self) -> bool:
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            logger.exception("PostgreSQL indexing health check failed")
            return False
