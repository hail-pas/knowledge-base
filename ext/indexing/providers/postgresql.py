"""PostgreSQL indexing backend using pgvector and PostgreSQL full-text search."""
from __future__ import annotations

import json
import re
import ssl
from datetime import date, datetime
from typing import Any, Literal, get_args, get_origin

import asyncpg
from loguru import logger
from pydantic import Field

from ext.indexing.base import BaseIndexModel, BaseProvider
from ext.indexing.providers.types import ProviderConfig
from ext.indexing.types import DenseSearchClause, FilterClause, SparseSearchClause
from ext.ext_tortoise.models.knowledge_base import IndexingBackendConfig


class PostgreSQLConfig(ProviderConfig):
    database: str = "postgres"
    schema: str = Field(default="public", pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    vector_metric: Literal["cosine", "l2", "ip"] = "cosine"
    vector_index: Literal["hnsw", "ivfflat"] = "hnsw"
    text_search_config: str = Field(
        default="simple",
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)?$",
    )
    pool_min_size: int = Field(default=1, ge=1)
    pool_max_size: int | None = Field(default=None, ge=1)
    hnsw_m: int = Field(default=16, ge=2, le=100)
    hnsw_ef_construction: int = Field(default=64, ge=4)
    ivfflat_lists: int = Field(default=100, ge=1)


class PostgreSQLProvider(BaseProvider[PostgreSQLConfig]):
    """One logical index maps to a typed PostgreSQL table; dimensions are part of its physical name."""

    def __init__(self, config: IndexingBackendConfig) -> None:
        super().__init__(config, PostgreSQLConfig)
        if self.extra_config.vector_metric not in {"cosine", "l2", "ip"}:
            raise ValueError("vector_metric must be cosine, l2 or ip")
        if self.extra_config.vector_index not in {"hnsw", "ivfflat"}:
            raise ValueError("vector_index must be hnsw or ivfflat")
        max_pool_size = self.extra_config.pool_max_size or config.max_connections
        if max_pool_size < self.extra_config.pool_min_size:
            raise ValueError("pool_max_size must be greater than or equal to pool_min_size")
        if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", self.extra_config.schema):
            raise ValueError("Invalid schema")
        if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)?", self.extra_config.text_search_config):
            raise ValueError("Invalid PostgreSQL text search configuration")
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        ssl_context = None
        if self.config.use_ssl:
            ssl_context = ssl.create_default_context()
            if not self.config.verify_ssl:
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
        self._pool = await asyncpg.create_pool(
            host=self.config.host, port=self.config.port or 5432, user=self.config.username,
            password=self.password, database=self.extra_config.database,
            min_size=self.extra_config.pool_min_size, max_size=self.extra_config.pool_max_size or self.config.max_connections,
            timeout=self.config.timeout, command_timeout=self.config.timeout,
            ssl=ssl_context or False,
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
                if not dimension or dimension > 4000:
                    raise ValueError("pgvector supports at most 4000 dimensions with halfvec")
                kind = f"{'vector' if dimension <= 2000 else 'halfvec'}({dimension})"
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
                prefix = "vector" if model_class.Meta.dense_vector_dimension <= 2000 else "halfvec"
                ops = {metric: f"{prefix}_{metric}_ops" for metric in ("cosine", "l2", "ip")}
                index_type = self.extra_config.vector_index.lower()
                idx = self._q(
                    f"{model_class.Meta.index_name}_{model_class.Meta.dense_vector_dimension}_{index_type}"
                )
                options = (
                    f" WITH (m = {self.extra_config.hnsw_m}, "
                    f"ef_construction = {self.extra_config.hnsw_ef_construction})"
                    if index_type == "hnsw"
                    else f" WITH (lists = {self.extra_config.ivfflat_lists})"
                )
                await conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {idx} ON {table} USING {index_type} "
                    f"({self._q(vector_field)} {ops[self.extra_config.vector_metric]}){options}"
                )
            for name, field in model_class.model_fields.items():
                extra = field.json_schema_extra or {}
                metadata = extra.get("index_metadata", {}) if isinstance(extra, dict) else {}
                if (
                    not model_class.Meta.dense_vector_field
                    and field.annotation is str
                    and metadata.get("enable_keyword", name == "content")
                ):
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
        next_position = start + len(values)
        for nested_clause in clause.and_conditions or []:
            sql, params = self._where(nested_clause, next_position)
            if sql:
                terms.append(f"({sql})")
                values.extend(params)
                next_position += len(params)
        if clause.or_conditions:
            nested = []
            for item in clause.or_conditions:
                sql, params = self._where(item, next_position)
                if sql:
                    nested.append(sql)
                    values.extend(params)
                    next_position += len(params)
            if nested:
                terms.append("(" + " OR ".join(f"({sql})" for sql in nested) + ")")
        return " AND ".join(terms), values

    @staticmethod
    def _decode(model: type[BaseIndexModel], row: Any) -> dict[str, Any]:
        document = dict(row)
        for name, field in model.model_fields.items():
            value = document.get(name)
            origin = get_origin(field.annotation)
            if isinstance(value, str) and (origin in (list, dict) or field.annotation in (list, dict)):
                document[name] = json.loads(value)
        vector_field = model.Meta.dense_vector_field
        if vector_field and isinstance(document.get(vector_field), str):
            document[vector_field] = json.loads(document[vector_field])
        return document

    async def get(self, model_class: type[BaseIndexModel], ids: list) -> list[dict]:
        if not ids:
            return []
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(f"SELECT * FROM {self._table(model_class)} WHERE id = ANY($1)", ids)
            return [self._decode(model_class, row) for row in rows]

    async def filter(self, model_class: type[BaseIndexModel], filter_clause: FilterClause | None,
                     limit: int = 10, offset: int = 0, sort: str | None = None) -> list[dict[str, Any]]:
        where, args = self._where(filter_clause)
        order = ""
        if sort:
            order = f" ORDER BY {self._q(sort.lstrip('-'))} {'DESC' if sort.startswith('-') else 'ASC'}"
        sql = f"SELECT * FROM {self._table(model_class)}" + (f" WHERE {where}" if where else "")
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql + order + f" LIMIT {int(limit)} OFFSET {int(offset)}", *args)
            return [self._decode(model_class, row) for row in rows]

    def _encode(self, model: type[BaseIndexModel], doc: dict[str, Any]) -> tuple[list[str], list[Any], list[str]]:
        names, values, casts = [], [], []
        for key, value in doc.items():
            names.append(self._q(key))
            if key == model.Meta.dense_vector_field and value is not None:
                if len(value) != model.Meta.dense_vector_dimension:
                    raise ValueError(f"Vector dimension mismatch for {key}")
                values.append("[" + ",".join(str(float(v)) for v in value) + "]")
                casts.append("::vector" if model.Meta.dense_vector_dimension <= 2000 else "::halfvec")
            elif isinstance(value, (dict, list)):
                values.append(json.dumps(value, ensure_ascii=False))
                casts.append("::jsonb")
            else:
                field = model.model_fields.get(key)
                annotation = field.annotation if field else None
                origin, args = get_origin(annotation), get_args(annotation)
                if origin is not None and type(None) in args:
                    annotation = next((arg for arg in args if arg is not type(None)), annotation)
                if annotation is datetime and isinstance(value, str):
                    value = datetime.fromisoformat(value.replace("Z", "+00:00"))
                elif annotation is date and isinstance(value, str):
                    value = date.fromisoformat(value)
                values.append(value)
                casts.append("")
        return names, values, casts

    async def _write(
        self, model: type[BaseIndexModel], documents: list[dict[str, Any]],
        upsert: bool, update_only: bool = False,
    ) -> list[dict]:
        if not documents:
            return []
        output = []
        async with self.pool.acquire() as conn, conn.transaction():
            for raw in documents:
                doc = dict(raw)
                if update_only:
                    if "id" not in doc:
                        raise ValueError("id is required for update")
                    document_id = doc.pop("id")
                    names, values, casts = self._encode(model, doc)
                    if not names:
                        continue
                    assignments = ", ".join(
                        f"{name} = ${index}{cast}"
                        for index, (name, cast) in enumerate(zip(names, casts, strict=True), 1)
                    )
                    values.append(document_id)
                    row = await conn.fetchrow(
                        f"UPDATE {self._table(model)} SET {assignments} "
                        f"WHERE id = ${len(values)} RETURNING *", *values
                    )
                    output.append(self._decode(model, row) if row else {"id": None})
                    continue
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
                output.append(self._decode(model, row))
        return output

    async def insert(self, model_class: type[BaseIndexModel], documents: list[dict[str, Any]]) -> list[dict]:
        return await self._write(model_class, documents, False)

    async def update(self, model_class: type[BaseIndexModel], documents: list[dict[str, Any]]) -> list:
        rows = await self._write(model_class, documents, False, update_only=True)
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
        vector_type = "vector" if model.Meta.dense_vector_dimension <= 2000 else "halfvec"
        where, args = self._where(filters, 2)
        sql = f"SELECT *, {self._q(field)} {op} $1::{vector_type} AS distance FROM {self._table(model)}"
        sql += f" WHERE {where}" if where else ""
        sql += f" ORDER BY {self._q(field)} {op} $1::{vector_type} LIMIT {int(limit)} OFFSET {int(offset)}"
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, vector, *args)
        return [
            (
                self._decode(model, row),
                float(1 - row["distance"])
                if self.extra_config.vector_metric == "cosine"
                else float(-row["distance"])
                if self.extra_config.vector_metric == "ip"
                else float(1 / (1 + row["distance"])),
            )
            for row in rows
        ]

    async def _sparse(self, model: type[BaseIndexModel], query: SparseSearchClause, filters: FilterClause | None,
                      limit: int, offset: int) -> list[tuple[dict, float]]:
        field = query.field_name or ("content" if "content" in model.model_fields else None)
        if not field or field not in model.model_fields or model.model_fields[field].annotation is not str:
            raise ValueError("Sparse search requires a text field")
        cfg = self.extra_config.text_search_config
        where, args = self._where(filters, 3)
        rank_expression = (
            f"ts_rank_cd(to_tsvector('{cfg}', coalesce({self._q(field)}, '')), "
            f"plainto_tsquery('{cfg}', $1))"
        )
        sql = (
            f"SELECT *, {rank_expression} AS score FROM {self._table(model)} "
            f"WHERE to_tsvector('{cfg}', coalesce({self._q(field)}, '')) @@ plainto_tsquery('{cfg}', $1) "
            f"AND {rank_expression} >= $2"
        )
        sql += f" AND {where}" if where else ""
        sql += f" ORDER BY score DESC LIMIT {int(limit)} OFFSET {int(offset)}"
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, query.query_text, query.min_score, *args)
        return [(self._decode(model, row), float(row["score"])) for row in rows]

    async def search_native(self, model_class: type[BaseIndexModel],
                     query_clause: DenseSearchClause | SparseSearchClause,
                     filter_clause: FilterClause | None = None, limit: int = 10, offset: int = 0) -> list[tuple[dict, float]]:
        if isinstance(query_clause, DenseSearchClause):
            return await self._dense(model_class, query_clause, filter_clause, limit, offset)
        if isinstance(query_clause, SparseSearchClause):
            return await self._sparse(model_class, query_clause, filter_clause, limit, offset)
        raise TypeError("PostgreSQL native search accepts DenseSearchClause or SparseSearchClause")

    async def health_check(self) -> bool:
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            logger.exception("PostgreSQL indexing health check failed")
            return False
