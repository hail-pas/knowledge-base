"""Milvus Provider 实现"""

from datetime import datetime
from typing import Type, Dict, Any, List, Optional

from ext.indexing.base import BaseProvider, BaseIndexModel
from ext.indexing.providers.types import MilvusConfig
from ext.indexing.types import DenseSearchClause, SparseSearchClause, HybridSearchClause, FilterClause

from ext.ext_tortoise.models.knowledge_base import IndexingBackendConfig
from pymilvus import AsyncMilvusClient, FieldSchema, CollectionSchema, DataType
from pymilvus.milvus_client.index import IndexParams


class MilvusProvider(BaseProvider[MilvusConfig]):
    """Milvus Provider 实现"""

    def __init__(self, config: "IndexingBackendConfig"):
        super().__init__(config, MilvusConfig)

        # Validate configuration
        if self.extra_config.index_type not in ["HNSW", "IVF_FLAT", "IVF_SQ8", "IVF_PQ"]:
            raise ValueError(
                f"Invalid index_type: {self.extra_config.index_type}. Must be one of: HNSW, IVF_FLAT, IVF_SQ8, IVF_PQ"
            )

        if self.extra_config.metric_type not in ["COSINE", "L2", "IP"]:
            raise ValueError(f"Invalid metric_type: {self.extra_config.metric_type}. Must be one of: COSINE, L2, IP")

    async def connect(self):
        """建立连接（Milvus SDK 是同步的，需要用 executor）"""
        scheme = "https" if self.config.verify_ssl else "http"
        uri = (
            f"{scheme}://{self.config.host}:{self.config.port}"
            if self.config.port
            else f"{scheme}://{self.config.host}"
        )

        kwargs = {
            "uri": uri,
            "secure": self.config.verify_ssl,
            "timeout": self.config.timeout or 30,
        }

        if self.config.username and self.config.password:
            kwargs["user"] = self.config.username
            kwargs["password"] = self.config.password

        if self.extra_config.db_name:
            kwargs["db_name"] = self.extra_config.db_name

        self._client = AsyncMilvusClient(**kwargs)

    async def disconnect(self):
        """断开连接"""
        if self._client:
            await self._client.close()

    async def create_collection(self, model_class: Type[BaseIndexModel], drop_existing: bool = False):
        """创建集合"""
        # Validate model configuration
        self._validate_model_config(model_class)

        collection_name = self.build_collection_name(model_class)

        if drop_existing:
            await self._client.drop_collection(collection_name)

        fields = []

        # 如果有 partition_key，添加为 Partition Key 字段
        if model_class.Meta.partition_key:
            fields.append(
                FieldSchema(
                    name=model_class.Meta.partition_key, dtype=DataType.VARCHAR, max_length=255, is_partition_key=True
                )
            )

        for field_name, field_info in model_class.model_fields.items():
            if field_name.startswith("_") or field_name == "id":
                continue

            # 跳过 partition_key 字段（已添加）
            if model_class.Meta.partition_key and field_name == model_class.Meta.partition_key:
                continue

            if field_name == model_class.Meta.dense_vector_field:
                fields.append(FieldSchema(name=field_name, dtype=DataType.FLOAT_VECTOR, dim=model_class.Meta.dense_vector_dimension))
            elif field_name == model_class.Meta.sparse_vector_field:
                fields.append(FieldSchema(name=field_name, dtype=DataType.SPARSE_FLOAT_VECTOR))
            elif field_info.annotation == bool:
                fields.append(FieldSchema(name=field_name, dtype=DataType.BOOL))
            elif field_info.annotation == int:
                fields.append(FieldSchema(name=field_name, dtype=DataType.INT64))
            elif field_info.annotation == float:
                fields.append(FieldSchema(name=field_name, dtype=DataType.DOUBLE))
            elif field_info.annotation == str:
                fields.append(
                    FieldSchema(
                        name=field_name, dtype=DataType.VARCHAR, max_length=65535, nullable=not field_info.is_required
                    )
                )
            elif field_info.annotation == datetime:
                # Store datetime as VARCHAR for compatibility (ISO 8601 format)
                fields.append(
                    FieldSchema(name=field_name, dtype=DataType.TIMESTAMPTZ, nullable=not field_info.is_required)
                )
            elif field_info.annotation in [dict, Dict, List]:
                fields.append(FieldSchema(name=field_name, dtype=DataType.JSON, nullable=not field_info.is_required))
            else:
                raise RuntimeError(f"Not supported field type: {field_name} - {field_info.annotation}")

        schema = CollectionSchema(fields=fields, description=collection_name, enable_dynamic_field=True)

        # Create index on vector field separately
        index_params = IndexParams()

        # Add HNSW-specific parameters
        if model_class.Meta.dense_vector_field:
            if self.extra_config.index_type == "HNSW":
                index_params.add_index(
                    field_name=model_class.Meta.dense_vector_field,
                    index_type=self.extra_config.index_type,
                    metric_type=self.extra_config.metric_type,
                    params={
                        "M": self.extra_config.M,
                        "efConstruction": self.extra_config.ef_construction,
                    },
                )
            else:
                index_params.add_index(
                    field_name=model_class.Meta.dense_vector_field,
                    index_type=self.extra_config.index_type,
                    metric_type=self.extra_config.metric_type,
                )

        if model_class.Meta.sparse_vector_field:
            index_params.add_index(
                field_name=model_class.Meta.sparse_vector_field,
                index_type="SPARSE_INVERTED_INDEX",
                index_name="sparse_inverted_index",
                metric_type=self.extra_config.metric_type,
                params={"inverted_index_algo": self.extra_config.inverted_index_algo},
            )

        # Use schema-based API for custom field support
        await self._client.create_collection(
            collection_name=collection_name,
            auto_id=model_class.Meta.auto_generate_id,
            schema=schema,
            index_params=index_params,
        )

        await self._client.create_index(
            collection_name=collection_name,
            index_params=index_params,
        )

    async def drop_collection(self, model_class: Type[BaseIndexModel]):
        """删除集合（使用 Partition Key，无需手动删除 partition）"""
        collection_name = self.build_collection_name(model_class)

        await self._client.drop_collection(collection_name)

    async def get(self, model_class: Type[BaseIndexModel], ids: list) -> list[dict]:
        """获取单个文档"""
        collection_name = self.build_collection_name(model_class)

        return await self._client.get(collection_name=collection_name, ids=ids)

    async def filter(
        self,
        model_class: Type[BaseIndexModel],
        filter_clause: Optional[FilterClause],
        limit: int = 10,
        offset: int = 0,
        sort: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """过滤查询"""
        collection_name = self.build_collection_name(model_class)

        filter_expr = self._convert_filter(filter_clause) if filter_clause else ""

        results = await self._client.query(
            collection_name=collection_name, filter=filter_expr, output_fields=["*"], limit=limit, offset=offset
        )

        return results

    async def insert(
        self, model_class: Type[BaseIndexModel], documents: List[Dict[str, Any]]
    ) -> Optional[List[Dict[str, Any]]]:
        """插入文档（批量）"""
        if not documents:
            return None

        collection_name = self.build_collection_name(model_class)

        result = await self._client.insert(collection_name=collection_name, data=documents)

        if result and result.get("insert_count", 0) > 0:
            return documents

        return None

    async def update(self, model_class: Type[BaseIndexModel], documents: List[Dict[str, Any]]):
        """更新文档（使用 upsert）"""
        if not documents:
            return

        collection_name = self.build_collection_name(model_class)

        await self._client.upsert(collection_name=collection_name, data=documents)

    async def bulk_upsert(
        self, model_class: Type[BaseIndexModel], documents: List[Dict[str, Any]]
    ) -> Optional[List[Dict[str, Any]]]:
        """批量插入或更新（upsert）"""
        if not documents:
            return None

        collection_name = self.build_collection_name(model_class)

        result = await self._client.upsert(collection_name=collection_name, data=documents)

        if result and result.get("upsert_count", 0) > 0:
            return documents

        return None

    async def delete(self, model_class: Type[BaseIndexModel], ids: List[str]):
        """删除文档（带 partition）"""
        if not ids:
            return

        collection_name = self.build_collection_name(model_class)

        ids_str = [str(id_) for id_ in ids]

        await self._client.delete(collection_name=collection_name, ids=ids_str)

    async def delete_by_query(self, model_class: Type[BaseIndexModel], filter_clause: FilterClause):
        """根据条件删除（带 partition）"""
        collection_name = self.build_collection_name(model_class)

        filter_expr = self._convert_filter(filter_clause)

        await self._client.delete(collection_name=collection_name, filter=filter_expr)

    async def count(self, model_class: Type[BaseIndexModel], filter_clause: Optional[FilterClause]) -> int:
        """统计文档数量"""
        collection_name = self.build_collection_name(model_class)

        filter_expr = self._convert_filter(filter_clause) if filter_clause else ""

        result = await self._client.query(
            collection_name=collection_name, filter=filter_expr, output_fields=["count(*)"]
        )

        return len(result)

    async def search(
        self,
        model_class: Type[BaseIndexModel],
        query_clause: DenseSearchClause | SparseSearchClause | HybridSearchClause,
        filter_clause: Optional[FilterClause] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> List[tuple[Dict[str, Any], float]]:
        """搜索"""
        collection_name = self.build_collection_name(model_class)

        filter_expr = self._convert_filter(filter_clause) if filter_clause else None

        if isinstance(query_clause, DenseSearchClause):
            if not model_class.Meta.dense_vector_field:
                raise RuntimeError("Dense search requires dense vector field")

            results = await self._client.search(
                collection_name=collection_name,
                data=[query_clause.vector],
                anns_field=model_class.Meta.dense_vector_field,
                param={"metric_type": self.extra_config.metric_type, "params": {}},
                limit=limit + offset,
                expr=filter_expr,
                output_fields=["*"],
            )

        elif isinstance(query_clause, SparseSearchClause):
            if not model_class.Meta.sparse_vector_field:
                raise RuntimeError("Sparse search requires sparse vector field")

            sparse_vector_list = [[query_clause.sparse_vector]]

            results = await self._client.search(
                collection_name=collection_name,
                data=sparse_vector_list,
                anns_field=model_class.Meta.sparse_vector_field,
                param={"metric_type": self.extra_config.metric_type, "params": {"drop_ratio_build": 0.0}},
                limit=limit + offset,
                expr=filter_expr,
                output_fields=["*"],
            )

        elif isinstance(query_clause, HybridSearchClause):
            if not model_class.Meta.dense_vector_field or not model_class.Meta.sparse_vector_field:
                raise RuntimeError("Hybrid search requires both dense and sparse vector fields")

            dense_results = await self._client.search(
                collection_name=collection_name,
                data=[query_clause.dense.vector],
                anns_field=model_class.Meta.dense_vector_field,
                param={"metric_type": self.extra_config.metric_type, "params": {}},
                limit=query_clause.dense.top_k * 2,
                expr=filter_expr,
                output_fields=["*"],
            )

            sparse_results = await self._client.search(
                collection_name=collection_name,
                data=[[query_clause.sparse.sparse_vector]],
                anns_field=model_class.Meta.sparse_vector_field,
                param={"metric_type": self.extra_config.metric_type, "params": {"drop_ratio_build": 0.0}},
                limit=query_clause.sparse.top_k * 2,
                expr=filter_expr,
                output_fields=["*"],
            )

            dense_dict = {result["id"]: (result, result["distance"]) for result in dense_results[0]}
            sparse_dict = {result["id"]: (result, result["distance"]) for result in sparse_results[0]}

            all_ids = set(dense_dict.keys()) | set(sparse_dict.keys())

            combined_results = []
            for id_ in all_ids:
                dense_doc, dense_score = dense_dict.get(id_, ({}, 0))
                sparse_doc, sparse_score = sparse_dict.get(id_, ({}, 0))

                final_doc = dense_doc if dense_doc else sparse_doc
                final_score = query_clause.weight_dense * dense_score + query_clause.weight_sparse * sparse_score

                combined_results.append((final_doc, final_score))

            combined_results.sort(key=lambda x: x[1], reverse=True)

            return combined_results[offset : offset + limit]

        else:
            raise RuntimeError(f"Unsupported query clause type: {type(query_clause)}")

        final_results = []
        if results and len(results) > 0:
            for result in results[0][offset : offset + limit]:
                final_results.append((result["entity"], result["distance"]))

        return final_results

    async def search_cursor(
        self,
        model_class: Type[BaseIndexModel],
        query_clause: DenseSearchClause | SparseSearchClause | HybridSearchClause,
        filter_clause: Optional[FilterClause] = None,
        page_size: int = 100,
        cursor: Optional[str] = None,
    ) -> tuple[List[tuple[Dict[str, Any], float]], Optional[str]]:
        """搜索（Cursor 方式）"""
        collection_name = self.build_collection_name(model_class)

        filter_expr = self._convert_filter(filter_clause) if filter_clause else None

        offset = 0
        if cursor:
            try:
                offset = int(cursor)
            except (ValueError, TypeError):
                offset = 0

        if isinstance(query_clause, DenseSearchClause):
            if not model_class.Meta.dense_vector_field:
                raise RuntimeError("Dense search requires dense vector field")

            results = await self._client.search(
                collection_name=collection_name,
                data=[query_clause.vector],
                anns_field=model_class.Meta.dense_vector_field,
                param={"metric_type": self.extra_config.metric_type, "params": {}},
                limit=page_size,
                expr=filter_expr,
                output_fields=["*"],
            )

        elif isinstance(query_clause, SparseSearchClause):
            if not model_class.Meta.sparse_vector_field:
                raise RuntimeError("Sparse search requires sparse vector field")

            sparse_vector_list = [[query_clause.sparse_vector]]

            results = await self._client.search(
                collection_name=collection_name,
                data=sparse_vector_list,
                anns_field=model_class.Meta.sparse_vector_field,
                param={"metric_type": self.extra_config.metric_type, "params": {"drop_ratio_build": 0.0}},
                limit=page_size,
                expr=filter_expr,
                output_fields=["*"],
            )

        elif isinstance(query_clause, HybridSearchClause):
            if not model_class.Meta.dense_vector_field or not model_class.Meta.sparse_vector_field:
                raise RuntimeError("Hybrid search requires both dense and sparse vector fields")

            dense_results = await self._client.search(
                collection_name=collection_name,
                data=[query_clause.dense.vector],
                anns_field=model_class.Meta.dense_vector_field,
                param={"metric_type": self.extra_config.metric_type, "params": {}},
                limit=query_clause.dense.top_k * 2,
                expr=filter_expr,
                output_fields=["*"],
            )

            sparse_results = await self._client.search(
                collection_name=collection_name,
                data=[[query_clause.sparse.sparse_vector]],
                anns_field=model_class.Meta.sparse_vector_field,
                param={"metric_type": self.extra_config.metric_type, "params": {"drop_ratio_build": 0.0}},
                limit=query_clause.sparse.top_k * 2,
                expr=filter_expr,
                output_fields=["*"],
            )

            dense_dict = {result["id"]: (result, result["distance"]) for result in dense_results[0]}
            sparse_dict = {result["id"]: (result, result["distance"]) for result in sparse_results[0]}

            all_ids = set(dense_dict.keys()) | set(sparse_dict.keys())

            combined_results = []
            for id_ in all_ids:
                dense_doc, dense_score = dense_dict.get(id_, ({}, 0))
                sparse_doc, sparse_score = sparse_dict.get(id_, ({}, 0))

                final_doc = dense_doc if dense_doc else sparse_doc
                final_score = query_clause.weight_dense * dense_score + query_clause.weight_sparse * sparse_score

                combined_results.append((final_doc, final_score))

            combined_results.sort(key=lambda x: x[1], reverse=True)

            final_offset = offset % len(combined_results) if combined_results else 0
            paginated_results = combined_results[final_offset : final_offset + page_size]

            next_cursor = str(final_offset + len(paginated_results)) if paginated_results else None

            return paginated_results, next_cursor

        else:
            raise RuntimeError(f"Unsupported query clause type: {type(query_clause)}")

        final_results = []
        if results and len(results) > 0:
            for result in results[0]:
                final_results.append((result["entity"], result["distance"]))

        next_cursor = str(offset + len(final_results)) if final_results else None

        return final_results, next_cursor

    async def health_check(self) -> bool:
        """健康检查"""
        await self._client.list_databases()
        return True

    async def exists(self, model_class: Type[BaseIndexModel], id: str | int) -> bool:
        """检查文档是否存在"""
        collection_name = self.build_collection_name(model_class)

        result = await self._client.get(collection_name=collection_name, ids=[id])
        return len(result) > 0

    def _convert_filter(self, filter_clause: FilterClause) -> str:
        """转换 FilterClause 为 Milvus 表达式"""
        conditions = []

        if filter_clause.equals:
            for field, value in filter_clause.equals.items():
                if isinstance(value, str):
                    escaped_value = self._escape_string(value)
                    conditions.append(f'{field} == "{escaped_value}"')
                else:
                    conditions.append(f"{field} == {value}")

        if filter_clause.in_list:
            for field, values in filter_clause.in_list.items():
                if values:
                    if all(isinstance(v, str) for v in values):
                        formatted_values = ", ".join(f'"{self._escape_string(v)}"' for v in values)
                    else:
                        formatted_values = ", ".join(str(v) for v in values)
                    conditions.append(f"{field} in [{formatted_values}]")

        if filter_clause.range:
            for field, range_value in filter_clause.range.items():
                for op, value in range_value.items():
                    if isinstance(value, str):
                        escaped_value = self._escape_string(value)
                        conditions.append(f'{field} {op} "{escaped_value}"')
                    else:
                        conditions.append(f"{field} {op} {value}")

        if filter_clause.and_conditions:
            and_conditions = [self._convert_filter(f) for f in filter_clause.and_conditions]
            if and_conditions:
                conditions.append(f"({' and '.join(and_conditions)})")

        if filter_clause.or_conditions:
            or_conditions = [self._convert_filter(f) for f in filter_clause.or_conditions]
            if or_conditions:
                conditions.append(f"({' or '.join(or_conditions)})")

        return " and ".join(conditions) if conditions else ""

    def _escape_string(self, value: str) -> str:
        """转义 Milvus 表达式中的字符串特殊字符"""
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return escaped
