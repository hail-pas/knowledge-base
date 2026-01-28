"""Milvus Provider 实现"""

from datetime import datetime
from typing import Type, Dict, Any, List, Optional, get_origin, get_args

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
                f"Invalid index_type: {self.extra_config.index_type}. Must be one of: HNSW, IVF_FLAT, IVF_SQ8, IVF_PQ",
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

    async def create_collection(self, model_class: type[BaseIndexModel], drop_existing: bool = False):
        """创建集合"""
        self._validate_model_config(model_class)

        collection_name = self.build_collection_name(model_class)

        if drop_existing:
            await self._client.drop_collection(collection_name)

        # 构建字段列表
        fields = []
        fields.append(self._build_primary_key_field(model_class))

        # 添加 partition key（如果有）
        if model_class.Meta.partition_key:
            fields.append(self._build_partition_key_field(model_class))

        # 添加其他字段
        for field_name, field_info in model_class.model_fields.items():
            if self._should_skip_field(field_name, model_class):
                continue
            field_schema = self._build_field_schema(field_name, field_info, model_class)
            if field_schema:
                fields.append(field_schema)

        # 创建 schema 和索引
        schema = CollectionSchema(fields=fields, description=collection_name, enable_dynamic_field=True)
        index_params = self._build_index_params(model_class)

        # 创建 collection
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

    def _build_primary_key_field(self, model_class: type[BaseIndexModel]) -> FieldSchema:
        """构建 primary key 字段"""
        id_field_info = model_class.model_fields.get("id")
        id_type = DataType.INT64

        # 确定 id 字段类型
        if id_field_info and id_field_info.annotation:
            from typing import Union  # type: ignore

            origin = get_origin(id_field_info.annotation)
            if origin is Union:
                args = get_args(id_field_info.annotation)
                id_type = DataType.INT64 if int in args else DataType.VARCHAR
            elif id_field_info.annotation == str:
                id_type = DataType.VARCHAR
            elif id_field_info.annotation == int:
                id_type = DataType.INT64

        return FieldSchema(name="id", dtype=id_type, is_primary=True, auto_id=model_class.Meta.auto_generate_id)

    def _build_partition_key_field(self, model_class: type[BaseIndexModel]) -> FieldSchema | None:
        """构建 partition key 字段"""
        if not model_class.Meta.partition_key:
            return None

        return FieldSchema(
            name=model_class.Meta.partition_key, dtype=DataType.VARCHAR, max_length=255, is_partition_key=True,
        )

    def _should_skip_field(self, field_name: str, model_class: type[BaseIndexModel]) -> bool:
        """判断是否应该跳过该字段"""
        if field_name.startswith("_") or field_name == "id":
            return True
        if model_class.Meta.partition_key and field_name == model_class.Meta.partition_key:
            return True
        return False

    def _build_field_schema(
        self, field_name: str, field_info, model_class: type[BaseIndexModel],
    ) -> FieldSchema | None:
        """构建单个字段的 schema"""
        # 向量字段
        if field_name == model_class.Meta.dense_vector_field:
            return FieldSchema(
                name=field_name, dtype=DataType.FLOAT_VECTOR, dim=model_class.Meta.dense_vector_dimension,
            )
        if field_name == model_class.Meta.sparse_vector_field:
            return FieldSchema(name=field_name, dtype=DataType.SPARSE_FLOAT_VECTOR)

        # 标量字段
        if field_info.annotation == bool:
            return FieldSchema(name=field_name, dtype=DataType.BOOL)
        if field_info.annotation == int:
            return FieldSchema(name=field_name, dtype=DataType.INT64)
        if field_info.annotation == float:
            return FieldSchema(name=field_name, dtype=DataType.DOUBLE)
        if field_info.annotation == str:
            return FieldSchema(
                name=field_name, dtype=DataType.VARCHAR, max_length=65535, nullable=not field_info.is_required,
            )
        if field_info.annotation == datetime:
            return FieldSchema(name=field_name, dtype=DataType.TIMESTAMPTZ, nullable=not field_info.is_required)
        if field_info.annotation in [dict, dict] or get_origin(field_info.annotation) == dict:
            return FieldSchema(name=field_name, dtype=DataType.JSON, nullable=not field_info.is_required)

        # ARRAY 字段
        if get_origin(field_info.annotation) == list:
            return self._build_array_field_schema(field_name, field_info)

        raise RuntimeError(f"Not supported field type: {field_name} - {field_info.annotation}")

    def _build_array_field_schema(self, field_name: str, field_info) -> FieldSchema:
        """构建 ARRAY 类型字段的 schema"""
        args = get_args(field_info.annotation)

        # 从 json_schema_extra 读取 index_metadata
        index_metadata: dict[str, Any] = {}  # type: ignore
        if field_info.json_schema_extra and isinstance(field_info.json_schema_extra, dict):
            extra_dict = field_info.json_schema_extra
            index_metadata = extra_dict.get("index_metadata", {})  # type: ignore

        # 获取配置，提供默认值
        default_max_capacity = 100
        default_max_length = 255
        max_capacity = index_metadata.get("max_capacity", default_max_capacity)

        # Python 类型到 Milvus DataType 的映射
        element_type_map = {
            str: DataType.VARCHAR,
            int: DataType.INT64,
            float: DataType.FLOAT,
        }

        if not args or args[0] not in element_type_map:
            raise RuntimeError(f"Not supported array element type: {args[0] if args else 'unknown'}")

        element_type = element_type_map[args[0]]

        # 构建参数字典
        array_params = {
            "name": field_name,
            "dtype": DataType.ARRAY,
            "max_capacity": max_capacity,
            "element_type": element_type,
            "nullable": not field_info.is_required,
        }

        # 如果元素类型是 VARCHAR，添加 max_length
        if element_type == DataType.VARCHAR:
            max_length = index_metadata.get("max_length", default_max_length)
            array_params["max_length"] = max_length

        return FieldSchema(**array_params)

    def _build_index_params(self, model_class: type[BaseIndexModel]) -> IndexParams:
        """构建索引参数"""
        index_params = IndexParams()

        # 稠密向量索引
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

        # 稀疏向量索引
        if model_class.Meta.sparse_vector_field:
            index_params.add_index(
                field_name=model_class.Meta.sparse_vector_field,
                index_type="SPARSE_INVERTED_INDEX",
                index_name="sparse_inverted_index",
                metric_type=self.extra_config.metric_type,
                params={"inverted_index_algo": self.extra_config.inverted_index_algo},
            )

        return index_params

    async def drop_collection(self, model_class: type[BaseIndexModel]):
        """删除集合（使用 Partition Key，无需手动删除 partition）"""
        collection_name = self.build_collection_name(model_class)

        await self._client.drop_collection(collection_name)

    async def get(self, model_class: type[BaseIndexModel], ids: list) -> list[dict]:
        """获取单个文档"""
        collection_name = self.build_collection_name(model_class)

        return await self._client.get(collection_name=collection_name, ids=ids)

    async def filter(
        self,
        model_class: type[BaseIndexModel],
        filter_clause: FilterClause | None,
        limit: int = 10,
        offset: int = 0,
        sort: str | None = None,
    ) -> list[dict[str, Any]]:
        """过滤查询"""
        collection_name = self.build_collection_name(model_class)

        filter_expr = self._convert_filter(model_class, filter_clause) if filter_clause else ""

        results = await self._client.query(
            collection_name=collection_name, filter=filter_expr, output_fields=["*"], limit=limit, offset=offset,
        )

        return results

    async def insert(
        self, model_class: type[BaseIndexModel], documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        """插入文档（批量）"""
        if not documents:
            return None

        collection_name = self.build_collection_name(model_class)

        result = await self._client.insert(collection_name=collection_name, data=documents)

        if result and result.get("insert_count", 0) > 0:
            return documents

        return None

    async def update(self, model_class: type[BaseIndexModel], documents: list[dict[str, Any]]):
        """更新文档（使用 upsert）"""
        if not documents:
            return

        collection_name = self.build_collection_name(model_class)

        await self._client.upsert(collection_name=collection_name, data=documents)

    async def bulk_upsert(
        self, model_class: type[BaseIndexModel], documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        """批量插入或更新（upsert）"""
        if not documents:
            return None

        collection_name = self.build_collection_name(model_class)

        result = await self._client.upsert(collection_name=collection_name, data=documents)

        if result and result.get("upsert_count", 0) > 0:
            return documents

        return None

    async def delete(self, model_class: type[BaseIndexModel], ids: list[str]):
        """删除文档（带 partition）"""
        if not ids:
            return

        collection_name = self.build_collection_name(model_class)

        ids_str = [str(id_) for id_ in ids]

        await self._client.delete(collection_name=collection_name, ids=ids_str)

    async def delete_by_query(self, model_class: type[BaseIndexModel], filter_clause: FilterClause):
        """根据条件删除（带 partition）"""
        collection_name = self.build_collection_name(model_class)

        filter_expr = self._convert_filter(model_class, filter_clause)

        await self._client.delete(collection_name=collection_name, filter=filter_expr)

    async def count(self, model_class: type[BaseIndexModel], filter_clause: FilterClause | None) -> int:
        """统计文档数量"""
        collection_name = self.build_collection_name(model_class)

        filter_expr = self._convert_filter(model_class, filter_clause) if filter_clause else ""

        result = await self._client.query(
            collection_name=collection_name, filter=filter_expr, output_fields=["count(*)"],
        )

        return len(result)

    async def search(
        self,
        model_class: type[BaseIndexModel],
        query_clause: DenseSearchClause | SparseSearchClause | HybridSearchClause,
        filter_clause: FilterClause | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[tuple[dict[str, Any], float]]:
        """搜索"""
        filter_expr = self._convert_filter(model_class, filter_clause) if filter_clause else None

        results = await self._execute_search(model_class, query_clause, filter_expr, limit + offset)

        if isinstance(query_clause, HybridSearchClause):
            return results[offset : offset + limit]

        final_results = []
        if results and len(results) > 0:
            for result in results[0][offset : offset + limit]:
                final_results.append((result["entity"], result["distance"]))

        return final_results

    async def _execute_search(
        self,
        model_class: type[BaseIndexModel],
        query_clause: DenseSearchClause | SparseSearchClause | HybridSearchClause,
        filter_expr: str | None,
        limit: int,
    ) -> Any:
        """执行搜索（公共方法）"""
        collection_name = self.build_collection_name(model_class)

        if isinstance(query_clause, DenseSearchClause):
            return await self._execute_dense_search(collection_name, model_class, query_clause, filter_expr, limit)

        if isinstance(query_clause, SparseSearchClause):
            return await self._execute_sparse_search(collection_name, model_class, query_clause, filter_expr, limit)

        if isinstance(query_clause, HybridSearchClause):
            return await self._execute_hybrid_search(collection_name, model_class, query_clause, filter_expr)

        raise RuntimeError(f"Unsupported query clause type: {type(query_clause)}")

    async def _execute_dense_search(
        self,
        collection_name: str,
        model_class: type[BaseIndexModel],
        query_clause: DenseSearchClause,
        filter_expr: str | None,
        limit: int,
    ):
        """执行稠密向量搜索"""
        if not model_class.Meta.dense_vector_field:
            raise RuntimeError("Dense search requires dense vector field")

        return await self._client.search(
            collection_name=collection_name,
            data=[query_clause.vector],
            anns_field=model_class.Meta.dense_vector_field,
            param={"metric_type": self.extra_config.metric_type, "params": {}},
            limit=limit,
            expr=filter_expr,
            output_fields=["*"],
        )

    async def _execute_sparse_search(
        self,
        collection_name: str,
        model_class: type[BaseIndexModel],
        query_clause: SparseSearchClause,
        filter_expr: str | None,
        limit: int,
    ):
        """执行稀疏向量搜索"""
        if not model_class.Meta.sparse_vector_field:
            raise RuntimeError("Sparse search requires sparse vector field")

        sparse_vector_list = [[query_clause.sparse_vector]]

        return await self._client.search(
            collection_name=collection_name,
            data=sparse_vector_list,
            anns_field=model_class.Meta.sparse_vector_field,
            param={"metric_type": self.extra_config.metric_type, "params": {"drop_ratio_build": 0.0}},
            limit=limit,
            expr=filter_expr,
            output_fields=["*"],
        )

    async def _execute_hybrid_search(
        self,
        collection_name: str,
        model_class: type[BaseIndexModel],
        query_clause: HybridSearchClause,
        filter_expr: str | None,
    ) -> list[tuple[dict[str, Any], float]]:
        """执行混合向量搜索并合并结果"""
        if not model_class.Meta.dense_vector_field or not model_class.Meta.sparse_vector_field:
            raise RuntimeError("Hybrid search requires both dense and sparse vector fields")

        dense_results = await self._execute_dense_search(
            collection_name, model_class, query_clause.dense, filter_expr, query_clause.dense.top_k * 2,
        )

        sparse_results = await self._execute_sparse_search(
            collection_name, model_class, query_clause.sparse, filter_expr, query_clause.sparse.top_k * 2,
        )

        return self._merge_hybrid_results(dense_results[0], sparse_results[0], query_clause)

    def _merge_hybrid_results(
        self,
        dense_results: list[dict[str, Any]],
        sparse_results: list[dict[str, Any]],
        query_clause: HybridSearchClause,
    ) -> list[tuple[dict[str, Any], float]]:
        """合并混合搜索结果"""
        dense_dict = {result["id"]: (result, result["distance"]) for result in dense_results}
        sparse_dict = {result["id"]: (result, result["distance"]) for result in sparse_results}

        all_ids = set(dense_dict.keys()) | set(sparse_dict.keys())

        combined_results = []
        for id_ in all_ids:
            dense_doc, dense_score = dense_dict.get(id_, ({}, 0))
            sparse_doc, sparse_score = sparse_dict.get(id_, ({}, 0))

            final_doc = dense_doc if dense_doc else sparse_doc
            final_score = query_clause.weight_dense * dense_score + query_clause.weight_sparse * sparse_score

            combined_results.append((final_doc, final_score))

        combined_results.sort(key=lambda x: x[1], reverse=True)
        return combined_results

    async def search_cursor(
        self,
        model_class: type[BaseIndexModel],
        query_clause: DenseSearchClause | SparseSearchClause | HybridSearchClause,
        filter_clause: FilterClause | None = None,
        page_size: int = 100,
        cursor: str | None = None,
    ) -> tuple[list[tuple[dict[str, Any], float]], str | None]:
        """搜索（Cursor 方式）"""
        filter_expr = self._convert_filter(model_class, filter_clause) if filter_clause else None

        offset = 0
        if cursor:
            try:
                offset = int(cursor)
            except (ValueError, TypeError):
                offset = 0

        results = await self._execute_search(model_class, query_clause, filter_expr, page_size)

        if isinstance(query_clause, HybridSearchClause):
            final_offset = offset % len(results) if results else 0
            paginated_results = results[final_offset : final_offset + page_size]
            next_cursor = str(final_offset + len(paginated_results)) if paginated_results else None
            return paginated_results, next_cursor

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

    async def exists(self, model_class: type[BaseIndexModel], id: str | int) -> bool:
        """检查文档是否存在"""
        collection_name = self.build_collection_name(model_class)

        result = await self._client.get(collection_name=collection_name, ids=[id])
        return len(result) > 0

    def _convert_filter(self, model_class: type[BaseIndexModel], filter_clause: FilterClause) -> str:
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
                    # 检查字段是否为 ARRAY 类型
                    field_is_array = self._is_array_field(model_class, field)

                    if field_is_array:
                        # ARRAY 类型使用 ARRAY_CONTAINS 或 ARRAY_CONTAINS_ALL
                        if all(isinstance(v, str) for v in values):
                            # 对于字符串 ARRAY，需要转义
                            escaped_values = [f'"{self._escape_string(v)}"' for v in values]
                            # 如果只有一个值，使用 ARRAY_CONTAINS
                            if len(escaped_values) == 1:
                                conditions.append(f"ARRAY_CONTAINS({field}, {escaped_values[0]})")
                            else:
                                # 多个值，使用 ARRAY_CONTAINS_ALL 检查是否包含所有值
                                formatted_values = ", ".join(escaped_values)
                                conditions.append(f"ARRAY_CONTAINS_ALL({field}, [{formatted_values}])")
                        else:
                            # 数值 ARRAY
                            formatted_values = ", ".join(str(v) for v in values)
                            if len(values) == 1:
                                conditions.append(f"ARRAY_CONTAINS({field}, {values[0]})")
                            else:
                                conditions.append(f"ARRAY_CONTAINS_ALL({field}, [{formatted_values}])")
                    else:
                        # 非 ARRAY 类型，使用标准的 in 语法
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
            and_conditions = [self._convert_filter(model_class, f) for f in filter_clause.and_conditions]
            if and_conditions:
                conditions.append(f"({' and '.join(and_conditions)})")

        if filter_clause.or_conditions:
            or_conditions = [self._convert_filter(model_class, f) for f in filter_clause.or_conditions]
            if or_conditions:
                conditions.append(f"({' or '.join(or_conditions)})")

        return " and ".join(conditions) if conditions else ""

    def _is_array_field(self, model_class: type[BaseIndexModel], field_name: str) -> bool:
        """检查字段是否为 ARRAY 类型（排除向量字段）"""
        field_info = model_class.model_fields.get(field_name)
        if not field_info or not field_info.annotation:
            return False

        # 判断是否为 List 类型
        if get_origin(field_info.annotation) != list:
            return False

        # 排除向量字段（向量字段是 List[float] 但不是 ARRAY 标量）
        if field_name == model_class.Meta.dense_vector_field:
            return False
        if field_name == model_class.Meta.sparse_vector_field:
            return False

        return True

    def _escape_string(self, value: str) -> str:
        """转义 Milvus 表达式中的字符串特殊字符"""
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return escaped
