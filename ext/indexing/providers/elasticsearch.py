"""Elasticsearch Provider 实现"""

from typing import Any
from datetime import datetime
import json

from loguru import logger

from enum import StrEnum

from ext.ext_tortoise.models.knowledge_base import IndexingBackendConfig
from ext.indexing.base import BaseProvider, BaseIndexModel
from ext.indexing.providers.types import ElasticsearchConfig
from ext.indexing.types import FilterClause, DenseSearchClause, SparseSearchClause, HybridSearchClause
from elasticsearch import AsyncElasticsearch
from typing import get_origin, get_args


class IndexMetadataEnum(StrEnum):
    """自定义IndexMetaEnum"""

    enable_keyword = "enable_keyword"  # 是否启用keyword


class ElasticsearchProvider(BaseProvider[ElasticsearchConfig]):
    """Elasticsearch Provider 实现"""

    def __init__(self, config: IndexingBackendConfig):
        super().__init__(config, ElasticsearchConfig)

        # Validate configuration
        if self.extra_config.vector_similarity not in ["cosine", "l2", "ip"]:
            raise ValueError(
                f"Invalid vector_similarity: {self.extra_config.vector_similarity}. Must be one of: cosine, l2, ip",
            )

    async def connect(self):
        """建立连接"""

        scheme = "https" if self.config.verify_ssl else "http"
        hosts = (
            f"{scheme}://{self.config.host}:{self.config.port}"
            if self.config.port
            else f"{scheme}://{self.config.host}"
        )
        basic_auth = (self.config.username, self.config.password) if self.config.username else None

        self._client = AsyncElasticsearch(
            hosts=hosts,
            basic_auth=basic_auth,
            verify_certs=self.config.verify_ssl,
            max_retries=self.config.max_retries,
            request_timeout=float(self.config.timeout),
        )
        await self._client.info()

    async def disconnect(self):
        """断开连接"""
        if self._client:
            await self._client.close()

    async def create_collection(self, model_class: type[BaseIndexModel], drop_existing: bool = False):
        """创建索引"""
        # Validate model configuration
        self._validate_model_config(model_class)

        index_name = self.build_collection_name(model_class)

        if drop_existing:
            await self._client.indices.delete(index=index_name, ignore_unavailable=True)

        mapping = self._build_mapping(model_class)
        settings = {
            "number_of_shards": self.extra_config.number_of_shards,
            "number_of_replicas": self.extra_config.number_of_replicas,
            "index": {"similarity": {"custom_bm25": {"type": "BM25", "k1": "1.3", "b": "0.6"}}},
        }

        await self._client.indices.create(index=index_name, mappings=mapping, settings=settings)

    async def drop_collection(self, model_class: type[BaseIndexModel]):
        """删除索引"""
        index_name = self.build_collection_name(model_class)
        await self._client.indices.delete(index=index_name, ignore_unavailable=True)

    def _build_mapping(self, model_class: type[BaseIndexModel]) -> dict:
        """根据模型类生成 Elasticsearch mapping"""

        mapping = {"properties": {}}

        for field_name, field_info in model_class.model_fields.items():
            if field_name.startswith("_") or field_name == "id":
                continue

            index_metadata: dict[str, Any] = {}
            if field_info.json_schema_extra and isinstance(field_info.json_schema_extra, dict):
                index_metadata = field_info.json_schema_extra.get("index_metadata", {})  # type: ignore

            if field_name == model_class.Meta.dense_vector_field:
                mapping["properties"][field_name] = {
                    "type": "dense_vector",
                    "dims": model_class.Meta.dense_vector_dimension,
                    "index": True,
                    "similarity": index_metadata.get("similarity") or self.extra_config.vector_similarity,
                }
            elif field_info.annotation == bool:
                mapping["properties"][field_name] = {"type": "boolean"}
            elif field_info.annotation == int:
                mapping["properties"][field_name] = {"type": "integer"}
            elif field_info.annotation == float:
                mapping["properties"][field_name] = {"type": "float"}
            elif field_info.annotation == str:
                mapping["properties"][field_name] = {
                    "type": "text",
                    "analyzer": index_metadata.get("analyzer") or self.extra_config.text_analyzer,
                    "search_analyzer": index_metadata.get("search_analyzer") or self.extra_config.search_analyzer,
                }
                if index_metadata.get(IndexMetadataEnum.enable_keyword.value, False):
                    mapping["properties"][field_name]["fields"] = {"keyword": {"type": "keyword"}}

            elif field_info.annotation == datetime:
                mapping["properties"][field_name] = {"type": "date"}
            elif field_info.annotation in [dict, dict] or get_origin(field_info.annotation) == dict:
                mapping["properties"][field_name] = {"type": "object"}
            elif get_origin(field_info.annotation) == list:
                # Handle list types: List[str], List[int], etc.
                args = get_args(field_info.annotation)
                if not args or args[0] in [str, int, float]:
                    mapping["properties"][field_name] = {"type": "keyword"}
                else:
                    mapping["properties"][field_name] = {"type": "object"}
            else:
                raise RuntimeError(f"Unknow field type {field_name}-{field_info.annotation}")

        if model_class.Meta.partition_key:
            mapping["_routing"] = {"required": True}

        return mapping

    def _build_term_or_match_query(self, model_class: type[BaseIndexModel], field: str, value: Any) -> dict:
        """根据字段类型构建 term 或 match 查询

        - 如果字段名以 .keyword 结尾，使用 term 查询（精确匹配）
        - 如果字段是 str 类型，使用 match 查询（全文匹配）
        - 其他类型使用 term 查询（精确匹配）
        """
        if field.endswith(".keyword"):
            return {"term": {field: value}}

        field_name = field
        field_type = model_class.model_fields.get(field_name)
        if field_type and field_type.annotation == str:
            return {"match": {field: value}}

        return {"term": {field: value}}

    def _get_default_text_field(self, model_class: type[BaseIndexModel]) -> str:
        """获取默认的文本字段名

        当 SparseSearchClause.field_name 为 None 时，返回第一个 str 类型的字段。
        """
        for field_name, field_info in model_class.model_fields.items():
            if field_name.startswith("_") or field_name == "id":
                continue
            if field_info.annotation == str:
                return field_name
        raise RuntimeError("No text fields available for sparse search")

    def _get_routing_value(self, model_class: type[BaseIndexModel], document: dict[str, Any]) -> str | None:
        """从 document 中提取 partition_key 字段的值作为 routing"""
        if model_class.Meta.partition_key:
            value = document.get(model_class.Meta.partition_key)
            if not value:
                raise RuntimeError("partition key value can not be empty")
            return value
        return None

    def _get_routing_value_from_filter(self, filter_clause: FilterClause | None, partition_key: str) -> str | None:
        """从 filter_clause 提取 routing 值（支持多个，逗号分隔）"""
        if not filter_clause:
            return None

        if filter_clause.in_list and partition_key in filter_clause.in_list:
            values = filter_clause.in_list[partition_key]
            if values:
                return ",".join(str(v) for v in values)

        if filter_clause.equals and partition_key in filter_clause.equals:
            value = filter_clause.equals[partition_key]
            if value:
                return str(value)

        return None

    async def get(self, model_class: type[BaseIndexModel], ids: list) -> list[dict]:
        """获取文档"""
        index_name = self.build_collection_name(model_class)

        # 如果有 partition_key，使用 ids 查询而不是 mget
        if model_class.Meta.partition_key:
            search_body = {"size": len(ids), "query": {"ids": {"values": ids}}}
            response = await self._client.search(index=index_name, body=search_body)
            result: list[dict] = []
            for hit in response["hits"]["hits"]:
                doc_data = hit["_source"]
                doc_data["id"] = hit["_id"]
                result.append(doc_data)
            return result

        # 无 partition_key，使用 mget
        docs = [{"_id": doc_id} for doc_id in ids]
        response = await self._client.mget(index=index_name, docs=docs)
        result: list[dict] = []
        for doc in response["docs"]:
            if doc.get("found"):
                doc_data = doc["_source"]
                doc_data["id"] = doc["_id"]
                result.append(doc_data)

        return result

    async def filter(
        self,
        model_class: type[BaseIndexModel],
        filter_clause: FilterClause | None,
        limit: int = 10,
        offset: int = 0,
        sort: str | None = None,
    ) -> list[dict[str, Any]]:
        """过滤查询"""
        index_name = self.build_collection_name(model_class)

        search_body = {"size": limit, "from": offset, "query": {"match_all": {}}}

        if filter_clause:
            search_body["query"] = self._convert_filter(model_class, filter_clause)  # type: ignore

        # 添加 routing（如果有 partition_key）
        if model_class.Meta.partition_key:
            # 从 filter_clause 中提取 partition_key 的值
            partition_value = (
                filter_clause.equals.get(model_class.Meta.partition_key)
                if filter_clause and filter_clause.equals
                else None
            )
            if partition_value:
                search_body["routing"] = partition_value

        if sort:
            sort_field, sort_order = sort.split(":")
            search_body["sort"] = [{sort_field: {"order": sort_order}}]

        response = await self._client.search(index=index_name, body=search_body)

        results = []
        for hit in response["hits"]["hits"]:
            doc = hit["_source"]
            doc["id"] = hit["_id"]
            results.append(doc)
        return results

    async def insert(
        self,
        model_class: type[BaseIndexModel],
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        """插入文档（批量）"""
        if not documents:
            return None

        index_name = self.build_collection_name(model_class)

        bulk_data = []
        for doc in documents:
            op = {"index": {"_index": index_name}}

            doc_id = doc.get("id")
            if doc_id and not model_class.Meta.auto_generate_id:
                op["index"]["_id"] = doc_id

            # 添加 routing（如果有 partition_key）
            if model_class.Meta.partition_key:
                routing_value = self._get_routing_value(model_class, doc)
                if routing_value:
                    op["index"]["routing"] = routing_value  # type: ignore

            # 如果 auto_generate_id=True，从文档中移除空的 id 字段
            doc_to_insert = doc.copy()
            if model_class.Meta.auto_generate_id:
                doc_to_insert.pop("id", None)

            bulk_data.append(op)
            bulk_data.append(doc_to_insert)

        response = await self._client.bulk(operations=bulk_data, refresh=self.extra_config.auto_flush)

        self._log_bulk_errors(response, "insert")
        return self._extract_bulk_results(response, documents, "index")

    def _log_bulk_errors(self, response: Any, operation: str = "bulk"):
        """记录批量操作错误"""
        response_dict = response.body if hasattr(response, "body") else response
        if response_dict.get("errors", False):
            items = response_dict.get("items", [])
            for item in items:
                result_key = next(iter(item.keys()))
                result = item.get(result_key, {})
                if "error" in result:
                    logger.warning(f"{operation} encounter error: {result['error']}")

    def _extract_bulk_results(
        self,
        response: Any,
        documents: list[dict[str, Any]],
        operation: str = "index",
    ) -> list[dict[str, Any]] | None:
        """从批量响应中提取结果文档"""
        response_dict = response.body if hasattr(response, "body") else response
        results = []
        items = response_dict.get("items", [])
        for i, item in enumerate(items):
            result = item.get(operation, {})
            if result.get("result") in ["created", "updated"]:
                result_id = result.get("_id")
                if result_id:
                    result_doc = documents[i].copy()
                    result_doc["id"] = result_id
                    results.append(result_doc)

        return results if results else None

    async def update(self, model_class: type[BaseIndexModel], documents: list[dict[str, Any]]) -> list:
        """更新文档"""
        if not documents:
            return []
        index_name = self.build_collection_name(model_class)
        bulk_data = []
        for doc in documents:
            op = {"update": {"_index": index_name}}
            doc_id = doc.get("id")
            assert doc_id
            op["update"]["_id"] = doc_id
            if model_class.Meta.partition_key:
                routing_value = self._get_routing_value(model_class, doc)
                if routing_value:
                    op["update"]["routing"] = routing_value

            # 移除 doc 中的空 id 字段，避免覆盖 ES 的 _id
            doc_to_update = {k: v for k, v in doc.items() if k != "id" or v}
            bulk_data.append(op)
            bulk_data.append({"doc": doc_to_update})
        response = await self._client.bulk(operations=bulk_data, refresh=self.extra_config.auto_flush)
        return []

        self._log_bulk_errors(response, "update")

    async def bulk_upsert(
        self,
        model_class: type[BaseIndexModel],
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        """批量插入或更新（upsert）"""
        if not documents:
            return None

        index_name = self.build_collection_name(model_class)

        bulk_data = []
        for doc in documents:
            op = {"index": {"_index": index_name}}

            doc_id = doc.get("id")
            if doc_id:
                op["index"]["_id"] = doc_id

            # 添加 routing（如果有 partition_key）
            if model_class.Meta.partition_key:
                routing_value = self._get_routing_value(model_class, doc)
                if routing_value:
                    op["index"]["routing"] = routing_value

            bulk_data.append(op)
            bulk_data.append(doc)

        response = await self._client.bulk(operations=bulk_data, refresh=self.extra_config.auto_flush)

        return self._extract_bulk_results(response, documents, "index")

    async def delete(self, model_class: type[BaseIndexModel], ids: list[str]):  # type: ignore
        """删除文档（根据 ID，支持 routing）"""
        if not ids:
            return

        index_name = self.build_collection_name(model_class)

        if not model_class.Meta.partition_key:
            operations = [{"delete": {"_index": index_name, "_id": id_}} for id_ in ids]
            await self._client.bulk(operations=operations)
            return

        # 有 partition_key，先查询获取文档及其 routing 值
        search_body = {"size": len(ids), "query": {"ids": {"values": ids}}}
        response = await self._client.search(index=index_name, body=search_body)
        hits = response.get("hits", {}).get("hits", [])

        # 按 routing 分组删除
        for hit in hits:
            doc_id = hit["_id"]
            routing_value = hit.get("_routing")

            if routing_value:
                # 删除指定 routing 的文档
                await self._client.delete(index=index_name, id=doc_id, routing=routing_value, refresh=self.extra_config.auto_flush)
            else:
                # 没有 routing 值，尝试直接删除（可能会失败）
                try:
                    await self._client.delete(index=index_name, id=doc_id, refresh=self.extra_config.auto_flush)
                except Exception as e:
                    logger.warning(f"Failed to delete document {doc_id} without routing: {e}")

    async def delete_by_query(self, model_class: type[BaseIndexModel], filter_clause: FilterClause):
        """根据条件删除（支持 routing）"""
        index_name = self.build_collection_name(model_class)
        query = self._convert_filter(model_class, filter_clause)

        if not model_class.Meta.partition_key:
            await self._client.delete_by_query(index=index_name, query=query)
            return

        # 如果有 partition_key，必须从 filter_clause 中获取 routing 值
        routing_value = self._get_routing_value_from_filter(filter_clause, model_class.Meta.partition_key)
        if not routing_value:
            raise ValueError(
                f"Cannot delete documents with partition key '{model_class.Meta.partition_key}'. "
                f"Please include the partition key value in the filter_clause.",
            )

        await self._client.delete_by_query(index=index_name, query=query, routing=routing_value, refresh=self.extra_config.auto_flush)

    async def count(self, model_class: type[BaseIndexModel], filter_clause: FilterClause | None) -> int:
        """统计文档数量"""
        index_name = self.build_collection_name(model_class)

        query = {"match_all": {}}
        if filter_clause:
            query = self._convert_filter(model_class, filter_clause)

        result = await self._client.count(index=index_name, query=query)
        return result["count"]

    async def search(
        self,
        model_class: type[BaseIndexModel],
        query_clause: DenseSearchClause | SparseSearchClause | HybridSearchClause,
        filter_clause: FilterClause | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[tuple[dict[str, Any], float]]:
        """搜索"""
        index_name = self.build_collection_name(model_class)

        search_body = self._build_search_body(model_class, query_clause, filter_clause)

        if query_clause.output_fields and "*" not in query_clause.output_fields:
            search_body["_source"] = query_clause.output_fields

        if model_class.Meta.partition_key:
            routing_value = self._get_routing_value_from_filter(filter_clause, model_class.Meta.partition_key)
            if routing_value:
                search_body["routing"] = routing_value

        search_body["size"] = limit
        search_body["from"] = offset

        logger.info(f"search body: {search_body}")

        response = await self._client.search(index=index_name, body=search_body)

        results = []
        for hit in response["hits"]["hits"]:
            doc = hit.get("_source", {})
            doc["id"] = hit["_id"]
            score = hit["_score"]
            results.append((doc, score))

        return results

    async def search_cursor(
        self,
        model_class: type[BaseIndexModel],
        query_clause: DenseSearchClause | SparseSearchClause | HybridSearchClause,
        filter_clause: FilterClause | None = None,
        page_size: int = 100,
        cursor: str | None = None,
    ) -> tuple[list[tuple[dict[str, Any], float]], str | None]:
        """搜索（Cursor 方式）"""
        index_name = self.build_collection_name(model_class)

        search_body = self._build_search_body(model_class, query_clause, filter_clause)

        if query_clause.output_fields and "*" not in query_clause.output_fields:
            search_body["_source"] = query_clause.output_fields

        if model_class.Meta.partition_key:
            routing_value = self._get_routing_value_from_filter(filter_clause, model_class.Meta.partition_key)
            if routing_value:
                search_body["routing"] = routing_value

        if "sort" not in search_body:
            search_body["sort"] = ["_score"]

        search_body["size"] = page_size
        search_body["from"] = 0

        # Parse cursor from JSON string to list
        if cursor:
            try:
                search_body["search_after"] = json.loads(cursor)
            except (json.JSONDecodeError, TypeError):
                # Fallback to single value for backward compatibility
                search_body["search_after"] = [cursor]

        response = await self._client.search(index=index_name, body=search_body)

        results = []
        for hit in response["hits"]["hits"]:
            doc = hit.get("_source", {})
            doc["id"] = hit["_id"]
            score = hit["_score"]
            results.append((doc, score))

        next_cursor = None
        if response["hits"]["hits"]:
            last_hit = response["hits"]["hits"][-1]
            if "sort" in last_hit and last_hit["sort"]:
                # Encode sort array as JSON string for cursor
                next_cursor = json.dumps(last_hit["sort"])
            else:
                # Fallback to using _id
                next_cursor = json.dumps([last_hit["_id"]])

        return results, next_cursor

    def _build_search_body(
        self,
        model_class: type[BaseIndexModel],
        query_clause: DenseSearchClause | SparseSearchClause | HybridSearchClause,
        filter_clause: FilterClause | None,
    ) -> dict:
        """根据查询类型构建 ES Query DSL"""

        if isinstance(query_clause, DenseSearchClause):
            search_body = self._build_dense_search_body(model_class, query_clause)
        elif isinstance(query_clause, SparseSearchClause):
            search_body = self._build_sparse_search_body(model_class, query_clause)
        elif isinstance(query_clause, HybridSearchClause):
            search_body = self._build_hybrid_search_body(model_class, query_clause)
        else:
            raise RuntimeError(f"Not supported query clause: {query_clause}")

        if not filter_clause:
            return search_body

        return self._apply_filter_to_search_body(model_class, search_body, query_clause, filter_clause)

    def _build_dense_search_body(self, model_class: type[BaseIndexModel], query_clause: DenseSearchClause) -> dict:
        """构建稠密向量搜索 query body"""
        if not model_class.Meta.dense_vector_field:
            raise RuntimeError("Not supported dense query without dense vector field")

        return {
            "query": {
                "knn": {
                    "field": model_class.Meta.dense_vector_field,
                    "query_vector": query_clause.vector,
                    "num_candidates": query_clause.top_k * 50,
                },
            },
        }

    def _build_sparse_search_body(self, model_class: type[BaseIndexModel], query_clause: SparseSearchClause) -> dict:
        """构建稀疏全文搜索 query body"""
        field_name = getattr(query_clause, "field_name", None)

        if not field_name:
            field_name = self._get_default_text_field(model_class)

        return {
            "query": {
                "match": {
                    field_name: {
                        "query": query_clause.query_text,
                        "min_score": query_clause.min_score,
                    },
                },
            },
        }

    def _build_hybrid_search_body(self, model_class: type[BaseIndexModel], query_clause: HybridSearchClause) -> dict:
        """构建混合搜索 query body（带权重）"""
        if not model_class.Meta.dense_vector_field:
            raise RuntimeError("Not supported hybrid query without dense vector field")

        field_name = getattr(query_clause.sparse, "field_name", None)
        if not field_name:
            field_name = self._get_default_text_field(model_class)

        return {
            "query": {
                "bool": {
                    "should": [
                        {
                            "knn": {
                                "field": model_class.Meta.dense_vector_field,
                                "query_vector": query_clause.dense.vector,
                                "num_candidates": query_clause.dense.top_k * 50,
                                "boost": query_clause.weight_dense,
                            },
                        },
                        {
                            "bool": {
                                "must": [
                                    {
                                        "match": {
                                            field_name: {
                                                "query": query_clause.sparse.query_text,
                                                "boost": query_clause.weight_sparse,
                                            },
                                        },
                                    },
                                ],
                                "min_score": query_clause.sparse.min_score,
                            },
                        },
                    ],
                },
            },
        }

    def _apply_filter_to_search_body(
        self,
        model_class: type[BaseIndexModel],
        search_body: dict,
        query_clause: DenseSearchClause | SparseSearchClause | HybridSearchClause,
        filter_clause: FilterClause,
    ) -> dict:
        """应用过滤条件到搜索 body"""
        if isinstance(query_clause, DenseSearchClause) or isinstance(query_clause, HybridSearchClause):
            converted_filter = self._convert_filter(model_class, filter_clause)
            if "knn" in search_body["query"]:
                search_body["query"]["knn"]["filter"] = converted_filter
            elif "bool" in search_body["query"] and "should" in search_body["query"]["bool"]:
                for clause in search_body["query"]["bool"]["should"]:
                    if "knn" in clause:
                        clause["knn"]["filter"] = converted_filter
        else:
            search_body["query"]["bool"] = {
                "must": [search_body["query"], self._convert_filter(model_class, filter_clause)],
            }

        return search_body

    def _convert_filter(self, model_class: type[BaseIndexModel], filter_clause: FilterClause) -> dict:
        """转换 FilterClause 为 ES Query DSL"""
        must = []

        if filter_clause.equals:
            for field, value in filter_clause.equals.items():
                must.append(self._build_term_or_match_query(model_class, field, value))

        if filter_clause.in_list:
            for field, values in filter_clause.in_list.items():
                must.append({"terms": {field: values}})

        if filter_clause.range:
            for field, range_value in filter_clause.range.items():
                must.append({"range": {field: range_value}})

        if filter_clause.and_conditions:
            and_filters = [self._convert_filter(model_class, f) for f in filter_clause.and_conditions]
            must.append({"bool": {"must": and_filters}})

        if filter_clause.or_conditions:
            or_filters = [self._convert_filter(model_class, f) for f in filter_clause.or_conditions]
            must.append({"bool": {"should": or_filters, "minimum_should_match": 1}})

        return {"bool": {"must": must}} if must else {"match_all": {}}

    async def exists(self, model_class: type[BaseIndexModel], id: str) -> bool:
        """检查文档是否存在"""
        index_name = self.build_collection_name(model_class)

        # 如果有 partition_key，使用 count 而不是 exists
        if model_class.Meta.partition_key:
            search_body = {"query": {"ids": {"values": [id]}}}
            response = await self._client.count(index=index_name, body=search_body)
            return response.get("count", 0) > 0

        # 无 partition_key，使用 exists
        return (await self._client.exists(index=index_name, id=id)).body

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            return await self._client.ping()
        except:
            return False
