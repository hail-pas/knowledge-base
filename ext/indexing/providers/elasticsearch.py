"""Elasticsearch Provider 实现"""

from typing import Type, Dict, Any, List, Optional
from datetime import datetime
import json

from loguru import logger

from ext.ext_tortoise.models.knowledge_base import IndexingBackendConfig
from ext.indexing.base import BaseProvider, BaseIndexModel
from ext.indexing.providers.types import ElasticsearchConfig
from ext.indexing.types import FilterClause, DenseSearchClause, SparseSearchClause, HybridSearchClause
from elasticsearch import AsyncElasticsearch
from typing import get_origin, get_args


class ElasticsearchProvider(BaseProvider[ElasticsearchConfig]):
    """Elasticsearch Provider 实现"""

    def __init__(self, config: IndexingBackendConfig):
        super().__init__(config, ElasticsearchConfig)

        # Validate configuration
        if self.extra_config.vector_similarity not in ["cosine", "l2", "ip"]:
            raise ValueError(
                f"Invalid vector_similarity: {self.extra_config.vector_similarity}. Must be one of: cosine, l2, ip"
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

    async def create_collection(self, model_class: Type[BaseIndexModel], drop_existing: bool = False):
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

    async def drop_collection(self, model_class: Type[BaseIndexModel]):
        """删除索引"""
        index_name = self.build_collection_name(model_class)
        await self._client.indices.delete(index=index_name, ignore_unavailable=True)

    def _build_mapping(self, model_class: Type[BaseIndexModel]) -> dict:
        """根据模型类生成 Elasticsearch mapping"""

        mapping = {"properties": {}}

        for field_name, field_info in model_class.model_fields.items():
            if field_name.startswith("_") or field_name == "id":
                continue

            if field_name == model_class.Meta.dense_vector_field:
                mapping["properties"][field_name] = {
                    "type": "dense_vector",
                    "dims": model_class.Meta.dense_vector_dimension,
                    "index": True,
                    "similarity": self.extra_config.vector_similarity,
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
                    "analyzer": self.extra_config.text_analyzer,
                    "search_analyzer": self.extra_config.search_analyzer,
                    "fields": {"keyword": {"type": "keyword"}},
                }
            elif field_info.annotation == datetime:
                mapping["properties"][field_name] = {"type": "date"}
            elif field_info.annotation in [dict, Dict] or get_origin(field_info.annotation) == dict:
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
            mapping["properties"]["_routing"] = {"type": "keyword"}

        return mapping

    def _build_term_or_match_query(self, model_class: Type[BaseIndexModel], field: str, value: Any) -> dict:
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

    def _get_routing_value(self, model_class: Type[BaseIndexModel], document: Dict[str, Any]) -> str | None:
        """从 document 中提取 partition_key 字段的值作为 routing"""
        if model_class.Meta.partition_key:
            value = document.get(model_class.Meta.partition_key)
            if not value:
                raise RuntimeError(f"partition key value can not be empty")
            return value
        return None

    def _get_routing_value_from_filter(self, filter_clause: Optional[FilterClause], partition_key: str) -> str | None:
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

    async def get(self, model_class: Type[BaseIndexModel], ids: list) -> list[dict]:
        """获取文档"""
        index_name = self.build_collection_name(model_class)

        docs = [{"_id": doc_id} for doc_id in ids]

        response = await self._client.mget(index=index_name, docs=docs)
        result: List[Dict] = []
        for doc in response["docs"]:
            if doc.get("found"):
                doc_data = doc["_source"]
                doc_data["id"] = doc["_id"]
                result.append(doc_data)

        return result

    async def filter(
        self,
        model_class: Type[BaseIndexModel],
        filter_clause: Optional[FilterClause],
        limit: int = 10,
        offset: int = 0,
        sort: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
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
        self, model_class: Type[BaseIndexModel], documents: List[Dict[str, Any]]
    ) -> Optional[List[Dict[str, Any]]]:
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

            bulk_data.append(op)
            bulk_data.append(doc)

        response = await self._client.bulk(operations=bulk_data, refresh=True)

        # Check for bulk errors
        if response.get("errors", False):
            items = response.get("items", [])
            for i, item in enumerate(items):
                index_result = item.get("index", {})
                if "error" in index_result:
                    # Use proper logging instead of print
                    logger.warning(f"insert encouter error: {index_result['error']}")

        # Extract generated IDs from response
        results = []
        items = response.get("items", [])
        for i, item in enumerate(items):
            index_result = item.get("index", {})
            if index_result.get("result") in ["created", "updated"]:
                result_id = index_result.get("_id")
                if result_id:
                    result_doc = documents[i].copy()
                    result_doc["id"] = result_id
                    results.append(result_doc)

        return results if results else None

    async def update(self, model_class: Type[BaseIndexModel], documents: List[Dict[str, Any]]):
        """更新文档"""
        if not documents:
            return
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
            bulk_data.append(op)
            bulk_data.append({"doc": doc})
        await self._client.bulk(operations=bulk_data)

    async def bulk_upsert(
        self, model_class: Type[BaseIndexModel], documents: List[Dict[str, Any]]
    ) -> Optional[List[Dict[str, Any]]]:
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
                    op["index"]["routing"] = routing_value  # type: ignore

            bulk_data.append(op)
            bulk_data.append(doc)

        response = await self._client.bulk(operations=bulk_data, refresh=True)

        # Extract generated IDs from response
        results = []
        items = response.get("items", [])
        for i, item in enumerate(items):
            index_result = item.get("index", {})
            if index_result.get("result") in ["created", "updated"]:
                result_id = index_result.get("_id")
                if result_id:
                    result_doc = documents[i].copy()
                    result_doc["id"] = result_id
                    results.append(result_doc)

        return results if results else None

    async def delete(self, model_class: Type[BaseIndexModel], ids: List[str]):  # type: ignore
        """删除文档（根据 ID，支持 routing）"""
        if not ids:
            return

        index_name = self.build_collection_name(model_class)

        if not model_class.Meta.partition_key:
            operations = [{"delete": {"_index": index_name, "_id": id_}} for id_ in ids]
            await self._client.bulk(operations=operations)
            return

        await self._client.delete_by_query(index=index_name, query={"terms": {"id": ids}})

    async def delete_by_query(self, model_class: Type[BaseIndexModel], filter_clause: FilterClause):
        """根据条件删除（支持 routing）"""
        index_name = self.build_collection_name(model_class)
        query = self._convert_filter(model_class, filter_clause)

        routing_value = None
        if model_class.Meta.partition_key:
            routing_value = self._get_routing_value_from_filter(filter_clause, model_class.Meta.partition_key)

        await self._client.delete_by_query(index=index_name, query=query, routing=routing_value)

    async def count(self, model_class: Type[BaseIndexModel], filter_clause: Optional[FilterClause]) -> int:
        """统计文档数量"""
        index_name = self.build_collection_name(model_class)

        query = {"match_all": {}}
        if filter_clause:
            query = self._convert_filter(model_class, filter_clause)

        result = await self._client.count(index=index_name, query=query)
        return result["count"]

    async def search(
        self,
        model_class: Type[BaseIndexModel],
        query_clause: DenseSearchClause | SparseSearchClause | HybridSearchClause,
        filter_clause: Optional[FilterClause] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> List[tuple[Dict[str, Any], float]]:
        """搜索"""
        index_name = self.build_collection_name(model_class)

        search_body = self._build_search_body(model_class, query_clause, filter_clause)

        if model_class.Meta.partition_key:
            routing_value = self._get_routing_value_from_filter(filter_clause, model_class.Meta.partition_key)
            if routing_value:
                search_body["routing"] = routing_value

        search_body["size"] = limit
        search_body["from"] = offset

        logger.info(f"searh body: {search_body}")

        response = await self._client.search(index=index_name, body=search_body)

        results = []
        for hit in response["hits"]["hits"]:
            doc = hit["_source"]
            doc["id"] = hit["_id"]
            score = hit["_score"]
            results.append((doc, score))

        return results

    async def search_cursor(
        self,
        model_class: Type[BaseIndexModel],
        query_clause: DenseSearchClause | SparseSearchClause | HybridSearchClause,
        filter_clause: Optional[FilterClause] = None,
        page_size: int = 100,
        cursor: Optional[str] = None,
    ) -> tuple[List[tuple[Dict[str, Any], float]], Optional[str]]:
        """搜索（Cursor 方式）"""
        index_name = self.build_collection_name(model_class)

        search_body = self._build_search_body(model_class, query_clause, filter_clause)

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
            doc = hit["_source"]
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
        model_class: Type[BaseIndexModel],
        query_clause: DenseSearchClause | SparseSearchClause | HybridSearchClause,
        filter_clause: FilterClause | None,
    ) -> dict:
        """根据查询类型构建 ES Query DSL"""

        if isinstance(query_clause, DenseSearchClause):
            if not model_class.Meta.dense_vector_field:
                raise RuntimeError("Not supported dense query without dense vector field")

            search_body = {
                "query": {
                    "knn": {
                        "field": model_class.Meta.dense_vector_field,
                        "query_vector": query_clause.vector,
                        "num_candidates": query_clause.top_k * 50,
                    }
                }
            }

        elif isinstance(query_clause, SparseSearchClause):
            search_body = {
                "query": {
                    "multi_match": {"query": query_clause.query_text, "fields": query_clause.output_fields},
                    "min_score": query_clause.min_score,
                }
            }

        elif isinstance(query_clause, HybridSearchClause):
            search_body = {
                "query": {
                    "bool": {
                        "should": [
                            {
                                "knn": {
                                    "field": model_class.Meta.dense_vector_field,
                                    "query_vector": query_clause.dense.vector,
                                    "num_candidates": query_clause.dense.top_k * 50,
                                }
                            },
                            {
                                "multi_match": {
                                    "query": query_clause.sparse.query_text,
                                    "fields": query_clause.output_fields,
                                }
                            },
                        ]
                    }
                }
            }

        else:
            raise RuntimeError(f"Not supported query clause: {query_clause}")

        if not filter_clause:
            return search_body

        if isinstance(query_clause, DenseSearchClause) or isinstance(query_clause, HybridSearchClause):
            converted_filter = self._convert_filter(model_class, filter_clause)
            if "knn" in search_body["query"]:
                search_body["query"]["knn"]["filter"] = converted_filter  # type: ignore
            elif "bool" in search_body["query"] and "should" in search_body["query"]["bool"]:
                for clause in search_body["query"]["bool"]["should"]:
                    if "knn" in clause:
                        clause["knn"]["filter"] = converted_filter
        else:
            search_body["query"]["bool"] = {
                "must": [search_body["query"], self._convert_filter(model_class, filter_clause)]
            }

        return search_body

    def _convert_filter(self, model_class: Type[BaseIndexModel], filter_clause: FilterClause) -> dict:
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

    async def exists(self, model_class: Type[BaseIndexModel], id: str) -> bool:
        """检查文档是否存在"""
        index_name = self.build_collection_name(model_class)
        return (await self._client.exists(index=index_name, id=id)).body

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            return await self._client.ping()
        except:
            return False
