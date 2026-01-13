"""
Elasticsearch Provider 实现

提供 Elasticsearch 后端的索引操作和 CRUD 功能。
支持稀疏索引、稠密索引和混合索引。
"""

from typing import Any, Dict, List, Optional, Type, Union
from datetime import datetime
import logging

from enhance.epydantic import create_sub_fields_model

from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import (
    NotFoundError,
    RequestError,
    ConnectionError as ESConnectionError,
    TransportError,
)

from ext.indexing.base import (
    BaseProvider,
    BaseIndexModel,
    SearchQuery,
    SearchResult,
    QueryCondition,
    BoolQuery,
    FieldDefinition,
    FieldType,
    MatchType,
    RangeOperator,
)
from ext.indexing.exceptions import (
    IndexingBackendError,
    IndexingIndexError,
    IndexingQueryError,
    IndexingDocumentError,
    IndexingConfigError,
)
from ext.ext_tortoise.enums import IndexingTypeEnum

logger = logging.getLogger(__name__)


class ElasticsearchProvider(BaseProvider):
    """Elasticsearch Provider 实现

    使用 Elasticsearch 作为索引后端，支持：
    - 稀疏索引（全文检索）
    - 稠密索引（向量检索，需安装 elasticsearch-dense-vector 插件）
    - 混合索引（组合检索）
    """

    # Elasticsearch 字段类型映射
    FIELD_TYPE_MAPPING = {
        FieldType.keyword: "keyword",
        FieldType.text: "text",
        FieldType.integer: "integer",
        FieldType.long: "long",
        FieldType.float: "float",
        FieldType.double: "double",
        FieldType.boolean: "boolean",
        FieldType.date: "date",
        FieldType.datetime: "date",
        FieldType.dense_vector: "dense_vector",
        FieldType.sparse_vector: "sparse_vector",
    }

    # 默认配置
    DEFAULT_TIMEOUT = 30
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_BATCH_SIZE = 500

    def __init__(self, backend_type, config):
        """
        初始化 Elasticsearch Provider

        Args:
            backend_type: 后端类型（固定为 IndexingBackendTypeEnum.elasticsearch）
            config: 配置字典，包含以下字段：
                - host: 主机地址（必需）
                - port: 端口（可选，默认 9200）
                - username: 用户名（可选）
                - password: 密码（可选）
                - api_key: API 密钥（可选）
                - secure: 是否使用 HTTPS（默认 False）
                - verify_certs: 是否验证 SSL 证书（默认与 secure 相同）
                - timeout: 请求超时时间（可选）
                - max_retries: 最大重试次数（可选）
                - 其他兼容性参数（可选，通过 config JSON 字段传递）：
                    - ssl_show_warn: 是否显示 SSL 警告（默认 False）
                    - headers: 自定义请求头（默认 {}）
                    - request_compress: 是否启用请求压缩（默认 False）
                    以及其他 elasticsearch-py 支持的客户端参数
        """
        super().__init__(backend_type, config)

        # 提取配置
        self.host = config["host"]
        self.port = config.get("port", 9200)
        self.secure = config.get("secure", False)
        self.timeout = config.get("timeout", self.DEFAULT_TIMEOUT)
        self.max_retries = config.get("max_retries", self.DEFAULT_MAX_RETRIES)
        self.batch_size = config.get("batch_size", self.DEFAULT_BATCH_SIZE)

        # 构建连接 URL
        scheme = "https" if self.secure else "http"
        self.url = f"{scheme}://{self.host}:{self.port}"

        # 客户端实例
        self._client: Optional[AsyncElasticsearch] = None

    async def connect(self) -> None:
        """建立 Elasticsearch 连接"""
        if self._client is not None:
            return

        try:
            # 构建认证信息
            http_auth = None
            if "username" in self.config and "password" in self.config:
                http_auth = (self.config["username"], self.config["password"])

            # 构建 API Key 认证
            api_key = self.config.get("api_key")

            # 构建客户端参数
            client_params = {
                "hosts": [self.url],
                "basic_auth": http_auth,
                "api_key": api_key,
                "request_timeout": self.timeout,
                "max_retries": self.max_retries,
            }

            # 证书验证配置
            verify_certs = self.config.get("verify_certs", self.secure)
            client_params["verify_certs"] = verify_certs

            # 阿里云 ES 兼容性参数 - 禁用 SSL 警告（默认 False）
            client_params["ssl_show_warn"] = self.config.get("ssl_show_warn", False)

            # 允许从 config JSON 字段传递额外的 Elasticsearch 客户端参数
            # 排除内部配置参数，只传递 ES 客户端支持的参数
            internal_params = {
                "name", "batch_size", "host", "port", "secure", "username",
                "password", "api_key", "timeout", "max_retries", "verify_certs"
            }
            for key, value in self.config.items():
                if key not in internal_params:
                    # 透传额外的兼容性参数给 ES 客户端
                    # 用户可以通过 config JSON 字段传递任何 elasticsearch-py 支持的参数
                    client_params[key] = value

            # 创建客户端
            self._client = AsyncElasticsearch(**client_params)

            # 测试连接
            await self._client.ping()

            logger.info(f"Elasticsearch provider connected: {self.url}")

        except ESConnectionError as e:
            raise IndexingBackendError(f"Failed to connect to Elasticsearch: {str(e)}") from e
        except Exception as e:
            raise IndexingBackendError(f"Unexpected error connecting to Elasticsearch: {str(e)}") from e

    async def disconnect(self) -> None:
        """断开 Elasticsearch 连接"""
        if self._client is not None:
            await self._client.close()
            self._client = None
            logger.info("Elasticsearch provider disconnected")

    async def ping(self) -> bool:
        """检查连接是否正常"""
        if self._client is None:
            return False

        try:
            return await self._client.ping()
        except Exception as e:
            logger.warning(f"Elasticsearch ping failed: {str(e)}")
            return False

    # =========================================================================
    # 索引操作
    # =========================================================================

    async def create_index(self, model_class: Type[BaseIndexModel]) -> bool:
        """创建索引"""
        index_name = model_class.get_index_name()

        # 检查索引是否已存在
        if await self.index_exists(model_class):
            logger.warning(f"Index '{index_name}' already exists")
            return False

        # 生成 mapping
        mapping = self._generate_mapping(model_class)

        # 生成 index settings
        settings = self._generate_settings(model_class)

        try:
            await self._client.indices.create(
                index=index_name,
                mappings=mapping,
                settings=settings,
            )
            logger.info(f"Created index '{index_name}'")
            return True

        except RequestError as e:
            if e.meta.status == 400:
                raise IndexingIndexError(f"Invalid index configuration for '{index_name}': {str(e)}") from e
            raise IndexingIndexError(f"Failed to create index '{index_name}': {str(e)}") from e
        except TransportError as e:
            raise IndexingIndexError(f"Failed to create index '{index_name}': {str(e)}") from e

    async def drop_index(self, model_class: Type[BaseIndexModel]) -> bool:
        """删除索引"""
        index_name = model_class.get_index_name()

        try:
            # 使用 ignore_unavailable 忽略不存在的索引
            await self._client.indices.delete(
                index=index_name,
                ignore_unavailable=True
            )
            logger.info(f"Dropped index '{index_name}'")
            return True

        except TransportError as e:
            raise IndexingIndexError(f"Failed to drop index '{index_name}': {str(e)}") from e

    async def index_exists(self, model_class: Type[BaseIndexModel]) -> bool:
        """检查索引是否存在"""
        index_name = model_class.get_index_name()

        try:
            result = await self._client.indices.exists(index=index_name)
            # HeadApiResponse 对象需要转换为布尔值
            return bool(result)
        except TransportError as e:
            logger.warning(f"Failed to check index existence for '{index_name}': {str(e)}")
            return False

    # =========================================================================
    # 文档操作
    # =========================================================================

    async def insert(self, document: BaseIndexModel) -> bool:
        """插入单条文档"""
        index_name = document.get_index_name()
        doc_id = self._get_document_id(document)

        try:
            await self._client.index(
                index=index_name,
                id=doc_id,
                document=self._model_to_dict(document),
                refresh=True,
            )
            return True

        except Exception as e:
            raise IndexingDocumentError(f"Failed to insert document into '{index_name}': {str(e)}") from e

    async def update(self, document: BaseIndexModel, doc_id: Optional[str] = None) -> bool:
        """更新文档"""
        index_name = document.get_index_name()
        doc_id = doc_id or self._get_document_id(document)

        if doc_id is None:
            raise IndexingDocumentError("Document ID is required for update operation")

        try:
            await self._client.update(
                index=index_name,
                id=doc_id,
                doc=self._model_to_dict(document),
            )
            return True

        except NotFoundError:
            raise IndexingDocumentError(f"Document with ID '{doc_id}' not found in index '{index_name}'") from None
        except Exception as e:
            raise IndexingDocumentError(f"Failed to update document in '{index_name}': {str(e)}") from e

    async def upsert(self, document: BaseIndexModel, doc_id: Optional[str] = None) -> bool:
        """插入或更新文档"""
        index_name = document.get_index_name()
        doc_id = doc_id or self._get_document_id(document)

        if doc_id is None:
            raise IndexingDocumentError("Document ID is required for upsert operation")

        try:
            await self._client.index(
                index=index_name,
                id=doc_id,
                document=self._model_to_dict(document),
                refresh=True,
            )
            return True

        except Exception as e:
            raise IndexingDocumentError(f"Failed to upsert document in '{index_name}': {str(e)}") from e

    async def bulk_insert(
        self,
        model_class: Type[BaseIndexModel],
        documents: List[BaseIndexModel],
        batch_size: Optional[int] = None
    ) -> int:
        """批量插入文档"""
        if not documents:
            return 0

        index_name = model_class.get_index_name()
        batch_size = batch_size or self.batch_size

        success_count = 0

        # 分批处理
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            # 构建 bulk 操作
            operations = []
            for doc in batch:
                doc_id = self._get_document_id(doc)
                operations.append({"index": {"_index": index_name, "_id": doc_id}})
                operations.append(self._model_to_dict(doc))

            try:
                # 添加 refresh 参数使文档立即可见
                response = await self._client.bulk(operations=operations, refresh=True)

                # 检查错误
                if response.get("errors"):
                    # 记录失败的操作，只计算成功的
                    for item in response["items"]:
                        if "index" in item:
                            result = item["index"]
                            if result.get("status", 200) >= 400:
                                logger.warning(
                                    f"Bulk insert error: index={index_name}, "
                                    f"id={result.get('_id')}, "
                                    f"error={result.get('error')}"
                                )
                            else:
                                # 只有成功的状态码才计数
                                success_count += 1
                else:
                    success_count += len(batch)

            except Exception as e:
                raise IndexingDocumentError(f"Failed to bulk insert into '{index_name}': {str(e)}") from e

        return success_count

    async def bulk_update(
        self,
        model_class: Type[BaseIndexModel],
        documents: List[BaseIndexModel],
        batch_size: Optional[int] = None
    ) -> int:
        """批量更新文档"""
        if not documents:
            return 0

        index_name = model_class.get_index_name()
        batch_size = batch_size or self.batch_size

        success_count = 0

        # 分批处理
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            # 构建 bulk 操作
            operations = []
            for doc in batch:
                doc_id = self._get_document_id(doc)
                if doc_id is None:
                    logger.warning("Skipping document without ID in bulk update")
                    continue
                operations.append({"update": {"_index": index_name, "_id": doc_id}})
                operations.append({"doc": self._model_to_dict(doc)})

            try:
                # 添加 refresh 参数使文档立即可见
                response = await self._client.bulk(operations=operations, refresh=True)

                # 检查错误
                if response.get("errors"):
                    for item in response["items"]:
                        if "update" in item:
                            result = item["update"]
                            if result.get("status", 0) < 400:
                                success_count += 1
                            else:
                                logger.warning(
                                    f"Bulk update error: index={index_name}, "
                                    f"id={result.get('_id')}, "
                                    f"error={result.get('error')}"
                                )
                else:
                    success_count += len(batch)

            except Exception as e:
                raise IndexingDocumentError(f"Failed to bulk update in '{index_name}': {str(e)}") from e

        return success_count

    async def bulk_upsert(
        self,
        model_class: Type[BaseIndexModel],
        documents: List[BaseIndexModel],
        batch_size: Optional[int] = None
    ) -> int:
        """批量插入或更新文档"""
        if not documents:
            return 0

        index_name = model_class.get_index_name()
        batch_size = batch_size or self.batch_size

        success_count = 0

        # 分批处理
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            # 构建 bulk 操作（使用 index 操作实现 upsert）
            operations = []
            for doc in batch:
                doc_id = self._get_document_id(doc)
                if doc_id is None:
                    logger.warning("Skipping document without ID in bulk upsert")
                    continue
                operations.append({"index": {"_index": index_name, "_id": doc_id}})
                operations.append(self._model_to_dict(doc))

            try:
                response = await self._client.bulk(operations=operations, refresh=True)

                # 检查错误
                if response.get("errors"):
                    for item in response["items"]:
                        if "index" in item:
                            result = item["index"]
                            if result.get("status", 0) < 400:
                                success_count += 1
                            else:
                                logger.warning(
                                    f"Bulk upsert error: index={index_name}, "
                                    f"id={result.get('_id')}, "
                                    f"error={result.get('error')}"
                                )
                else:
                    success_count += len(batch)

            except Exception as e:
                raise IndexingDocumentError(f"Failed to bulk upsert in '{index_name}': {str(e)}") from e

        return success_count

    async def delete(
        self,
        model_class: Type[BaseIndexModel],
        doc_id: Optional[str] = None,
        query: Optional[BoolQuery] = None
    ) -> int:
        """删除文档"""
        index_name = model_class.get_index_name()

        # 删除单个文档
        if doc_id is not None:
            try:
                await self._client.delete(index=index_name, id=doc_id)
                return 1
            except NotFoundError:
                logger.warning(f"Document with ID '{doc_id}' not found in index '{index_name}'")
                return 0
            except Exception as e:
                raise IndexingDocumentError(f"Failed to delete document from '{index_name}': {str(e)}") from e

        # 根据查询条件删除
        if query is not None:
            es_query = self._build_es_query(query)
            try:
                response = await self._client.delete_by_query(
                    index=index_name,
                    query=es_query,
                    refresh=True,
                )
                return response.get("deleted", 0)
            except Exception as e:
                raise IndexingDocumentError(f"Failed to delete documents from '{index_name}': {str(e)}") from e

        raise IndexingDocumentError("Either doc_id or query must be provided for delete operation")

    async def get_by_id(
        self,
        model_class: Type[BaseIndexModel],
        doc_id: str
    ) -> Optional[BaseIndexModel]:
        """根据 ID 获取文档"""
        index_name = model_class.get_index_name()

        try:
            response = await self._client.get(index=index_name, id=doc_id)
            return self._dict_to_model(model_class, response["_source"], doc_id=response["_id"])

        except NotFoundError:
            return None
        except Exception as e:
            raise IndexingQueryError(f"Failed to get document from '{index_name}': {str(e)}") from e

    # =========================================================================
    # 搜索操作
    # =========================================================================

    async def search(
        self,
        model_class: Type[BaseIndexModel],
        query: SearchQuery
    ) -> SearchResult[BaseIndexModel]:
        """搜索文档"""
        index_name = model_class.get_index_name()

        start_time = datetime.now()

        try:
            # 构建 Elasticsearch 查询
            es_query = {}
            size = query.limit
            from_ = query.offset

            # 文本查询或布尔查询
            if query.query:
                # 简单文本查询 - 使用 multi_match 在所有文本字段中搜索
                field_defs = model_class.get_field_definitions()
                text_fields = [f.name for f in field_defs if f.type == FieldType.text]

                if text_fields:
                    # 有文本字段，使用 multi_match
                    es_query["query"] = {
                        "multi_match": {
                            "query": query.query,
                            "fields": text_fields,
                            "type": "best_fields"
                        }
                    }
                else:
                    # 没有文本字段，使用 match_all
                    es_query["query"] = {"match_all": {}}
            elif query.bool_query:
                # 布尔查询
                es_query["query"] = self._build_es_query(query.bool_query)

            # 向量检索
            elif query.vector_param:
                es_query = self._build_vector_query(query.vector_param, query.fields)
                size = query.vector_param.k

            # 混合检索
            elif query.hybrid_param:
                es_query = self._build_hybrid_query(query.hybrid_param, query.fields)
                size = query.hybrid_param.vector_param.k if query.hybrid_param.vector_param else query.limit

            # 排序
            if query.sort:
                es_query["sort"] = [
                    {item["field"]: {"order": item.get("order", "asc")}}
                    for item in query.sort
                ]

            # 返回字段
            source = query.fields
            if not query.include_vectors:
                # 排除向量字段
                vector_fields = self._get_vector_fields(model_class)
                if source is None:
                    # 如果没有指定返回字段，则排除向量字段
                    if vector_fields:
                        es_query["_source"] = {"excludes": vector_fields}
                else:
                    # 如果指定了返回字段，则过滤掉向量字段
                    es_query["_source"] = [f for f in source if f not in vector_fields]

            # 执行搜索
            response = await self._client.search(
                index=index_name,
                query=es_query.get("query", {"match_all": {}}),
                knn=es_query.get("knn"),
                sort=es_query.get("sort"),
                source=es_query.get("_source"),
                size=size,
                from_=from_,
            )

            # 解析结果
            hits = response["hits"]
            total = hits["total"]["value"]
            documents = []
            scores = []

            for hit in hits["hits"]:
                # 如果指定了返回字段列表，创建子模型类
                if query.fields is not None:
                    field_set = set(query.fields)
                    # 确保 id 字段包含在结果中
                    if hit.get("_id"):
                        field_set.add("id")
                    result_model = create_sub_fields_model(model_class, field_set)
                elif not query.include_vectors:
                    # 如果排除了向量字段，创建不包含向量字段的子模型
                    field_defs = model_class.get_field_definitions()
                    vector_fields = {f.name for f in field_defs if f.type in [FieldType.dense_vector, FieldType.sparse_vector]}
                    if vector_fields:
                        field_set = {f.name for f in field_defs} - vector_fields
                        if hit.get("_id"):
                            field_set.add("id")
                        result_model = create_sub_fields_model(model_class, field_set)
                    else:
                        result_model = model_class
                else:
                    result_model = model_class

                doc = self._dict_to_model(result_model, hit["_source"], doc_id=hit["_id"])
                documents.append(doc)
                if query.include_scores:
                    # 处理分数：排序时 _score 可能为 None
                    score = hit.get("_score")
                    if score is None:
                        score = 0.0
                    scores.append(float(score))

            query_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            return SearchResult(
                documents=documents,
                total=total,
                scores=scores if query.include_scores else None,
                has_more=from_ + size < total,
                query_time_ms=query_time_ms,
            )

        except Exception as e:
            raise IndexingQueryError(f"Failed to search in '{index_name}': {str(e)}") from e

    async def count(
        self,
        model_class: Type[BaseIndexModel],
        query: Optional[BoolQuery] = None
    ) -> int:
        """统计文档数量"""
        index_name = model_class.get_index_name()

        try:
            if query is None:
                response = await self._client.count(index=index_name)
            else:
                es_query = self._build_es_query(query)
                response = await self._client.count(index=index_name, query=es_query)

            return response.get("count", 0)

        except Exception as e:
            raise IndexingQueryError(f"Failed to count documents in '{index_name}': {str(e)}") from e

    # =========================================================================
    # 工具方法
    # =========================================================================

    def _generate_mapping(self, model_class: Type[BaseIndexModel]) -> Dict[str, Any]:
        """生成 Elasticsearch mapping"""
        field_defs = model_class.get_field_definitions()
        index_config = model_class.get_index_config()

        mapping = {"properties": {}}

        for field_def in field_defs:
            field_mapping = self._field_to_mapping(field_def)
            mapping["properties"][field_def.name] = field_mapping

        # 添加动态映射配置
        dynamic = index_config.get("dynamic", "strict")
        mapping["dynamic"] = dynamic

        return mapping

    def _field_to_mapping(self, field_def: FieldDefinition) -> Dict[str, Any]:
        """将字段定义转换为 Elasticsearch mapping"""
        es_type = self.FIELD_TYPE_MAPPING.get(field_def.type)

        if es_type is None:
            raise ValueError(f"Unsupported field type: {field_def.type}")

        mapping = {"type": es_type}

        # 向量字段配置
        if field_def.type in [FieldType.dense_vector, FieldType.sparse_vector]:
            if field_def.dimension:
                mapping["dims"] = field_def.dimension
            if field_def.metric_type:
                mapping["index"] = True
                mapping["similarity"] = field_def.metric_type.lower()
            else:
                mapping["index"] = field_def.index

        # 文本字段配置
        elif field_def.type == FieldType.text:
            if field_def.analyzer:
                mapping["analyzer"] = field_def.analyzer
            mapping["index"] = field_def.index

        # 通用字段配置
        else:
            mapping["store"] = field_def.store
            mapping["index"] = field_def.index

        return mapping

    def _generate_settings(self, model_class: Type[BaseIndexModel]) -> Dict[str, Any]:
        """生成 Elasticsearch index settings"""
        index_config = model_class.get_index_config()

        settings = {
            "number_of_shards": index_config.get("number_of_shards", 1),
            "number_of_replicas": index_config.get("number_of_replicas", 1),
        }

        # 添加自定义设置
        if "settings" in index_config:
            settings.update(index_config["settings"])

        return settings

    def _build_es_query(self, bool_query: BoolQuery) -> Dict[str, Any]:
        """构建 Elasticsearch 布尔查询"""
        es_query = {"bool": {}}

        if bool_query.must:
            es_query["bool"]["must"] = [
                self._build_condition(cond) for cond in bool_query.must
            ]

        if bool_query.should:
            es_query["bool"]["should"] = [
                self._build_condition(cond) for cond in bool_query.should
            ]
            if bool_query.minimum_should_match:
                es_query["bool"]["minimum_should_match"] = bool_query.minimum_should_match

        if bool_query.must_not:
            es_query["bool"]["must_not"] = [
                self._build_condition(cond) for cond in bool_query.must_not
            ]

        if bool_query.filter:
            es_query["bool"]["filter"] = [
                self._build_condition(cond) for cond in bool_query.filter
            ]

        return es_query

    def _build_condition(self, condition: Union[QueryCondition, BoolQuery]) -> Dict[str, Any]:
        """构建查询条件"""
        if isinstance(condition, BoolQuery):
            return self._build_es_query(condition)

        # QueryCondition
        if condition.match_type == MatchType.term:
            return {"term": {condition.field: {"value": condition.value, "boost": condition.boost}}}

        elif condition.match_type == MatchType.terms:
            return {"terms": {condition.field: condition.values, "boost": condition.boost}}

        elif condition.match_type == MatchType.match:
            return {"match": {condition.field: {"query": condition.value, "boost": condition.boost}}}

        elif condition.match_type == MatchType.match_phrase:
            return {"match_phrase": {condition.field: {"query": condition.value, "boost": condition.boost}}}

        elif condition.match_type == MatchType.range:
            range_query = {}
            if condition.range_gte is not None:
                range_query["gte"] = condition.range_gte
            if condition.range_gt is not None:
                range_query["gt"] = condition.range_gt
            if condition.range_lte is not None:
                range_query["lte"] = condition.range_lte
            if condition.range_lt is not None:
                range_query["lt"] = condition.range_lt
            return {"range": {condition.field: range_query}}

        elif condition.match_type == MatchType.wildcard:
            return {"wildcard": {condition.field: {"value": condition.value, "boost": condition.boost}}}

        elif condition.match_type == MatchType.prefix:
            return {"prefix": {condition.field: {"value": condition.value, "boost": condition.boost}}}

        elif condition.match_type == MatchType.fuzzy:
            return {"fuzzy": {condition.field: {"value": condition.value, "boost": condition.boost}}}

        else:
            raise ValueError(f"Unsupported match type: {condition.match_type}")

    def _build_vector_query(
        self,
        vector_param: Any,
        fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """构建向量检索查询"""
        # 查找向量字段
        vector_field = self._find_vector_field(fields)

        knn_query = {
            "field": vector_field,
            "query_vector": vector_param.vector,
            "k": vector_param.k,
            "num_candidates": vector_param.k * 10,  # 默认取 10 倍候选
        }

        if vector_param.filter:
            # filter 可以是 BoolQuery 或 Dict[str, Any]
            if isinstance(vector_param.filter, BoolQuery):
                knn_query["filter"] = self._build_es_query(vector_param.filter)
            elif isinstance(vector_param.filter, dict):
                # 如果是 dict，直接使用作为 ES query DSL
                knn_query["filter"] = vector_param.filter
            else:
                logger.warning(f"Unsupported filter type: {type(vector_param.filter)}")

        if vector_param.ef:
            knn_query["num_candidates"] = vector_param.ef

        return {"knn": knn_query}

    def _build_hybrid_query(
        self,
        hybrid_param: Any,
        fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """构建混合检索查询"""
        query = {}

        # 向量检索
        if hybrid_param.vector_param:
            knn_query = self._build_vector_query(hybrid_param.vector_param, fields)
            query["knn"] = knn_query["knn"]

        # 稀疏检索
        if hybrid_param.sparse_query:
            bool_query = self._build_es_query(hybrid_param.sparse_query)
            query["query"] = bool_query

        return query

    def _get_vector_fields(self, model_class: Type[BaseIndexModel]) -> List[str]:
        """获取向量字段列表"""
        field_defs = model_class.get_field_definitions()
        return [f.name for f in field_defs if f.type in [FieldType.dense_vector, FieldType.sparse_vector]]

    def _find_vector_field(self, fields: Optional[List[str]] = None) -> str:
        """查找向量字段名称"""
        if fields:
            for field in fields:
                if "vector" in field.lower() or "embedding" in field.lower():
                    return field
        # 默认返回第一个向量字段（需要在实际使用时从模型中获取）
        return "embedding"

    def _get_document_id(self, document: BaseIndexModel) -> str | None:
        """从文档中提取 ID"""
        # 假设文档有 id 字段
        doc_id = getattr(document, "id", None)
        if doc_id is None:
            return None
        return str(doc_id)

    def _model_to_dict(self, document: BaseIndexModel) -> Dict[str, Any]:
        """将模型转换为字典"""
        data = document.model_dump(exclude_none=True)
        # 移除内部字段
        data.pop("_provider", None)
        return data

    def _dict_to_model(
        self,
        model_class: Type[BaseIndexModel],
        data: Dict[str, Any],
        doc_id: Optional[str] = None
    ) -> BaseIndexModel:
        """将字典转换为模型，结合字段定义进行智能类型转换"""

        # 获取字段定义
        field_defs = model_class.get_field_definitions()

        # 创建字段名到定义的映射
        field_type_map = {f.name: f.type for f in field_defs}

        # 检查是否为子模型（部分字段模型）
        # 通过比较 get_field_definitions 返回的字段数量和 model_fields 数量

        # 处理每个字段，根据其定义进行类型转换
        processed_data = {}
        for key, value in data.items():
            field_type = field_type_map.get(key)

            # 如果没有字段定义，直接使用原始值
            if field_type is None:
                processed_data[key] = value
                continue

            # 根据字段类型进行转换
            try:
                if field_type == FieldType.datetime or field_type == FieldType.date:
                    # 日期时间字段：ES 返回 ISO 格式字符串
                    if isinstance(value, str):
                        # 处理 ISO 8601 格式
                        processed_data[key] = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    elif isinstance(value, (int, float)):
                        # ES 可能返回时间戳（毫秒）
                        processed_data[key] = datetime.fromtimestamp(value / 1000)
                    else:
                        processed_data[key] = value

                elif field_type == FieldType.integer:
                    # 整数字段
                    if isinstance(value, (int, float)):
                        processed_data[key] = int(value)
                    elif isinstance(value, str) and value.isdigit():
                        processed_data[key] = int(value)
                    else:
                        processed_data[key] = value

                elif field_type == FieldType.long:
                    # 长整数字段
                    if isinstance(value, (int, float)):
                        processed_data[key] = int(value)
                    else:
                        processed_data[key] = value

                elif field_type == FieldType.float or field_type == FieldType.double:
                    # 浮点数字段
                    if isinstance(value, (int, float)):
                        processed_data[key] = float(value)
                    else:
                        processed_data[key] = value

                elif field_type == FieldType.boolean:
                    # 布尔字段
                    if isinstance(value, str):
                        processed_data[key] = value.lower() in ("true", "1", "yes")
                    else:
                        processed_data[key] = bool(value)

                else:
                    # 其他类型保持原样
                    processed_data[key] = value

            except (ValueError, AttributeError, TypeError) as e:
                logger.warning(
                    f"Failed to convert field '{key}' to type {field_type}: {str(e)}. Using original value."
                )
                processed_data[key] = value

        # 添加文档 ID
        if doc_id is not None:
            processed_data["id"] = doc_id

        return model_class.model_validate(processed_data, strict=False)
