"""
Milvus Provider 实现

提供 Milvus 后端的索引操作和 CRUD 功能。
支持稠密向量检索和标量字段过滤。
"""

from typing import Any, Dict, List, Optional, Type, Union
from datetime import datetime
from loguru import logger

from pymilvus import (
    MilvusClient,
    DataType,
    MilvusException,
    connections,
    utility,
    FieldSchema,
)
from pymilvus.milvus_client.index import IndexParams
from pymilvus.orm import collection as milvus_collection

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


class MilvusProvider(BaseProvider):
    """Milvus Provider 实现

    使用 Milvus 作为索引后端，支持：
    - 稠密索引（向量检索）
    - 标量字段过滤（通过 expression）
    """

    # Milvus 字段类型映射
    FIELD_TYPE_MAPPING = {
        FieldType.integer: DataType.INT64,
        FieldType.long: DataType.INT64,
        FieldType.float: DataType.FLOAT,
        FieldType.double: DataType.DOUBLE,
        FieldType.boolean: DataType.BOOL,
        FieldType.keyword: DataType.VARCHAR,
        FieldType.text: DataType.VARCHAR,
        FieldType.datetime: DataType.INT64,  # 使用 timestamp
        FieldType.date: DataType.INT64,  # 使用 timestamp
        FieldType.dense_vector: DataType.FLOAT_VECTOR,
        FieldType.sparse_vector: DataType.SPARSE_FLOAT_VECTOR,
        FieldType.json: DataType.JSON,  # JSON 类型，支持动态字段
    }

    # 默认配置
    DEFAULT_TIMEOUT = 30
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_BATCH_SIZE = 100
    DEFAULT_VARCHAR_MAX_LENGTH = 65535
    DEFAULT_INDEX_TYPE = "IVF_FLAT"
    DEFAULT_METRIC_TYPE = "L2"

    def __init__(self, backend_type, config):
        """
        初始化 Milvus Provider

        Args:
            backend_type: 后端类型（固定为 IndexingBackendTypeEnum.milvus）
            config: 配置字典，包含以下字段：
                - host: 主机地址（必需）
                - port: 端口（可选，默认 19530）
                - username: 用户名（可选）
                - password: 密码（可选）
                - api_key: API 密钥（可选，用于 Zilliz Cloud）
                - secure: 是否使用 TLS（默认 False）
                - timeout: 请求超时时间（可选）
                - db_name: 数据库名称（可选）
        """
        super().__init__(backend_type, config)

        # 提取配置
        self.host = config["host"]
        self.port = config.get("port", 19530)
        self.secure = config.get("secure", False)
        self.timeout = config.get("timeout", self.DEFAULT_TIMEOUT)
        self.max_retries = config.get("max_retries", self.DEFAULT_MAX_RETRIES)
        self.batch_size = config.get("batch_size", self.DEFAULT_BATCH_SIZE)
        self.db_name = config.get("db_name", "default")
        self.alias = config.get("alias", "default")

        # 检查是否使用 Zilliz Cloud
        self.api_key = config.get("api_key")

        # Milvus 客户端实例
        self._client: Optional[MilvusClient] = None

    async def connect(self) -> None:
        """建立 Milvus 连接"""
        if self._client is not None:
            return

        try:
            # 构建连接参数
            connect_params = {
                "host": self.host,
                "port": self.port,
                "timeout": self.timeout,
                "alias": self.alias,
            }

            # 添加认证信息
            if "username" in self.config and "password" in self.config:
                connect_params["user"] = self.config["username"]
                connect_params["password"] = self.config["password"]

            # 添加 TLS 配置
            if self.secure:
                connect_params["secure"] = True

            # 如果使用 Zilliz Cloud（API Key 方式），只使用 MilvusClient，不需要 connections.connect()
            if self.api_key:
                # 创建 MilvusClient（使用 API Key 认证）
                self._client = MilvusClient(
                    uri=f"https://{self.host}:{self.port}",
                    token=self.api_key,
                    timeout=self.timeout,
                )
            else:
                # 自托管 Milvus 或使用用户名/密码认证
                # 先使用旧版连接方式建立连接
                connections.connect(**connect_params)

                # 创建 MilvusClient（共享已有连接）
                # 使用 alias 参数而不是 uri，以使用已建立的连接
                self._client = MilvusClient(
                    self.alias,
                    token="",
                    timeout=self.timeout,
                )

            # 测试连接
            await self.ping()

            logger.info(f"Milvus provider connected: {self.host}:{self.port}")

        except MilvusException as e:
            raise IndexingBackendError(f"Failed to connect to Milvus: {str(e)}") from e
        except Exception as e:
            raise IndexingBackendError(f"Unexpected error connecting to Milvus: {str(e)}") from e

    async def disconnect(self) -> None:
        """断开 Milvus 连接"""
        if self._client is not None:
            try:
                self._client.close()
            except Exception as e:
                logger.warning(f"Error closing Milvus client: {str(e)}")
            finally:
                self._client = None

            try:
                connections.disconnect(alias=self.alias)
            except Exception as e:
                logger.warning(f"Error disconnecting from Milvus: {str(e)}")

            logger.info("Milvus provider disconnected")

    async def ping(self) -> bool:
        """检查连接是否正常"""
        if self._client is None:
            return False

        try:
            # 尝试列出 collection
            self._client.list_collections() # type: ignore
            return True
        except Exception as e:
            logger.warning(f"Milvus ping failed: {str(e)}")
            return False

    # =========================================================================
    # 索引操作
    # =========================================================================

    async def create_index(self, model_class: Type[BaseIndexModel]) -> bool:
        """创建索引（Milvus 中称为 Collection）"""
        collection_name = model_class.get_index_name()

        # 检查 collection 是否已存在
        if await self.index_exists(model_class):
            logger.warning(f"Collection '{collection_name}' already exists")
            return False

        # 获取字段定义
        field_defs = model_class.get_field_definitions()

        # 转换为 Milvus 字段定义
        schema_fields = []
        primary_key_field = None
        vector_fields = []

        for field_def in field_defs:
            milvus_field = self._field_to_schema(field_def)
            schema_fields.append(milvus_field)

            # 查找主键字段
            if field_def.name == "id":
                primary_key_field = milvus_field

            # 记录向量字段
            if field_def.type in [FieldType.dense_vector, FieldType.sparse_vector]:
                vector_fields.append(field_def)

        # 如果没有定义 id 字段，自动添加
        if primary_key_field is None:
            primary_key_field = FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=True,
                max_length=255,
            )
            schema_fields.insert(0, primary_key_field)

        # 创建 schema
        from pymilvus import CollectionSchema

        schema = CollectionSchema(
            fields=schema_fields,
            description=collection_name,
            enable_dynamic_field=True,  # 允许动态字段
        )

        try:
            # 创建 collection
            self._client.create_collection( # type: ignore
                collection_name=collection_name,
                schema=schema,
                # properties={"partitionkey.isolation": True}
            )

            # 为向量字段创建索引
            index_config = model_class.get_index_config()
            for field_def in vector_fields:
                index_name = f"{field_def.name}_index"
                index_type = index_config.get("index_type", self.DEFAULT_INDEX_TYPE)
                metric_type = field_def.metric_type or index_config.get("metric_type", self.DEFAULT_METRIC_TYPE)

                # 创建 IndexParams 对象
                index_params_obj = IndexParams()
                index_params_obj.add_index(
                    field_name=field_def.name,
                    index_type=index_type,
                    metric_type=metric_type,
                    params=index_config.get("index_params", {"nlist": 128}),
                    index_name=index_name,
                )

                self._client.create_index( # type: ignore
                    collection_name=collection_name,
                    index_params=index_params_obj,
                )

                logger.info(f"Created index '{index_name}' on field '{field_def.name}'")

            # 加载 collection 到内存
            self._client.load_collection(collection_name=collection_name) # type: ignore

            logger.info(f"Created collection '{collection_name}'")
            return True

        except MilvusException as e:
            raise IndexingIndexError(f"Failed to create collection '{collection_name}': {str(e)}") from e
        except Exception as e:
            raise IndexingIndexError(f"Failed to create collection '{collection_name}': {str(e)}") from e

    async def drop_index(self, model_class: Type[BaseIndexModel]) -> bool:
        """删除索引（Collection）"""
        collection_name = model_class.get_index_name()

        try:
            # 使用 drop_collection 而不是 delete，这样更彻底
            self._client.drop_collection(collection_name=collection_name) # type: ignore
            logger.info(f"Dropped collection '{collection_name}'")
            return True

        except MilvusException as e:
            # 如果 collection 不存在，视为成功
            if "not found" in str(e).lower():
                logger.warning(f"Collection '{collection_name}' not found")
                return True
            raise IndexingIndexError(f"Failed to drop collection '{collection_name}': {str(e)}") from e
        except Exception as e:
            raise IndexingIndexError(f"Failed to drop collection '{collection_name}': {str(e)}") from e

    async def index_exists(self, model_class: Type[BaseIndexModel]) -> bool:
        """检查索引是否存在"""
        collection_name = model_class.get_index_name()

        try:
            return self._client.has_collection(collection_name=collection_name) # type: ignore
        except Exception as e:
            logger.warning(f"Failed to check collection existence for '{collection_name}': {str(e)}")
            return False

    # =========================================================================
    # 文档操作
    # =========================================================================

    async def insert(self, document: BaseIndexModel) -> bool:
        """插入单条文档"""
        collection_name = document.get_index_name()

        try:
            # 转换为字典
            data = self._model_to_dict(document)

            # 插入数据
            self._client.insert(collection_name=collection_name, data=[data]) # type: ignore

            return True

        except MilvusException as e:
            raise IndexingDocumentError(f"Failed to insert document into '{collection_name}': {str(e)}") from e
        except Exception as e:
            raise IndexingDocumentError(f"Failed to insert document into '{collection_name}': {str(e)}") from e

    async def update(self, document: BaseIndexModel, doc_id: Optional[str] = None) -> bool:
        """更新文档

        注意：Milvus 不支持直接更新，需要先删除再插入（upsert）
        """
        return await self.upsert(document, doc_id=doc_id)

    async def upsert(self, document: BaseIndexModel, doc_id: Optional[str] = None) -> bool:
        """插入或更新文档"""
        collection_name = document.get_index_name()

        try:
            # 转换为字典
            data = self._model_to_dict(document)

            # 如果指定了 doc_id，添加到数据中
            if doc_id is not None:
                data["id"] = doc_id

            # Upsert 数据
            self._client.upsert(collection_name=collection_name, data=[data]) # type: ignore

            return True

        except MilvusException as e:
            raise IndexingDocumentError(f"Failed to upsert document into '{collection_name}': {str(e)}") from e
        except Exception as e:
            raise IndexingDocumentError(f"Failed to upsert document into '{collection_name}': {str(e)}") from e

    async def bulk_insert(
        self,
        model_class: Type[BaseIndexModel],
        documents: List[BaseIndexModel],
        batch_size: Optional[int] = None
    ) -> int:
        """批量插入文档"""
        if not documents:
            return 0

        collection_name = model_class.get_index_name()
        batch_size = batch_size or self.batch_size

        # 转换为字典列表
        data_list = [self._model_to_dict(doc) for doc in documents]

        try:
            # 分批插入
            success_count = 0
            for i in range(0, len(data_list), batch_size): # type: ignore
                batch = data_list[i:i + batch_size] # type: ignore

                self._client.insert(collection_name=collection_name, data=batch) # type: ignore
                success_count += len(batch)

            return success_count

        except MilvusException as e:
            raise IndexingDocumentError(f"Failed to bulk insert into '{collection_name}': {str(e)}") from e
        except Exception as e:
            raise IndexingDocumentError(f"Failed to bulk insert into '{collection_name}': {str(e)}") from e

    async def bulk_update(
        self,
        model_class: Type[BaseIndexModel],
        documents: List[BaseIndexModel],
        batch_size: Optional[int] = None
    ) -> int:
        """批量更新文档

        注意：Milvus 不支持直接更新，使用 upsert
        """
        return await self.bulk_upsert(model_class, documents, batch_size=batch_size)

    async def bulk_upsert(
        self,
        model_class: Type[BaseIndexModel],
        documents: List[BaseIndexModel],
        batch_size: Optional[int] = None
    ) -> int:
        """批量插入或更新文档"""
        if not documents:
            return 0

        collection_name = model_class.get_index_name()
        batch_size = batch_size or self.batch_size

        # 转换为字典列表
        data_list = [self._model_to_dict(doc) for doc in documents]

        try:
            # 分批 upsert
            success_count = 0
            for i in range(0, len(data_list), batch_size): # type: ignore
                batch = data_list[i:i + batch_size] # type: ignore

                self._client.upsert(collection_name=collection_name, data=batch) # type: ignore
                success_count += len(batch)

            return success_count

        except MilvusException as e:
            raise IndexingDocumentError(f"Failed to bulk upsert into '{collection_name}': {str(e)}") from e
        except Exception as e:
            raise IndexingDocumentError(f"Failed to bulk upsert into '{collection_name}': {str(e)}") from e

    async def delete(
        self,
        model_class: Type[BaseIndexModel],
        doc_id: Optional[str] = None,
        query: Optional[BoolQuery] = None
    ) -> int:
        """删除文档"""
        collection_name = model_class.get_index_name()

        try:
            # 删除单个文档
            if doc_id is not None:
                self._client.delete( # type: ignore
                    collection_name=collection_name,
                    ids=[doc_id]
                )
                return 1

            # 根据查询条件删除
            if query is not None:
                # 将 BoolQuery 转换为 Milvus expression
                expression = self._build_expression(query)
                self._client.delete( # type: ignore
                    collection_name=collection_name,
                    filter=expression
                )
                # Milvus delete 不返回删除数量，返回一个估算值
                return -1

            raise IndexingDocumentError("Either doc_id or query must be provided for delete operation")

        except MilvusException as e:
            raise IndexingDocumentError(f"Failed to delete documents from '{collection_name}': {str(e)}") from e
        except Exception as e:
            raise IndexingDocumentError(f"Failed to delete documents from '{collection_name}': {str(e)}") from e

    async def get_by_id(
        self,
        model_class: Type[BaseIndexModel],
        doc_id: str
    ) -> Optional[BaseIndexModel]:
        """根据 ID 获取文档"""
        collection_name = model_class.get_index_name()

        try:
            # 获取字段定义
            field_defs = model_class.get_field_definitions()

            results = self._client.get( # type: ignore
                collection_name=collection_name,
                ids=[doc_id],
            )

            if not results or len(results) == 0:
                return None

            # 返回第一条记录
            data = results[0]
            # 提取 id 字段（Milvus 返回的数据可能不包含 id）
            if "id" not in data:
                data["id"] = doc_id

            return self._dict_to_model(model_class, data)

        except MilvusException as e:
            if "not found" in str(e).lower():
                return None
            raise IndexingQueryError(f"Failed to get document from '{collection_name}': {str(e)}") from e
        except Exception as e:
            raise IndexingQueryError(f"Failed to get document from '{collection_name}': {str(e)}") from e

    async def flush(self, model_class: Type[BaseIndexModel]) -> bool:
        """手动刷新数据到磁盘

        Milvus 会自动持久化数据，通常不需要手动调用。
        仅在测试或需要立即确保数据持久化时使用。

        Args:
            model_class: 索引模型类

        Returns:
            是否成功刷新
        """
        collection_name = model_class.get_index_name()

        try:
            self._client.flush(collection_name=collection_name) # type: ignore
            return True
        except MilvusException as e:
            raise IndexingIndexError(f"Failed to flush collection '{collection_name}': {str(e)}") from e
        except Exception as e:
            raise IndexingIndexError(f"Failed to flush collection '{collection_name}': {str(e)}") from e

    # =========================================================================
    # 搜索操作
    # =========================================================================

    async def search(
        self,
        model_class: Type[BaseIndexModel],
        query: SearchQuery
    ) -> SearchResult[BaseIndexModel]:
        """搜索文档"""
        collection_name = model_class.get_index_name()

        start_time = datetime.now()

        try:
            # 获取向量字段
            vector_fields = self._get_vector_fields(model_class)
            vector_field = vector_fields[0] if vector_fields else "embedding"

            # 获取 partition key 字段列表
            field_defs = model_class.get_field_definitions()
            partition_key_fields = [field.name for field in field_defs if field.is_partition_key]

            # 执行搜索
            search_params = {
                "collection_name": collection_name,
                "limit": query.limit,
                "output_fields": [f.name for f in field_defs],  # 包含所有字段，包括 JSON 字段
            }

            # 支持分区检索
            if query.partition_name:
                search_params["partition_names"] = [query.partition_name]
                # 如果有 partition key，警告用户不应同时使用
                if partition_key_fields:
                    logger.warning(
                        f"Searching in specific partition '{query.partition_name}' while collection has partition key(s): "
                        f"{partition_key_fields}. Partition key filtering may be more efficient."
                    )

            # 处理 offset
            if query.offset > 0:
                search_params["offset"] = query.offset

            # 构建过滤表达式
            filter_expr = None
            if query.bool_query:
                filter_expr = self._build_expression(query.bool_query, model_class)
                search_params["filter"] = filter_expr

            # 向量检索
            if query.vector_param:
                search_params["data"] = [query.vector_param.vector]
                search_params["anns_field"] = vector_field
                search_params["search_params"] = {
                    "metric_type": query.vector_param.metric_type,
                    "params": {"nprobe": 10},  # 默认探测 10 个 cluster
                }
                if query.vector_param.ef:
                    search_params["search_params"]["params"]["ef"] = query.vector_param.ef
                if query.vector_param.k:
                    search_params["limit"] = query.vector_param.k

                if query.vector_param.filter:
                    # 合并 filter 表达式
                    vector_filter_expr = self._build_expression(query.vector_param.filter, model_class) # type: ignore
                    if filter_expr:
                        filter_expr = f"({filter_expr}) and ({vector_filter_expr})"
                    else:
                        filter_expr = vector_filter_expr

                if filter_expr:
                    search_params["filter"] = filter_expr

                results = self._client.search(**search_params) # type: ignore

            elif query.hybrid_param:
                # 混合检索（Milvus 本身不支持混合，这里只使用向量检索）
                vector_param = query.hybrid_param.vector_param
                search_params["data"] = [vector_param.vector] # type: ignore
                search_params["anns_field"] = vector_field
                search_params["search_params"] = {
                    "metric_type": vector_param.metric_type, # type: ignore
                    "params": {"nprobe": 10},
                }
                if vector_param.k: # type: ignore
                    search_params["limit"] = vector_param.k # type: ignore

                if query.hybrid_param.sparse_query:
                    filter_expr = self._build_expression(query.hybrid_param.sparse_query, model_class)
                    search_params["filter"] = filter_expr

                results = self._client.search(**search_params) # type: ignore

            elif query.query:
                # 文本查询：使用向量检索（需要先 embedding）
                raise IndexingQueryError(
                    "Milvus does not support text query directly. "
                    "Please use vector_param or hybrid_param with pre-computed embeddings."
                )
            else:
                raise IndexingQueryError("Either vector_param or hybrid_param must be provided for Milvus search")

            # 解析结果
            documents = []
            scores = []
            total = 0

            if results and len(results) > 0:
                total = results[0].__len__()  # 估算总数
                for hit in results[0]:
                    data = hit.get("entity", {})
                    # 添加 id 字段
                    if "id" not in data:
                        data["id"] = hit.get("id")
                    doc = self._dict_to_model(model_class, data)
                    documents.append(doc)
                    if query.include_scores:
                        scores.append(hit.get("score", 0.0))

            query_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            return SearchResult(
                documents=documents,
                total=total,
                scores=scores if query.include_scores else None,
                has_more=query.offset + query.limit < total,
                query_time_ms=query_time_ms,
            )

        except MilvusException as e:
            raise IndexingQueryError(f"Failed to search in '{collection_name}': {str(e)}") from e
        except Exception as e:
            raise IndexingQueryError(f"Failed to search in '{collection_name}': {str(e)}") from e

    async def count(
        self,
        model_class: Type[BaseIndexModel],
        query: Optional[BoolQuery] = None
    ) -> int:
        """统计文档数量"""
        collection_name = model_class.get_index_name()

        try:
            if query is None:
                # 统计所有文档
                stats = self._client.get_collection_stats(collection_name=collection_name) # type: ignore
                return stats.get("row_count", 0)
            else:
                # 根据查询条件统计
                # Milvus 没有直接支持，需要先查询后统计
                filter_expr = self._build_expression(query)
                results = self._client.query( # type: ignore
                    collection_name=collection_name,
                    filter=filter_expr,
                    output_fields=["id"],
                )
                return len(results)

        except MilvusException as e:
            raise IndexingQueryError(f"Failed to count documents in '{collection_name}': {str(e)}") from e
        except Exception as e:
            raise IndexingQueryError(f"Failed to count documents in '{collection_name}': {str(e)}") from e

    # =========================================================================
    # 工具方法
    # =========================================================================

    def _field_to_schema(self, field_def: FieldDefinition) -> FieldSchema:
        """将字段定义转换为 Milvus schema 字段定义"""
        milvus_type = self.FIELD_TYPE_MAPPING.get(field_def.type)

        if milvus_type is None:
            raise ValueError(f"Unsupported field type: {field_def.type}")

        # 构建字段参数
        field_params = {
            "name": field_def.name,
            "dtype": milvus_type,
        }

        # 向量字段配置
        if field_def.type in [FieldType.dense_vector, FieldType.sparse_vector]:
            if field_def.dimension:
                field_params["dim"] = field_def.dimension

        # VARCHAR 字段配置
        elif milvus_type == DataType.VARCHAR:
            field_params["max_length"] = self.DEFAULT_VARCHAR_MAX_LENGTH

        # JSON 字段配置（支持动态字段）
        elif milvus_type == DataType.JSON:
            # JSON 字段不需要额外配置，直接使用
            pass

        # 主键配置
        if field_def.name == "id":
            field_params["is_primary"] = True
            field_params["auto_id"] = field_def.auto_id  # 使用字段定义中的 auto_id 设置

        # Partition key 配置
        if field_def.is_partition_key:
            field_params["is_partition_key"] = True

        return FieldSchema(**field_params)

    def _build_expression(self, bool_query: BoolQuery, model_class: Optional[Type[BaseIndexModel]] = None) -> str:
        """构建 Milvus 过滤表达式"""
        expressions = []

        if bool_query.must:
            must_exprs = [self._build_condition_expr(cond, model_class) for cond in bool_query.must]
            if must_exprs:
                expressions.append(f"({' and '.join(must_exprs)})")

        if bool_query.should:
            should_exprs = [self._build_condition_expr(cond, model_class) for cond in bool_query.should]
            if should_exprs:
                min_match = bool_query.minimum_should_match or 1
                # Milvus 不直接支持 minimum_should_match，简化处理
                if len(should_exprs) == min_match:
                    expressions.append(f"({' and '.join(should_exprs)})")
                else:
                    expressions.append(f"({' or '.join(should_exprs)})")

        if bool_query.must_not:
            must_not_exprs = [self._build_condition_expr(cond, model_class) for cond in bool_query.must_not]
            if must_not_exprs:
                expressions.append(f"not ({' and '.join(must_not_exprs)})")

        if bool_query.filter:
            filter_exprs = [self._build_condition_expr(cond, model_class) for cond in bool_query.filter]
            if filter_exprs:
                expressions.append(f"({' and '.join(filter_exprs)})")

        if not expressions:
            return ""

        return " and ".join(expressions)

    def _build_condition_expr(self, condition: Union[QueryCondition, BoolQuery], model_class: Optional[Type[BaseIndexModel]] = None) -> str:
        """构建查询条件表达式"""
        if isinstance(condition, BoolQuery):
            return self._build_expression(condition, model_class)

        # QueryCondition
        field = condition.field
        value = condition.value

        # 检查字段是否包含 JSON 访问语法（例如：extras["special_category"]）
        is_json_access = '["' in field or '["' in field

        # 如果有 model_class，检查是否为 JSON 字段
        if model_class and not is_json_access:
            field_defs = model_class.get_field_definitions()
            field_type_map = {f.name: f.type for f in field_defs}
            field_type = field_type_map.get(field)
            if field_type == FieldType.json:
                # JSON 字段但没有使用访问语法，将整个 JSON 对象当作字符串处理
                is_json_access = True

        # 根据匹配类型构建表达式
        if condition.match_type == MatchType.term:
            if isinstance(value, str):
                return f'{field} == "{value}"'
            else:
                return f"{field} == {value}"

        elif condition.match_type == MatchType.terms:
            # Milvus JSON 字段的 terms 查询需要特殊处理
            if is_json_access:
                values_str = ", ".join([f'"{v}"' if isinstance(v, str) else str(v) for v in condition.values]) # type: ignore
                return f"{field} in [{values_str}]"
            else:
                values_str = ", ".join([f'"{v}"' if isinstance(v, str) else str(v) for v in condition.values]) # type: ignore
                return f"{field} in [{values_str}]"

        elif condition.match_type == MatchType.range:
            range_exprs = []
            if condition.range_gte is not None:
                if isinstance(condition.range_gte, str):
                    range_exprs.append(f'{field} >= "{condition.range_gte}"')
                else:
                    range_exprs.append(f"{field} >= {condition.range_gte}")
            if condition.range_gt is not None:
                if isinstance(condition.range_gt, str):
                    range_exprs.append(f'{field} > "{condition.range_gt}"')
                else:
                    range_exprs.append(f"{field} > {condition.range_gt}")
            if condition.range_lte is not None:
                if isinstance(condition.range_lte, str):
                    range_exprs.append(f'{field} <= "{condition.range_lte}"')
                else:
                    range_exprs.append(f"{field} <= {condition.range_lte}")
            if condition.range_lt is not None:
                if isinstance(condition.range_lt, str):
                    range_exprs.append(f'{field} < "{condition.range_lt}"')
                else:
                    range_exprs.append(f"{field} < {condition.range_lt}")
            return f"({' and '.join(range_exprs)})"

        elif condition.match_type in [MatchType.match, MatchType.match_phrase]:
            # Milvus 不支持全文检索，使用 contains
            return f'{field} like "%{value}%"'

        elif condition.match_type == MatchType.wildcard:
            # Milvus 支持 LIKE，转换 wildcard 为 LIKE
            pattern = value.replace("*", "%").replace("?", "_")
            return f'{field} like "{pattern}"'

        elif condition.match_type == MatchType.prefix:
            return f'{field} like "{value}%"'

        elif condition.match_type == MatchType.fuzzy:
            # Milvus 不支持模糊匹配，使用 contains 作为近似
            return f'{field} like "%{value}%"'

        else:
            raise ValueError(f"Unsupported match type for Milvus: {condition.match_type}")

    def _get_vector_fields(self, model_class: Type[BaseIndexModel]) -> List[str]:
        """获取向量字段列表"""
        field_defs = model_class.get_field_definitions()
        return [f.name for f in field_defs if f.type in [FieldType.dense_vector, FieldType.sparse_vector]]

    def _model_to_dict(self, document: BaseIndexModel) -> Dict[str, Any]:
        """将模型转换为字典"""
        data = document.model_dump(exclude_none=True)
        # 移除内部字段
        data.pop("_provider", None)

        # 转换 datetime 字段为 timestamp（Milvus 使用 int64 存储 datetime）
        field_defs = document.get_field_definitions()
        for field_def in field_defs:
            if field_def.type in [FieldType.datetime, FieldType.date]:
                field_name = field_def.name
                if field_name in data and data[field_name] is not None:
                    if isinstance(data[field_name], datetime):
                        data[field_name] = int(data[field_name].timestamp())

        return data

    def _dict_to_model(
        self,
        model_class: Type[BaseIndexModel],
        data: Dict[str, Any],
        doc_id: Optional[str] = None
    ) -> BaseIndexModel:
        """将字典转换为模型"""
        # 添加文档 ID 到数据中（如果模型有 id 字段）
        if doc_id is not None and "id" not in data and hasattr(model_class, "id"):
            data["id"] = doc_id
        return model_class.model_validate(data)
