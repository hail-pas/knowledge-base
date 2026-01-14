"""
Indexing 模块基类 - 定义索引模型的抽象接口和核心数据结构
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Generic
from typing_extensions import Self
from pydantic import BaseModel, Field, model_validator
from enum import Enum
from loguru import logger

from ext.ext_tortoise.enums import IndexingBackendTypeEnum, IndexingTypeEnum

# 定义类型变量
T = TypeVar('T')


# =============================================================================
# 查询条件相关
# =============================================================================


class MatchType(str, Enum):
    """匹配类型枚举"""

    term = "term"
    """精确匹配"""

    terms = "terms"
    """多值精确匹配"""

    match = "match"
    """全文匹配"""

    match_phrase = "match_phrase"
    """短语匹配"""

    range = "range"
    """范围查询"""

    wildcard = "wildcard"
    """通配符匹配"""

    prefix = "prefix"
    """前缀匹配"""

    fuzzy = "fuzzy"
    """模糊匹配"""


class RangeOperator(str, Enum):
    """范围操作符枚举"""

    gt = "gt"
    """大于"""

    gte = "gte"
    """大于等于"""

    lt = "lt"
    """小于"""

    lte = "lte"
    """小于等于"""


class QueryCondition(BaseModel):
    """查询条件

    用于构建复杂的查询条件树
    """

    field: str
    """字段名"""

    match_type: MatchType
    """匹配类型"""

    value: Any = None
    """匹配值"""

    values: Optional[List[Any]] = None
    """多值匹配时的值列表"""

    range_gte: Optional[Any] = None
    """范围查询 - 大于等于"""

    range_gt: Optional[Any] = None
    """范围查询 - 大于"""

    range_lte: Optional[Any] = None
    """范围查询 - 小于等于"""

    range_lt: Optional[Any] = None
    """范围查询 - 小于"""

    boost: float = 1.0
    """权重提升因子"""

    @model_validator(mode="after")
    def validate_match_type(self) -> Self:
        """验证查询条件"""
        if self.match_type == MatchType.terms:
            if self.values is None or len(self.values) == 0:
                raise ValueError("terms match requires 'values' to be provided")
        elif self.match_type == MatchType.range:
            if not any([self.range_gte, self.range_gt, self.range_lte, self.range_lt]):
                raise ValueError("range match requires at least one range operator")
        else:
            if self.value is None:
                raise ValueError(f"{self.match_type} match requires 'value' to be provided")
        return self


class BoolQuery(BaseModel):
    """布尔查询

    支持 must、should、must_not、filter 组合
    """

    must: Optional[List[Union["QueryCondition", "BoolQuery"]]] = None
    """必须匹配的条件（AND）"""

    should: Optional[List[Union["QueryCondition", "BoolQuery"]]] = None
    """应该匹配的条件（OR）"""

    must_not: Optional[List[Union["QueryCondition", "BoolQuery"]]] = None
    """必须不匹配的条件（NOT）"""

    filter: Optional[List[Union["QueryCondition", "BoolQuery"]]] = None
    """过滤条件（不参与评分）"""

    minimum_should_match: Optional[int] = None
    """should 子句中至少需要匹配的数量"""


# =============================================================================
# 向量检索相关
# =============================================================================


class VectorSearchParam(BaseModel):
    """向量检索参数"""

    vector: List[float]
    """查询向量"""

    k: int = 10
    """返回的 Top-K 结果数量"""

    ef: Optional[int] = None
    """搜索时的 ef 参数（HNSW 索引），越大越精确但越慢"""

    metric_type: str = "L2"
    """距离度量类型：L2, IP, COSINE"""

    filter: Optional[Union[Dict[str, Any], BoolQuery]] = None
    """过滤条件（可选）"""


class HybridSearchParam(BaseModel):
    """混合检索参数（稠密 + 稀疏）"""

    vector_param: Optional[VectorSearchParam] = None
    """向量检索参数"""

    sparse_query: Optional[BoolQuery] = None
    """稀疏检索查询条件"""

    alpha: float = 0.5
    """稠密和稀疏结果的权重混合系数 (0-1)，0.5 表示等权重"""

    rerank_top_k: Optional[int] = None
    """混合后重新排序的 Top-K 数量"""


# =============================================================================
# 搜索查询和结果
# =============================================================================


class SearchQuery(BaseModel):
    """搜索查询参数

    支持多种查询类型：稀疏检索、稠密检索、混合检索
    """

    query: Optional[str] = None
    """文本查询（用于稀疏检索）"""

    bool_query: Optional[BoolQuery] = None
    """布尔查询（用于稀疏检索）"""

    vector_param: Optional[VectorSearchParam] = None
    """向量检索参数（用于稠密检索）"""

    hybrid_param: Optional[HybridSearchParam] = None
    """混合检索参数（用于混合检索）"""

    limit: int = 10
    """返回结果数量限制"""

    offset: int = 0
    """分页偏移量"""

    sort: Optional[List[Dict[str, str]]] = None
    """排序字段列表，例如 [{"field": "created_at", "order": "desc"}]"""

    fields: Optional[List[str]] = None
    """返回字段列表，None 表示返回所有字段"""

    include_vectors: bool = False
    """是否包含向量字段（稠密检索时）"""

    include_scores: bool = True
    """是否包含相关性分数"""

    partition_name: Optional[str] = None
    """分区名称（用于 Milvus 分区检索）"""

    @model_validator(mode="after")
    def validate_query_type(self) -> Self:
        """验证查询类型"""
        query_types = [
            self.query is not None,
            self.bool_query is not None,
            self.vector_param is not None,
            self.hybrid_param is not None,
        ]
        if sum(query_types) == 0:
            raise ValueError("At least one query type must be specified")
        if sum(query_types) > 1:
            raise ValueError("Only one query type can be specified at a time")
        return self


class SearchResult(BaseModel, Generic[T]):
    """搜索结果

    泛型类型 T 为文档对象类型
    """
    documents: List[T]
    """文档对象列表"""

    total: int
    """匹配的文档总数"""

    scores: Optional[List[float]] = None
    """相关性分数列表（与 documents 一一对应）"""

    has_more: bool = False
    """是否还有更多结果"""

    query_time_ms: float = 0.0
    """查询耗时（毫秒）"""

    @model_validator(mode="after")
    def validate_scores(self) -> Self:
        """验证分数列表"""
        if self.scores is not None and len(self.scores) != len(self.documents):
            raise ValueError("scores length must match documents length")
        return self


# =============================================================================
# 字段类型定义
# =============================================================================


class FieldType(str, Enum):
    """字段类型枚举"""

    keyword = "keyword"
    """关键字类型（精确匹配）"""

    text = "text"
    """文本类型（分词检索）"""

    integer = "integer"
    """整型"""

    long = "long"
    """长整型"""

    float = "float"
    """浮点型"""

    double = "double"
    """双精度浮点型"""

    boolean = "boolean"
    """布尔型"""

    date = "date"
    """日期类型"""

    datetime = "datetime"
    """日期时间类型"""

    dense_vector = "dense_vector"
    """稠密向量类型"""

    sparse_vector = "sparse_vector"
    """稀疏向量类型"""

    json = "json"
    """JSON 类型（动态字段）"""


class FieldDefinition(BaseModel):
    """字段定义

    用于定义索引 schema 中的字段
    """

    name: str
    """字段名"""

    type: FieldType
    """字段类型"""

    dimension: Optional[int] = None
    """向量维度（向量类型字段必需）"""

    metric_type: Optional[str] = None
    """距离度量类型（向量类型字段可选，默认 L2）"""

    index: bool = True
    """是否建立索引"""

    store: bool = True
    """是否存储原始值"""

    analyzer: Optional[str] = None
    """分词器（text 类型字段可选）"""

    description: Optional[str] = None
    """字段描述"""

    is_partition_key: bool = False
    """是否为分区键（仅 Milvus 支持，用于自动分区隔离）"""

    auto_id: bool = True
    """是否自动生成 ID（仅主键字段有效，默认为 True）"""

    @model_validator(mode="after")
    def validate_partition_key(self) -> Self:
        """验证分区键字段"""
        if self.is_partition_key:
            # Partition key 字段类型限制
            if self.type not in [FieldType.keyword, FieldType.text, FieldType.integer, FieldType.long]:
                raise ValueError(
                    f"Partition key field '{self.name}' must be of type keyword, text, integer, or long"
                )
            # Partition key 不能是主键
            if self.name == "id":
                raise ValueError(f"Partition key field '{self.name}' cannot be the primary key")
            # Partition key 不能有索引
            if self.index:
                raise ValueError(f"Partition key field '{self.name}' cannot have an index")
        return self

    @model_validator(mode="after")
    def validate_vector_field(self) -> Self:
        """验证向量字段"""
        if self.type in [FieldType.dense_vector, FieldType.sparse_vector]:
            if self.dimension is None:
                raise ValueError(f"Vector field '{self.name}' must specify dimension")
        return self


# =============================================================================
# 索引模型基类
# =============================================================================


class BaseIndexModel(BaseModel):
    """索引模型基类

    所有索引模型必须继承此类。
    提供字段定义、索引配置和 provider 交互的抽象。

    注意：
    - 此类是数据层的代理，实际的 schema 生成和 backend 交互由 Provider 完成
    - 子类需要实现 get_index_config() 和 get_field_definitions() 方法
    """

    class Config:
        arbitrary_types_allowed = True
        """允许任意类型（用于存储 provider 实例）"""

    _provider: Optional["BaseProvider"] = None
    """Provider 实例（内部使用）"""

    @classmethod
    @abstractmethod
    def get_index_name(cls) -> str:
        """获取索引名称

        Returns:
            索引名称（在 backend 中的唯一标识）
        """
        pass

    @classmethod
    @abstractmethod
    def get_index_type(cls) -> IndexingTypeEnum:
        """获取索引类型

        Returns:
            索引类型（sparse/dense/hybrid）
        """
        pass

    @classmethod
    @abstractmethod
    def get_field_definitions(cls) -> List[FieldDefinition]:
        """获取字段定义列表

        Returns:
            字段定义列表，用于生成索引 schema
        """
        pass

    @classmethod
    def get_index_config(cls) -> Dict[str, Any]:
        """获取索引配置

        Returns:
            索引配置字典，包含索引级别设置
        """
        return {}

    @classmethod
    def get_backend_type(cls) -> Optional[IndexingBackendTypeEnum]:
        """获取后端类型（可选）

        如果指定，则必须使用指定类型的后端
        否则可以使用任何兼容的后端

        Returns:
            后端类型，None 表示不限制
        """
        return None

    @classmethod
    def validate_backend_type(cls, backend_type: IndexingBackendTypeEnum) -> bool:
        """验证后端类型是否兼容

        Args:
            backend_type: 后端类型

        Returns:
            是否兼容
        """
        required_type = cls.get_backend_type()
        if required_type is None:
            return True
        return backend_type == required_type

    @classmethod
    async def create_index(cls, provider: "BaseProvider") -> bool:
        """创建索引

        Args:
            provider: Provider 实例

        Returns:
            是否成功创建
        """
        return await provider.create_index(cls)

    @classmethod
    async def drop_index(cls, provider: "BaseProvider") -> bool:
        """删除索引

        Args:
            provider: Provider 实例

        Returns:
            是否成功删除
        """
        return await provider.drop_index(cls)

    @classmethod
    async def index_exists(cls, provider: "BaseProvider") -> bool:
        """检查索引是否存在

        Args:
            provider: Provider 实例

        Returns:
            索引是否存在
        """
        return await provider.index_exists(cls)

    async def insert(self, provider: "BaseProvider") -> bool:
        """插入单条文档

        Args:
            provider: Provider 实例

        Returns:
            是否成功插入
        """
        return await provider.insert(self)

    async def update(self, provider: "BaseProvider", doc_id: Optional[str] = None) -> bool:
        """更新文档

        Args:
            provider: Provider 实例
            doc_id: 文档 ID，如果为 None 则尝试从模型中获取

        Returns:
            是否成功更新
        """
        return await provider.update(self, doc_id=doc_id)

    async def upsert(self, provider: "BaseProvider", doc_id: Optional[str] = None) -> bool:
        """插入或更新文档

        Args:
            provider: Provider 实例
            doc_id: 文档 ID，如果为 None 则尝试从模型中获取

        Returns:
            是否成功
        """
        return await provider.upsert(self, doc_id=doc_id)

    @classmethod
    async def bulk_insert(
        cls,
        provider: "BaseProvider",
        documents: List["BaseIndexModel"],
        batch_size: Optional[int] = None
    ) -> int:
        """批量插入文档

        Args:
            provider: Provider 实例
            documents: 文档列表
            batch_size: 批次大小，None 表示使用 provider 默认值

        Returns:
            成功插入的文档数量
        """
        return await provider.bulk_insert(cls, documents, batch_size=batch_size)

    @classmethod
    async def bulk_update(
        cls,
        provider: "BaseProvider",
        documents: List["BaseIndexModel"],
        batch_size: Optional[int] = None
    ) -> int:
        """批量更新文档

        Args:
            provider: Provider 实例
            documents: 文档列表
            batch_size: 批次大小，None 表示使用 provider 默认值

        Returns:
            成功更新的文档数量
        """
        return await provider.bulk_update(cls, documents, batch_size=batch_size)

    @classmethod
    async def bulk_upsert(
        cls,
        provider: "BaseProvider",
        documents: List["BaseIndexModel"],
        batch_size: Optional[int] = None
    ) -> int:
        """批量插入或更新文档

        Args:
            provider: Provider 实例
            documents: 文档列表
            batch_size: 批次大小，None 表示使用 provider 默认值

        Returns:
            成功的文档数量
        """
        return await provider.bulk_upsert(cls, documents, batch_size=batch_size)

    @classmethod
    async def delete(
        cls,
        provider: "BaseProvider",
        doc_id: Optional[str] = None,
        query: Optional[BoolQuery] = None
    ) -> int:
        """删除文档

        Args:
            provider: Provider 实例
            doc_id: 文档 ID（指定 doc_id 时，query 参数会被忽略）
            query: 删除条件（布尔查询）

        Returns:
            删除的文档数量
        """
        return await provider.delete(cls, doc_id=doc_id, query=query)

    @classmethod
    async def get_by_id(cls, provider: "BaseProvider", doc_id: str) -> Optional["BaseIndexModel"]:
        """根据 ID 获取文档

        Args:
            provider: Provider 实例
            doc_id: 文档 ID

        Returns:
            文档对象，不存在则返回 None
        """
        return await provider.get_by_id(cls, doc_id)

    @classmethod
    async def search(
        cls,
        provider: "BaseProvider",
        query: SearchQuery
    ) -> SearchResult["BaseIndexModel"]:
        """搜索文档

        Args:
            provider: Provider 实例
            query: 搜索查询参数

        Returns:
            搜索结果
        """
        return await provider.search(cls, query)

    @classmethod
    async def count(cls, provider: "BaseProvider", query: Optional[BoolQuery] = None) -> int:
        """统计文档数量

        Args:
            provider: Provider 实例
            query: 查询条件，None 表示统计所有文档

        Returns:
            文档数量
        """
        return await provider.count(cls, query=query)

    @classmethod
    async def flush(cls, provider: "BaseProvider"):
        """显式 flush 确保数据持久化

        Args:
            provider: Provider 实例
        """
        await provider.flush(cls)


class DenseIndexModel(BaseIndexModel):
    """稠密索引模型基类

    用于向量检索场景的索引模型。
    默认索引类型为 dense。
    """

    @classmethod
    def get_index_type(cls) -> IndexingTypeEnum:
        """获取索引类型"""
        return IndexingTypeEnum.dense


class SparseIndexModel(BaseIndexModel):
    """稀疏索引模型基类

    用于文本检索场景的索引模型。
    默认索引类型为 sparse。
    """

    @classmethod
    def get_index_type(cls) -> IndexingTypeEnum:
        """获取索引类型"""
        return IndexingTypeEnum.sparse


class HybridIndexModel(BaseIndexModel):
    """混合索引模型基类

    用于混合检索场景的索引模型。
    默认索引类型为 hybrid。
    """

    @classmethod
    def get_index_type(cls) -> IndexingTypeEnum:
        """获取索引类型"""
        return IndexingTypeEnum.hybrid


# =============================================================================
# Provider 基类
# =============================================================================


class BaseProvider(ABC):
    """Provider 抽象基类

    所有后端 provider 实现（Elasticsearch、Milvus等）必须继承此类。
    负责：
    - 连接管理
    - 索引操作（创建、删除、检查存在）
    - 数据持久化（增删改查）
    - Schema 生成
    - 查询执行

    注意：
    - Provider 直接操作后端服务
    - Provider 需要处理后端的原生错误，尽量保留原始错误信息
    - Provider 负责将后端返回的结果转换为 IndexModel 对象
    """

    def __init__(
        self,
        backend_type: IndexingBackendTypeEnum,
        config: Dict[str, Any]
    ):
        """
        初始化 Provider

        Args:
            backend_type: 后端类型
            config: 后端配置字典
        """
        self.backend_type = backend_type
        self.config = config

    @abstractmethod
    async def connect(self) -> None:
        """建立连接"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """断开连接"""
        pass

    @abstractmethod
    async def ping(self) -> bool:
        """检查连接是否正常"""
        pass

    @abstractmethod
    async def create_index(self, model_class: Type[BaseIndexModel]) -> bool:
        """
        创建索引

        Args:
            model_class: 索引模型类

        Returns:
            是否成功创建
        """
        pass

    @abstractmethod
    async def drop_index(self, model_class: Type[BaseIndexModel]) -> bool:
        """
        删除索引

        Args:
            model_class: 索引模型类

        Returns:
            是否成功删除
        """
        pass

    @abstractmethod
    async def index_exists(self, model_class: Type[BaseIndexModel]) -> bool:
        """
        检查索引是否存在

        Args:
            model_class: 索引模型类

        Returns:
            索引是否存在
        """
        pass

    @abstractmethod
    async def insert(self, document: BaseIndexModel) -> bool:
        """
        插入单条文档

        Args:
            document: 文档对象

        Returns:
            是否成功插入
        """
        pass

    @abstractmethod
    async def update(
        self,
        document: BaseIndexModel,
        doc_id: Optional[str] = None
    ) -> bool:
        """
        更新文档

        Args:
            document: 文档对象
            doc_id: 文档 ID

        Returns:
            是否成功更新
        """
        pass

    @abstractmethod
    async def upsert(
        self,
        document: BaseIndexModel,
        doc_id: Optional[str] = None
    ) -> bool:
        """
        插入或更新文档

        Args:
            document: 文档对象
            doc_id: 文档 ID

        Returns:
            是否成功
        """
        pass

    @abstractmethod
    async def bulk_insert(
        self,
        model_class: Type[BaseIndexModel],
        documents: List[BaseIndexModel],
        batch_size: Optional[int] = None
    ) -> int:
        """
        批量插入文档

        Args:
            model_class: 索引模型类
            documents: 文档列表
            batch_size: 批次大小

        Returns:
            成功插入的文档数量
        """
        pass

    @abstractmethod
    async def bulk_update(
        self,
        model_class: Type[BaseIndexModel],
        documents: List[BaseIndexModel],
        batch_size: Optional[int] = None
    ) -> int:
        """
        批量更新文档

        Args:
            model_class: 索引模型类
            documents: 文档列表
            batch_size: 批次大小

        Returns:
            成功更新的文档数量
        """
        pass

    @abstractmethod
    async def bulk_upsert(
        self,
        model_class: Type[BaseIndexModel],
        documents: List[BaseIndexModel],
        batch_size: Optional[int] = None
    ) -> int:
        """
        批量插入或更新文档

        Args:
            model_class: 索引模型类
            documents: 文档列表
            batch_size: 批次大小

        Returns:
            成功的文档数量
        """
        pass

    @abstractmethod
    async def delete(
        self,
        model_class: Type[BaseIndexModel],
        doc_id: Optional[str] = None,
        query: Optional[BoolQuery] = None
    ) -> int:
        """
        删除文档

        Args:
            model_class: 索引模型类
            doc_id: 文档 ID
            query: 删除条件

        Returns:
            删除的文档数量
        """
        pass

    @abstractmethod
    async def get_by_id(
        self,
        model_class: Type[BaseIndexModel],
        doc_id: str
    ) -> Optional[BaseIndexModel]:
        """
        根据 ID 获取文档

        Args:
            model_class: 索引模型类
            doc_id: 文档 ID

        Returns:
            文档对象，不存在则返回 None
        """
        pass

    @abstractmethod
    async def search(
        self,
        model_class: Type[BaseIndexModel],
        query: SearchQuery
    ) -> SearchResult[BaseIndexModel]:
        """
        搜索文档

        Args:
            model_class: 索引模型类
            query: 搜索查询参数

        Returns:
            搜索结果
        """
        pass

    @abstractmethod
    async def count(
        self,
        model_class: Type[BaseIndexModel],
        query: Optional[BoolQuery] = None
    ) -> int:
        """
        统计文档数量

        Args:
            model_class: 索引模型类
            query: 查询条件

        Returns:
            文档数量
        """
        pass

    async def flush(self, model_class: Type[BaseIndexModel]) -> bool:
        """
        手动刷新数据到磁盘（可选操作）

        某些后端（如 Milvus）支持手动刷新数据以确保持久化。
        大多数情况下，后端会自动持久化数据，不需要显式调用。
        仅在测试或需要立即确保数据持久化时使用。

        Args:
            model_class: 索引模型类

        Returns:
            是否成功刷新（默认返回 True）
        """
        # 默认实现：不执行任何操作
        return True

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.disconnect()
