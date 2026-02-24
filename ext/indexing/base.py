"""IndexModel 数据层定义"""

import asyncio
from datetime import datetime
from typing import Generic, Optional, TypeVar, Any, TYPE_CHECKING, Self, Union, get_origin, get_args, ClassVar
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field

from ext.ext_tortoise.models.knowledge_base import IndexingBackendConfig, EmbeddingModelConfig
from ext.indexing.types import FilterClause, SearchCursor, DenseSearchClause, SparseSearchClause, HybridSearchClause

if TYPE_CHECKING:
    from ext.indexing.base import BaseIndexModel
    from ext.ext_tortoise.models.knowledge_base import EmbeddingModelConfig

ExtraConfigT = TypeVar("ExtraConfigT", bound=BaseModel)


class BaseProvider(ABC, Generic[ExtraConfigT]):
    """Provider 抽象基类"""

    def __init__(self, config: IndexingBackendConfig, extra_config_type: type[ExtraConfigT]):
        self.config = config
        self.extra_config = extra_config_type.model_validate(config.extra_config or {})
        self._client = None

    def _validate_model_config(self, model_class: type["BaseIndexModel"]) -> None:
        """验证模型配置"""

        if (
            model_class.Meta.dense_vector_field
            and model_class.Meta.dense_vector_field not in model_class.__pydantic_fields__
        ):
            raise ValueError(
                f"Model {model_class.__name__} must have an '{model_class.Meta.dense_vector_field}' field when enable dense search"
            )

        if model_class.Meta.partition_key and model_class.Meta.partition_key not in model_class.__pydantic_fields__:
            raise ValueError(
                f"Model {model_class.__name__} must have an '{model_class.Meta.partition_key}' field when enable partition"
            )

        if model_class.Meta.dense_vector_field and (
            not model_class.Meta.dense_vector_dimension or model_class.Meta.dense_vector_dimension <= 0
        ):
            raise ValueError(
                f"Model {model_class.__name__} has invalid dimension: {model_class.Meta.dense_vector_dimension}",
            )

        if model_class.Meta.partition_key:
            partition_key_type = model_class.__pydantic_fields__.get(model_class.Meta.partition_key)
            if partition_key_type and partition_key_type.annotation != str:
                raise ValueError(f"Partition key '{model_class.Meta.partition_key}' must be of type str")

    @abstractmethod
    async def connect(self):
        """建立连接"""

    @abstractmethod
    async def disconnect(self):
        """断开连接"""

    @abstractmethod
    async def create_collection(self, model_class: type["BaseIndexModel"], drop_existing: bool = False):
        """创建索引集合/table"""

    @abstractmethod
    async def drop_collection(self, model_class: type["BaseIndexModel"]):
        """删除索引集合/table"""

    @abstractmethod
    async def get(self, model_class: type["BaseIndexModel"], ids: list) -> list[dict]:
        """获取单个文档"""

    @abstractmethod
    async def filter(
        self,
        model_class: type["BaseIndexModel"],
        filter_clause: FilterClause | None,
        limit: int = 10,
        offset: int = 0,
        sort: str | None = None,
    ) -> list[dict[str, Any]]:
        """过滤查询"""

    @abstractmethod
    async def insert(
        self,
        model_class: type["BaseIndexModel"],
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        """插入文档（支持批量）"""

    @abstractmethod
    async def update(self, model_class: type["BaseIndexModel"], documents: list[dict[str, Any]]) -> list:
        """更新文档"""

    @abstractmethod
    async def delete(self, model_class: type["BaseIndexModel"], ids: list[str]):
        """删除文档（根据 ID）"""

    @abstractmethod
    async def delete_by_query(self, model_class: type["BaseIndexModel"], filter_clause: FilterClause):
        """根据条件删除文档（支持 routing）"""

    @abstractmethod
    async def count(self, model_class: type["BaseIndexModel"], filter_clause: FilterClause | None) -> int:
        """统计文档数量"""

    @abstractmethod
    async def search(
        self,
        model_class: type["BaseIndexModel"],
        query_clause: DenseSearchClause | SparseSearchClause | HybridSearchClause,
        filter_clause: FilterClause | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[tuple[dict[str, Any], float]]:
        """搜索（返回 [(文档, 分数)]）"""

    @abstractmethod
    async def search_cursor(
        self,
        model_class: type["BaseIndexModel"],
        query_clause: DenseSearchClause | SparseSearchClause | HybridSearchClause,
        filter_clause: FilterClause | None = None,
        page_size: int = 100,
        cursor: str | None = None,
    ) -> tuple[list[tuple[dict[str, Any], float]], str | None]:
        """搜索（返回 [(文档, 分数)], next_cursor）"""

    @abstractmethod
    async def bulk_upsert(
        self,
        model_class: type["BaseIndexModel"],
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        """批量插入或更新（upsert）"""

    @abstractmethod
    async def health_check(self) -> bool:
        """健康检查"""

    def build_collection_name(self, model_class: type["BaseIndexModel"]) -> str:
        """按维度区分"""
        name = model_class.Meta.index_name

        if model_class.Meta.dense_vector_dimension:
            name = f"{name}_{model_class.Meta.dense_vector_dimension}"

        return name


T = TypeVar("T", bound="BaseIndexModel")


class BaseIndexModelMeta:
    """索引模型元数据基类（仅用于类型标注）

    注意：子类 Meta 不需要继承此类，
    BaseIndexModel.__init_subclass__ 会自动合并父类 Meta 属性
    """

    index_name: str
    dense_vector_field: str | None
    dense_vector_dimension: int | None
    partition_key: str | None
    auto_generate_id: bool
    provider: Optional["BaseProvider"]


class BaseIndexModel(BaseModel):
    """索引模型基类"""

    class Meta:
        """索引模型元数据配置

        子类可以覆盖以下字段：
        - index_name: 索引名称
        - dense_vector_field: 稠密向量字段名（默认 "dense_vector"）
        - dense_vector_dimension: 向量维度
        - partition_key: 分区键字段名
        - auto_generate_id: 是否自动生成 ID
        - provider: 索引提供者实例
        """

        index_name: str = "default_index"
        dense_vector_field: str | None = "dense_vector"
        dense_vector_dimension: int | None = None
        partition_key: str | None = None
        auto_generate_id: bool = True
        provider: Optional["BaseProvider"] = None

    def __init_subclass__(cls, **kwargs):
        """自动合并父类 Meta 属性到子类 Meta"""
        super().__init_subclass__(**kwargs)

        # 遍历父类 MRO，合并 Meta 属性
        # 优先级：子类 > 直接父类 > 间接父类
        for parent_cls in reversed(cls.__mro__[:-1]):  # 排除 object
            if hasattr(parent_cls, "Meta"):
                for attr_name, attr_value in parent_cls.Meta.__dict__.items():
                    # 跳过私有属性和已定义的属性
                    if not attr_name.startswith("_") and not hasattr(cls.Meta, attr_name):
                        setattr(cls.Meta, attr_name, attr_value)

    id: str | int = Field(default_factory=lambda: BaseIndexModel._get_id_default(), index_metadata={})  # type: ignore
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()

    @staticmethod
    def _extract_type(annotation):
        origin = get_origin(annotation)
        if origin is Union:
            args = get_args(annotation)
            return str if str in args else int if int in args else annotation
        return annotation

    @classmethod
    def _get_id_default(cls):
        id_field = cls.model_fields.get("id")
        if id_field and id_field.annotation:
            id_type = cls._extract_type(id_field.annotation)
            return "" if id_type == str else 0 if id_type == int else ""
        return ""

    @classmethod
    def get_provider(cls) -> "BaseProvider":
        """获取绑定的 provider 实例（从工厂获取）"""
        assert cls.Meta.provider, "Must specify provider"
        return cls.Meta.provider

    @classmethod
    async def create_schema(cls, drop_existing: bool = False):
        """创建索引 schema"""
        provider = cls.get_provider()
        await provider.create_collection(cls, drop_existing=drop_existing)

    @classmethod
    async def drop_schema(cls):
        """删除索引 schema"""
        provider = cls.get_provider()
        await provider.drop_collection(cls)

    @classmethod
    async def get(cls, ids: list) -> list[Self]:
        """通过 ID 获取文档"""
        assert isinstance(ids, list)
        provider = cls.get_provider()
        result = await provider.get(cls, ids=ids)
        return [cls.model_validate(i) for i in result]  # type: ignore

    @classmethod
    async def filter(
        cls,
        filter_clause: FilterClause | None = None,
        limit: int = 10,
        offset: int = 0,
        sort: str | None = None,
    ) -> list[Self]:
        """过滤查询"""
        provider = cls.get_provider()
        results = await provider.filter(cls, filter_clause=filter_clause, limit=limit, offset=offset, sort=sort)
        return [cls.model_validate(r) for r in results]  # type: ignore

    @classmethod
    async def search(
        cls,
        query_clause: DenseSearchClause | SparseSearchClause | HybridSearchClause,
        filter_clause: FilterClause | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[tuple[Self, float]]:
        """搜索（Provider 自动转换）"""
        provider = cls.get_provider()
        results = await provider.search(
            cls,
            query_clause=query_clause,
            filter_clause=filter_clause,
            limit=limit,
            offset=offset,
        )
        return [(cls.model_validate(r), score) for r, score in results]  # type: ignore

    @classmethod
    async def search_cursor(
        cls,
        query_clause: DenseSearchClause | SparseSearchClause | HybridSearchClause,
        filter_clause: FilterClause | None = None,
        page_size: int = 100,
        cursor: str | None = None,
    ) -> SearchCursor:
        """搜索（Cursor 方式）"""
        provider = cls.get_provider()
        results, next_cursor = await provider.search_cursor(
            cls,
            query_clause=query_clause,
            filter_clause=filter_clause,
            page_size=page_size,
            cursor=cursor,
        )

        converted_results = [(cls.model_validate(r), score) for r, score in results]

        return SearchCursor(results=converted_results, next_cursor=next_cursor)

    async def save(self):
        """保存/更新当前实例"""
        if self.__class__.Meta.auto_generate_id and self.id:
            self.id = self.__class__._get_id_default()

        if not self.__class__.Meta.auto_generate_id and not self.id:
            raise RuntimeError("Must specify id")

        self.updated_at = datetime.now()

        provider = self.get_provider()
        doc_data = self.model_dump(mode="json")
        if self.__class__.Meta.auto_generate_id:
            doc_data.pop("id", None)

        result = await provider.insert(self.__class__, documents=[doc_data])
        if result and len(result) > 0:
            self.id = result[0].get("id", self.id)

    @classmethod
    async def bulk_insert(cls, documents: list[Self], batch_size: int = 100, concurrent_batches: int = 5):
        """批量插入（自动并发）"""
        provider = cls.get_provider()

        batches = [documents[i : i + batch_size] for i in range(0, len(documents), batch_size)]

        # Use semaphore to limit concurrent batch operations
        semaphore = asyncio.Semaphore(concurrent_batches)

        async def insert_batch(batch: list[Self]):
            async with semaphore:
                batch_docs = []
                for doc in batch:
                    doc_data = doc.model_dump(mode="json")
                    if cls.Meta.auto_generate_id:
                        doc_data.pop("id", None)
                    batch_docs.append(doc_data)

                result = await provider.bulk_upsert(cls, documents=batch_docs)
                # Update each document with the actual id from the result
                if result and len(result) > 0:
                    for i, doc in enumerate(batch):
                        if i < len(result) and "id" in result[i]:
                            doc.id = result[i]["id"]

        await asyncio.gather(*[insert_batch(batch) for batch in batches])  # type: ignore

    @classmethod
    async def bulk_update(cls, documents: list[Self]):
        """批量更新"""
        assert all([isinstance(doc, cls) for doc in documents])

        provider = cls.get_provider()
        ids = await provider.update(cls, documents=[d.model_dump(mode="json") for d in documents])

        # milvus 会更新id
        for doc, id in zip(documents, ids):
            if id:
                doc.id = id

    @classmethod
    async def bulk_upsert(cls, documents: list[Self], batch_size: int = 100, concurrent_batches: int = 5):
        """批量插入或更新（upsert，自动并发）"""
        provider = cls.get_provider()

        batches = [documents[i : i + batch_size] for i in range(0, len(documents), batch_size)]

        # Use semaphore to limit concurrent batch operations
        semaphore = asyncio.Semaphore(concurrent_batches)

        async def upsert_batch(batch: list[Self]):
            async with semaphore:
                batch_docs = []
                for doc in batch:
                    doc_data = doc.model_dump(mode="json")
                    # 如果 auto_generate_id 为 True，只在 id 无效时移除（用于插入）
                    # id 有效则保留（用于更新）
                    if cls.Meta.auto_generate_id:
                        doc_id = doc_data.get("id")
                        if not doc_id:
                            doc_data.pop("id", None)
                    batch_docs.append(doc_data)

                result = await provider.bulk_upsert(cls, documents=batch_docs)
                # Update each document with the actual id from the result
                if result and len(result) > 0:
                    for i, doc in enumerate(batch):
                        if i < len(result) and "id" in result[i]:
                            doc.id = result[i]["id"]

        await asyncio.gather(*[upsert_batch(batch) for batch in batches])

    @classmethod
    async def bulk_delete(cls, ids: list[str]):
        """批量删除"""
        provider = cls.get_provider()
        await provider.delete(cls, ids=ids)

    @classmethod
    async def delete_by_query(cls, filter_clause: FilterClause):
        """根据条件删除"""
        provider = cls.get_provider()
        await provider.delete_by_query(cls, filter_clause=filter_clause)

    @classmethod
    async def count(cls, filter_clause: FilterClause | None = None) -> int:
        """统计文档数量"""
        provider = cls.get_provider()
        return await provider.count(cls, filter_clause=filter_clause)

    @classmethod
    async def exists(cls, id: str) -> bool:
        """检查文档是否存在"""
        return bool(await cls.get([id]))


class IndexModelFactory:
    """IndexModel 动态工厂

    根据 embedding 配置动态创建带正确维度的 IndexModel 类

    设计思路：
    1. 在 register.py 中定义基础 IndexModel 类并绑定 provider
    2. 运行时根据 embedding_config 的维度动态创建对应模型
    3. 动态模型复用基础模型的 provider 配置

    Example:
        # ext/register.py
        class ChunkIndex(BaseIndexModel):
            content: str
            class Meta:
                index_name = "chunks"
                provider = None  # 将在 register 时绑定

        async def register():
            provider = await IndexingProviderFactory.create(config)
            ChunkIndex.Meta.provider = provider

        # 运行时使用
        emb_config = await EmbeddingModelConfig.filter(name="model-1536").first()
        DynamicChunkIndex = IndexModelFactory.create_for_embedding(
            base_model=ChunkIndex,
            embedding_config=emb_config
        )
        # 现在 DynamicChunkIndex.Meta.dense_vector_dimension == 1536
        # collection name 会自动变为 "chunks_1536"
    """

    _model_registry: dict[str, type[BaseIndexModel]] = {}

    @classmethod
    def create_for_embedding(
        cls,
        base_model: type[BaseIndexModel],
        embedding_config: "EmbeddingModelConfig",
    ) -> type[BaseIndexModel]:
        """根据 embedding 配置动态创建 IndexModel 类

        Args:
            base_model: 基础 IndexModel 类（已在 register 时绑定 provider）
            embedding_config: embedding 模型配置（包含维度信息）

        Returns:
            动态创建的 IndexModel 子类，Meta.dense_vector_dimension 已设置

        Raises:
            ValueError: 如果 embedding_config.dimension 无效
            RuntimeError: 如果 base_model.Meta.provider 未绑定
        """

        if not isinstance(embedding_config, EmbeddingModelConfig):
            raise TypeError(f"embedding_config must be EmbeddingModelConfig, got {type(embedding_config)}")

        dimension = embedding_config.dimension

        if not dimension or dimension <= 0:
            raise ValueError(f"Invalid dimension: {dimension}")

        if not base_model.Meta.provider:
            raise RuntimeError(
                f"base_model.Meta.provider is not bound. "
                f"Please bind provider in register.py first: {base_model.__name__}.Meta.provider = await IndexingProviderFactory.create(...)"
            )

        model_key = cls._get_model_key(base_model.__name__, dimension)

        if model_key not in cls._model_registry:
            cls._model_registry[model_key] = cls._create_dynamic_class(base_model, dimension, model_key)

        return cls._model_registry[model_key]

    @classmethod
    def _get_model_key(cls, base_name: str, dimension: int) -> str:
        """生成唯一模型标识

        Args:
            base_name: 基础模型名称
            dimension: 向量维度

        Returns:
            唯一标识字符串，如 "ChunkIndex_1536"
        """
        return f"{base_name}_{dimension}"

    @classmethod
    def _create_dynamic_class(
        cls,
        base_model: type[BaseIndexModel],
        dimension: int,
        class_name: str,
    ) -> type[BaseIndexModel]:
        """动态创建 IndexModel 子类

        创建一个继承自 base_model 的新类，并覆盖 Meta.dense_vector_dimension

        Args:
            base_model: 基础模型类
            dimension: 向量维度
            class_name: 新类名称

        Returns:
            动态创建的 IndexModel 子类
        """

        class DynamicMeta(base_model.Meta):
            """动态 Meta 类

            继承 base_model.Meta 的所有配置，只覆盖 dense_vector_dimension
            """

            dense_vector_dimension = dimension

        namespace = {
            "Meta": DynamicMeta,
            "__module__": base_model.__module__,
            "__qualname__": f"{base_model.__qualname__}.{class_name}",
            "__annotations__": {"Meta": ClassVar},
        }

        dynamic_class = type(class_name, (base_model,), namespace)

        return dynamic_class

    @classmethod
    def clear_cache(cls) -> None:
        """清除缓存的动态模型类

        通常在测试或需要重新加载模型时使用

        Example:
            IndexModelFactory.clear_cache()
        """
        cls._model_registry.clear()

    @classmethod
    def get_registered_models(cls) -> list[str]:
        """获取已注册的动态模型列表

        Returns:
            模型 key 列表，如 ["ChunkIndex_1536", "ChunkIndex_3072"]

        Example:
            >>> models = IndexModelFactory.get_registered_models()
            >>> models
            ['ChunkIndex_1536', 'ChunkIndex_3072']
        """
        return list(cls._model_registry.keys())

    @classmethod
    def get_model(cls, base_name: str, dimension: int) -> type[BaseIndexModel] | None:
        """根据基础模型名和维度获取已注册的动态模型

        Args:
            base_name: 基础模型名称，如 "ChunkIndex"
            dimension: 向量维度

        Returns:
            动态模型类，如果不存在返回 None

        Example:
            >>> model_cls = IndexModelFactory.get_model("ChunkIndex", 1536)
            >>> if model_cls:
            ...     print(model_cls.Meta.dense_vector_dimension)
        """
        model_key = cls._get_model_key(base_name, dimension)
        return cls._model_registry.get(model_key)
