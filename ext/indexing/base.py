"""IndexModel 数据层定义"""

import asyncio
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Self,
    Union,
    Generic,
    TypeVar,
    ClassVar,
    Optional,
    get_args,
    get_origin,
)
from datetime import UTC, datetime

from pydantic import Field, BaseModel

from ext.indexing.types import QueryClause, FilterClause, SearchCursor
from ext.ext_tortoise.models.knowledge_base import (
    EmbeddingModelConfig,
    IndexingBackendConfig,
)
from util.secret import decrypt_secret, transform_extra_config_secrets

if TYPE_CHECKING:
    from ext.indexing.base import BaseIndexModel
    from ext.ext_tortoise.models.knowledge_base import EmbeddingModelConfig

ExtraConfigT = TypeVar("ExtraConfigT", bound=BaseModel)


class BaseProvider(ABC, Generic[ExtraConfigT]):
    """Provider 抽象基类"""

    def __init__(self, config: IndexingBackendConfig, extra_config_type: type[ExtraConfigT]) -> None:
        self.config = config
        self.password = (
            decrypt_secret(config.password, purpose="indexing_backend.password")
            if config._saved_in_db
            else config.password
        )
        extra_config = (
            transform_extra_config_secrets(
                config.extra_config,
                purpose="indexing_backend.extra_config",
                decrypt=True,
            )
            if config._saved_in_db
            else config.extra_config or {}
        )
        self.extra_config = extra_config_type.model_validate(extra_config)
        self._client = None

    def _validate_model_config(self, model_class: type["BaseIndexModel"]) -> None:
        """验证模型配置"""

        if (
            model_class.Meta.dense_vector_field
            and model_class.Meta.dense_vector_field not in model_class.__pydantic_fields__
        ):
            raise ValueError(
                f"Model {model_class.__name__} must have "
                f"an '{model_class.Meta.dense_vector_field}' field when enable dense search",
            )

        if model_class.Meta.partition_key and model_class.Meta.partition_key not in model_class.__pydantic_fields__:
            raise ValueError(
                f"Model {model_class.__name__} must have "
                f"an '{model_class.Meta.partition_key}' field when enable partition",
            )

        if model_class.Meta.dense_vector_field and (
            not model_class.Meta.dense_vector_dimension or model_class.Meta.dense_vector_dimension <= 0
        ):
            raise ValueError(
                f"Model {model_class.__name__} has invalid dimension: {model_class.Meta.dense_vector_dimension}",
            )

        if model_class.Meta.partition_key:
            partition_key_type = model_class.__pydantic_fields__.get(model_class.Meta.partition_key)
            if partition_key_type and partition_key_type.annotation is not str:
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

        # Always give subclasses their own Meta object. Mutating an inherited
        # Meta leaks provider/dimension changes into sibling model classes.
        declared_meta = cls.__dict__.get("Meta")
        inherited = next(
            (parent.Meta for parent in cls.__mro__[1:] if hasattr(parent, "Meta")),
            BaseIndexModel.Meta,
        )
        if declared_meta is None:
            cls.Meta = type("Meta", (inherited,), {})
        elif not issubclass(declared_meta, inherited):
            overrides = {
                name: value
                for name, value in vars(declared_meta).items()
                if not name.startswith("__")
            }
            cls.Meta = type("Meta", (inherited,), overrides)

    id: str | int = Field(default_factory=lambda: BaseIndexModel._get_id_default(), index_metadata={})  # type: ignore
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        index_metadata={"server_incompatible": True},
    )  # type: ignore
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        index_metadata={"server_incompatible": True},
    )  # type: ignore

    @staticmethod
    def _extract_type(annotation):
        origin = get_origin(annotation)
        if origin is Union:
            args = get_args(annotation)
            return str if str in args else int if int in args else annotation
        return annotation

    @classmethod
    def _get_id_default(cls) -> str | int:
        id_field = cls.model_fields.get("id")
        if not id_field:
            return ""

        # 直接检查 annotation 是否就是 int 类型
        if id_field.annotation is int:
            return 0

        # 否则使用 _extract_type 处理 Union 类型
        id_type = cls._extract_type(id_field.annotation)
        if id_type is int:
            return 0

        return ""

    @classmethod
    def get_provider(cls) -> "BaseProvider":
        """获取绑定的 provider 实例（从工厂获取）"""
        assert cls.Meta.provider, "Must specify provider"
        return cls.Meta.provider

    @classmethod
    async def create_schema(cls, drop_existing: bool = False) -> None:
        """创建索引 schema"""
        provider = cls.get_provider()
        await provider.create_collection(cls, drop_existing=drop_existing)

    @classmethod
    async def drop_schema(cls) -> None:
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
        query_clause: QueryClause,
        filter_clause: FilterClause | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[tuple[Self, float]]:
        """Execute a backend-neutral retrieval strategy."""
        from ext.indexing.search import SearchStrategyFactory

        provider = cls.get_provider()
        results = await SearchStrategyFactory.execute(
            provider, cls, query_clause, filter_clause, limit, offset
        )
        return [(cls.model_validate(document), score) for document, score in results]

    @classmethod
    async def search_cursor(
        cls,
        query_clause: QueryClause,
        filter_clause: FilterClause | None = None,
        page_size: int = 100,
        cursor: str | None = None,
    ) -> SearchCursor:
        """Cursor pagination is handled by the strategy layer."""
        from ext.indexing.search import SearchStrategyFactory

        provider = cls.get_provider()
        results, next_cursor = await SearchStrategyFactory.execute_cursor(
            provider, cls, query_clause, filter_clause, page_size, cursor
        )
        converted = [(cls.model_validate(document), score) for document, score in results]
        return SearchCursor(results=converted, next_cursor=next_cursor)

    async def save(self) -> None:
        """Insert new documents, upsert documents that already have a stable ID."""
        cls = type(self)
        self.updated_at = datetime.now(UTC)
        payload = self.model_dump(mode="json")
        if not self.id:
            if not cls.Meta.auto_generate_id:
                raise ValueError("An ID is required when auto_generate_id is disabled")
            payload.pop("id", None)
            rows = await cls.get_provider().insert(cls, [payload])
        else:
            rows = await cls.get_provider().bulk_upsert(cls, [payload])
        if rows and rows[0].get("id") is not None:
            self.id = rows[0]["id"]

    @classmethod
    async def _write_batches(
        cls, documents: list[Self], *, upsert: bool,
        batch_size: int = 100, concurrent_batches: int = 5,
    ) -> None:
        """One bounded batch pipeline for insert and upsert operations."""
        if batch_size <= 0 or concurrent_batches <= 0:
            raise ValueError("batch_size and concurrent_batches must be positive")
        if not documents:
            return
        if any(not isinstance(doc, cls) for doc in documents):
            raise TypeError("Every document must match the index model")
        provider = cls.get_provider()
        semaphore = asyncio.Semaphore(concurrent_batches)

        async def run(batch: list[Self]) -> None:
            async with semaphore:
                payloads = []
                for doc in batch:
                    data = doc.model_dump(mode="json")
                    if not doc.id:
                        if not cls.Meta.auto_generate_id:
                            raise ValueError("An ID is required when auto_generate_id is disabled")
                        data.pop("id", None)
                    elif not upsert and cls.Meta.auto_generate_id:
                        data.pop("id", None)
                    payloads.append(data)
                if upsert:
                    rows = await provider.bulk_upsert(cls, payloads)
                else:
                    rows = await provider.insert(cls, payloads)
                if rows:
                    for document, row in zip(batch, rows, strict=False):
                        if row.get("id") is not None:
                            document.id = row["id"]

        await asyncio.gather(
            *(run(documents[i:i + batch_size]) for i in range(0, len(documents), batch_size))
        )

    @classmethod
    async def bulk_insert(
        cls, documents: list[Self], batch_size: int = 100, concurrent_batches: int = 5,
    ) -> None:
        await cls._write_batches(
            documents, upsert=False, batch_size=batch_size, concurrent_batches=concurrent_batches,
        )

    @classmethod
    async def bulk_update(cls, documents: list[Self]) -> None:
        if any(not isinstance(doc, cls) for doc in documents):
            raise TypeError("Every document must match the index model")
        if not documents:
            return
        ids = await cls.get_provider().update(
            cls, [document.model_dump(mode="json") for document in documents],
        )
        if ids:
            for document, new_id in zip(documents, ids, strict=False):
                if new_id is not None:
                    document.id = new_id

    @classmethod
    async def bulk_upsert(
        cls, documents: list[Self], batch_size: int = 100, concurrent_batches: int = 5,
    ) -> None:
        await cls._write_batches(
            documents, upsert=True, batch_size=batch_size, concurrent_batches=concurrent_batches,
        )

    @classmethod
    async def bulk_delete(cls, ids: list[str]) -> None:
        """批量删除"""
        provider = cls.get_provider()
        await provider.delete(cls, ids=ids)

    @classmethod
    async def delete_by_query(cls, filter_clause: FilterClause) -> None:
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
