"""查询条件类型定义（Provider 无感知）"""

from abc import ABC
from enum import StrEnum
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator
from dataclasses import dataclass


class QueryTypeEnum(StrEnum):
    dense = "dense"
    sparse = "sparse"
    hybrid = "hybrid"
    graph = "graph"  # TODO


class QueryClause(ABC):
    """查询条件基类（Provider 无感知）"""

    output_fields: list[str] = Field(default=["*"], description="返回字段")

    @field_validator("output_fields", mode="after")
    def model_validator(cls, value: list[str]) -> list[str]:
        if "*" in value:
            return ["*"]

        if "id" not in value:
            value.append("id")
        return value


class DenseSearchClause(BaseModel, QueryClause):
    """稠密向量搜索"""

    vector: list[float] = Field(description="查询向量")
    top_k: int = Field(default=10, ge=1, le=1000, description="返回结果数量")


class SparseSearchClause(BaseModel, QueryClause):
    """稀疏全文搜索"""

    query_text: str = Field(description="查询文本")
    field_name: str | None = Field(default=None, description="指定使用哪个字段的稀疏向量（如 'title', 'content'）")
    top_k: int = Field(default=10, ge=1, le=1000, description="返回结果数量")
    min_score: float = Field(default=0.0, description="最小分数")


class HybridSearchClause(BaseModel, QueryClause):
    """混合搜索（Dense + Sparse）"""

    dense: DenseSearchClause
    sparse: SparseSearchClause
    weight_dense: float = Field(default=0.7, ge=0.0, le=1.0, description="稠密搜索权重")
    weight_sparse: float = Field(default=0.3, ge=0.0, le=1.0, description="稀疏搜索权重")


class FilterClause(BaseModel, QueryClause):
    """过滤条件（支持：相等、范围、AND/OR）"""

    equals: dict[str, Any] | None = Field(default=None, description="相等条件：{'category': 'tech'}")
    in_list: dict[str, list[Any]] | None = Field(
        default=None,
        description="IN 条件：{'status': ['active', 'pending']}",
    )
    range: dict[str, dict[Literal["gte", "gt", "lt", "lte"], Any]] | None = Field(
        default=None,
        description="范围条件：{'created_at': {'gte': '2024-01-01'}}",
    )

    and_conditions: list["FilterClause"] | None = Field(default=None, description="AND 逻辑")
    or_conditions: list["FilterClause"] | None = Field(default=None, description="OR 逻辑")


@dataclass
class SearchCursor:
    """搜索游标（用于大数据量分页）"""

    results: list[tuple[Any, float]]
    next_cursor: str | None
    total: int | None = None
