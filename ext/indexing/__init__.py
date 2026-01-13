"""
Indexing 模块 - RAG 系统索引抽象层

提供统一的索引管理接口，支持多种后端（Elasticsearch、Milvus等）。
"""

from ext.indexing.base import (
    BaseIndexModel,
    BaseProvider,
    SearchQuery,
    SearchResult,
    QueryCondition,
)
from ext.indexing.factory import ProviderFactory

__all__ = [
    "BaseIndexModel",
    "BaseProvider",
    "SearchQuery",
    "SearchResult",
    "QueryCondition",
    "ProviderFactory",
]
