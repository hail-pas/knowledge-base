"""
Indexing Providers 包

包含不同后端服务的 Provider 实现（Elasticsearch、Milvus 等）
"""

# Provider 类将在具体实现模块中导入和注册到 ProviderFactory

# Elasticsearch Provider
from ext.indexing.providers.elasticsearch import ElasticsearchProvider

# Milvus Provider
from ext.indexing.providers.milvus import MilvusProvider

__all__ = [
    "ElasticsearchProvider",
    "MilvusProvider",
]
