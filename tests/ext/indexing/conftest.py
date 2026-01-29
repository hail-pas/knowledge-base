"""Indexing 模块的 conftest.py

定义测试所需的 fixtures
"""

import os
from datetime import datetime
import pytest
from pydantic import Field

from ext.indexing.base import BaseIndexModel
from ext.ext_tortoise.enums import IndexingBackendTypeEnum
from ext.ext_tortoise.models.knowledge_base import IndexingBackendConfig
from ext.indexing.types import DenseSearchClause, SparseSearchClause, HybridSearchClause, FilterClause


ES_HOST = os.getenv("ES_HOST")
ES_PORT = os.getenv("ES_PORT", "9200")
ES_USERNAME = os.getenv("ES_USERNAME")
ES_PASSWORD = os.getenv("ES_PASSWORD")
ES_SECURE = os.getenv("ES_SECURE", "true").lower() == "true"

MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_SECURE = os.getenv("MILVUS_SECURE", "false").lower() == "true"
MILVUS_USERNAME = os.getenv("MILVUS_USERNAME")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD")
MILVUS_DATABASE = os.getenv("MILVUS_DATABASE", "default")


@pytest.fixture
def elasticsearch_config_dict():
    """返回 Elasticsearch 配置字典"""
    return {
        "name": "test-elasticsearch",
        "type": IndexingBackendTypeEnum.elasticsearch,
        "host": ES_HOST or "localhost",
        "port": int(ES_PORT) if ES_PORT else 9200,
        "username": ES_USERNAME,
        "password": ES_PASSWORD,
        "use_ssl": ES_SECURE,
        "verify_ssl": ES_SECURE,
        "timeout": 30,
        "max_retries": 3,
        "is_enabled": True,
        "is_default": True,
        "description": "测试用 Elasticsearch 配置",
        "extra_config": {
            "number_of_shards": 3,
            "number_of_replicas": 2,
            "vector_similarity": "cosine",
            "text_analyzer": "standard",
        },
    }


@pytest.fixture
async def elasticsearch_config(elasticsearch_config_dict):
    """创建并保存 Elasticsearch 配置到数据库"""
    await IndexingBackendConfig.filter(name=elasticsearch_config_dict["name"]).delete()
    config = await IndexingBackendConfig.create(**elasticsearch_config_dict)
    return config


@pytest.fixture
def milvus_config_dict():
    """返回 Milvus 配置字典"""
    return {
        "name": "test-milvus",
        "type": IndexingBackendTypeEnum.milvus,
        "host": MILVUS_HOST or "localhost",
        "port": int(MILVUS_PORT) if MILVUS_PORT else 19530,
        "username": MILVUS_USERNAME,
        "password": MILVUS_PASSWORD,
        "use_ssl": MILVUS_SECURE,
        "verify_ssl": MILVUS_SECURE,
        "timeout": 30,
        "max_retries": 3,
        "is_enabled": True,
        "is_default": True,
        "description": "测试用 Milvus 配置",
        "extra_config": {
            "db_name": MILVUS_DATABASE or "default",
            "index_type": "HNSW",
            "metric_type": "IP",
            "M": 16,
            "ef_construction": 64,
        },
    }


@pytest.fixture
async def milvus_config(milvus_config_dict):
    """创建并保存 Milvus 配置到数据库"""
    await IndexingBackendConfig.filter(name=milvus_config_dict["name"]).delete()
    config = await IndexingBackendConfig.create(**milvus_config_dict)
    return config


@pytest.fixture
def sample_query_vector():
    """示例查询向量（维度1536）"""
    return [0.1, 0.2, 0.3] + [0.0] * 1533


@pytest.fixture
def sample_documents():
    """示例文档数据"""
    return [
        {
            "id": "doc1",
            "title": "Machine Learning Basics",
            "content": "Introduction to machine learning algorithms",
            "category": "tech",
            "embedding": [0.1, 0.2, 0.3] + [0.0] * 1533,
        },
        {
            "id": "doc2",
            "title": "Python Programming",
            "content": "Learn Python from scratch",
            "category": "programming",
            "embedding": [0.4, 0.5, 0.6] + [0.0] * 1533,
        },
        {
            "id": "doc3",
            "title": "Data Science Guide",
            "content": "Complete guide to data science",
            "category": "tech",
            "embedding": [0.7, 0.8, 0.9] + [0.0] * 1533,
        },
    ]


@pytest.fixture
def sample_filter_clauses():
    """示例过滤条件"""
    return {
        "simple_equals": FilterClause(equals={"category": "tech"}),
        "in_list": FilterClause(in_list={"status": ["active", "pending"]}),
        "range": FilterClause(range={"created_at": {"gte": "2024-01-01"}}),  # type: ignore
        "complex_and": FilterClause(
            equals={"category": "tech"},
            range={"created_at": {"gte": "2024-01-01"}},  # type: ignore
        ),
        "complex_or": FilterClause(
            or_conditions=[
                FilterClause(equals={"category": "tech"}),
                FilterClause(equals={"category": "programming"}),
            ]
        ),
    }


@pytest.fixture
def sample_search_clauses():
    """示例搜索条件"""
    return {
        "dense": DenseSearchClause(vector=[0.1, 0.2, 0.3] + [0.0] * 1533, top_k=10),
        "sparse": SparseSearchClause(query_text="machine learning", top_k=10, min_score=0.5),
        "hybrid": HybridSearchClause(
            dense=DenseSearchClause(vector=[0.1, 0.2, 0.3] + [0.0] * 1533, top_k=10),
            sparse=SparseSearchClause(query_text="machine learning", top_k=10, min_score=0.5),
            weight_dense=0.7,
            weight_sparse=0.3,
        ),
    }


@pytest.fixture
def test_index_model():
    """测试用索引模型类"""

    class TestIndexModel(BaseIndexModel):
        title: str = Field(index_metadata={"enable_keyword": True}) # type: ignore
        content: str
        category: list[str]
        embedding: list[float]
        created_at: datetime = datetime.now()
        updated_at: datetime = datetime.now()

        class Meta:  # type: ignore
            index_name: str = "test_es_indexing"
            dense_vector_field: str = "embedding"
            dense_vector_dimension: int = 1536

    return TestIndexModel


@pytest.fixture
def test_index_model_with_partition():
    """测试用索引模型类（带 partition key）"""

    class TestIndexModelWithPartition(BaseIndexModel):
        tenant_id: str
        title: str = Field(index_metadata={"enable_keyword": True}) # type: ignore
        content: str
        category: list[str]
        embedding: list[float]
        created_at: datetime = datetime.now()
        updated_at: datetime = datetime.now()

        class Meta:  # type: ignore
            index_name: str = "test_es_indexing"
            dense_vector_field: str = "embedding"
            dense_vector_dimension: int = 1536
            partition_key: str = "tenant_id"

    return TestIndexModelWithPartition
