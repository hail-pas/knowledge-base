"""
测试索引模块的 conftest.py

提供测试所需的 fixtures 和辅助函数
"""

import os
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel

from ext.indexing.base import (
    BaseIndexModel,
    DenseIndexModel,
    SparseIndexModel,
    FieldDefinition,
    FieldType,
    SearchQuery,
    VectorSearchParam,
)
from ext.ext_tortoise.enums import IndexingBackendTypeEnum, IndexingTypeEnum
from ext.indexing.factory import ProviderFactory
from ext.ext_tortoise.models.knowledge_base import IndexingBackendConfig

# =============================================================================
# 环境变量配置
# =============================================================================

# Elasticsearch 配置
ES_HOST = os.getenv("ES_HOST", "")
ES_PORT = int(os.getenv("ES_PORT", "9200"))
ES_USERNAME = os.getenv("ES_USERNAME", "")
ES_PASSWORD = os.getenv("ES_PASSWORD", "")
ES_SECURE = os.getenv("ES_SECURE", "false").lower() == "true"

# Milvus 配置
MILVUS_HOST = os.getenv("MILVUS_HOST", "")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_SECURE = os.getenv("MILVUS_SECURE", "false").lower() == "true"
MILVUS_USERNAME = os.getenv("MILVUS_USERNAME", "root")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "")
MILVUS_DATABASE = os.getenv("MILVUS_DATABASE", "default")

# =============================================================================
# Skip 标记
# =============================================================================

skip_if_no_es = pytest.mark.skipif(
    not ES_HOST,
    reason="ES_HOST not set, skipping Elasticsearch tests"
)

skip_if_no_milvus = pytest.mark.skipif(
    not MILVUS_HOST,
    reason="MILVUS_HOST not set, skipping Milvus tests"
)

# =============================================================================
# 测试数据生成函数
# =============================================================================

def generate_test_documents_sparse(count: int = 10) -> List["TestDocumentSparse"]:
    """生成稀疏索引测试文档"""
    docs = []
    categories = ["tech", "business", "sports", "entertainment"]
    authors = ["Alice", "Bob", "Charlie", "David"]

    for i in range(count):
        category = categories[i % len(categories)]
        author = authors[i % len(authors)]

        doc = TestDocumentSparse(
            id=f"doc_{i:04d}",
            title=f"文档标题 {i}",
            content=f"这是第 {i} 篇文档的内容，主题关于 {category}。",
            author=author,
            category=category,
            tags=[f"tag_{j}" for j in range(3)],
            created_at=datetime.now(),
            views=random.randint(0, 1000),
            published=i % 2 == 0,
        )
        docs.append(doc)

    return docs


def generate_test_documents_dense(count: int = 10, dimension: int = 1536) -> List["TestDocumentDense"]:
    """生成稠密索引测试文档"""
    docs = []

    for i in range(count):
        embedding = [random.random() for _ in range(dimension)]

        doc = TestDocumentDense(
            id=f"doc_{i:04d}",
            title=f"向量文档 {i}",
            content=f"这是第 {i} 篇向量文档的内容。",
            embedding=embedding,
            created_at=datetime.now(),
        )
        docs.append(doc)

    return docs


# =============================================================================
# 测试模型
# =============================================================================

class TestDocumentSparse(SparseIndexModel):
    """测试用的稀疏索引文档模型"""

    id: str
    title: str
    content: str
    author: str
    category: str
    tags: List[str]
    created_at: datetime
    views: int
    published: bool

    @classmethod
    def get_index_name(cls) -> str:
        return "test_documents_sparse"

    @classmethod
    def get_field_definitions(cls) -> List[FieldDefinition]:
        return [
            FieldDefinition(name="id", type=FieldType.keyword, description="文档ID"),
            FieldDefinition(name="title", type=FieldType.text, analyzer="standard", description="文档标题"),
            FieldDefinition(name="content", type=FieldType.text, analyzer="standard", description="文档内容"),
            FieldDefinition(name="author", type=FieldType.keyword, description="作者"),
            FieldDefinition(name="category", type=FieldType.keyword, description="分类"),
            FieldDefinition(name="tags", type=FieldType.keyword, description="标签"),
            FieldDefinition(name="created_at", type=FieldType.datetime, description="创建时间"),
            FieldDefinition(name="views", type=FieldType.integer, description="浏览次数"),
            FieldDefinition(name="published", type=FieldType.boolean, description="是否发布"),
        ]


class TestDocumentDense(DenseIndexModel):
    """测试用的稠密索引文档模型"""

    id: str
    title: str
    content: str
    embedding: List[float]
    created_at: datetime

    @classmethod
    def get_index_name(cls) -> str:
        return "test_documents_dense"

    @classmethod
    def get_field_definitions(cls) -> List[FieldDefinition]:
        return [
            FieldDefinition(name="id", type=FieldType.keyword, description="文档ID"),
            FieldDefinition(name="title", type=FieldType.text, analyzer="standard", description="文档标题"),
            FieldDefinition(name="content", type=FieldType.text, analyzer="standard", description="文档内容"),
            FieldDefinition(
                name="embedding",
                type=FieldType.dense_vector,
                dimension=1536,
                metric_type="COSINE",
                description="向量嵌入"
            ),
            FieldDefinition(name="created_at", type=FieldType.datetime, description="创建时间"),
        ]

    @classmethod
    def get_index_config(cls) -> dict:
        return {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        }


class TestDocumentWithExtras(DenseIndexModel):
    """测试支持 extras JSON 字段和 partition 的文档模型"""

    id: str
    title: str
    content: str
    embedding: List[float]
    partition: str
    extras: Optional[Dict[str, Any]] = None
    created_at: datetime

    @classmethod
    def get_index_name(cls) -> str:
        return "test_documents_with_extras"

    @classmethod
    def get_field_definitions(cls) -> List[FieldDefinition]:
        return [
            FieldDefinition(name="id", type=FieldType.keyword, description="文档ID"),
            FieldDefinition(name="title", type=FieldType.text, analyzer="standard", description="文档标题"),
            FieldDefinition(name="content", type=FieldType.text, analyzer="standard", description="文档内容"),
            FieldDefinition(
                name="embedding",
                type=FieldType.dense_vector,
                dimension=1536,
                metric_type="COSINE",
                description="向量嵌入"
            ),
            FieldDefinition(name="partition", type=FieldType.keyword, description="分区字段"),
            FieldDefinition(
                name="extras",
                type=FieldType.json,
                description="额外信息（JSON 类型，支持动态字段）"
            ),
            FieldDefinition(name="created_at", type=FieldType.datetime, description="创建时间"),
        ]

    @classmethod
    def get_index_config(cls) -> dict:
        return {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        }


# =============================================================================
# ES Provider Fixtures
# =============================================================================

@pytest.fixture
async def es_backend_config():
    """创建 Elasticsearch 后端配置"""
    config = IndexingBackendConfig(
        name="test_es",
        type=IndexingBackendTypeEnum.elasticsearch,
        host=ES_HOST,
        port=ES_PORT,
        username=ES_USERNAME if ES_USERNAME else None,
        password=ES_PASSWORD if ES_PASSWORD else None,
        secure=ES_SECURE,
        is_enabled=True,
        config={},
    )
    # 保存到数据库（临时对象）
    config._saved_in_db = False
    return config


@pytest.fixture
async def es_provider(es_backend_config):
    """创建 Elasticsearch Provider 实例"""
    if not ES_HOST:
        pytest.skip("ES_HOST not set")

    provider = await ProviderFactory.create(es_backend_config, use_cache=False)
    await provider.connect()
    yield provider

    # 清理测试索引
    try:
        await TestDocumentDense.drop_index(provider)
    except Exception:
        pass
    try:
        await TestDocumentSparse.drop_index(provider)
    except Exception:
        pass

    await provider.disconnect()


@pytest.fixture
async def es_dense_index(es_provider):
    """创建并初始化稠密索引"""
    if not ES_HOST:
        pytest.skip("ES_HOST not set")

    if await TestDocumentDense.index_exists(es_provider):
        await TestDocumentDense.drop_index(es_provider)

    await TestDocumentDense.create_index(es_provider)

    return es_provider


@pytest.fixture
async def es_sparse_index(es_provider):
    """创建并初始化稀疏索引"""
    if not ES_HOST:
        pytest.skip("ES_HOST not set")

    if await TestDocumentSparse.index_exists(es_provider):
        await TestDocumentSparse.drop_index(es_provider)

    await TestDocumentSparse.create_index(es_provider)

    return es_provider


# =============================================================================
# Milvus Provider Fixtures
# =============================================================================

@pytest.fixture
async def milvus_backend_config():
    """创建 Milvus 后端配置"""
    config = IndexingBackendConfig(
        name="test_milvus",
        type=IndexingBackendTypeEnum.milvus,
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        username=MILVUS_USERNAME if MILVUS_USERNAME else None,
        password=MILVUS_PASSWORD if MILVUS_PASSWORD else None,
        secure=MILVUS_SECURE,
        is_enabled=True,
        config={"db_name": MILVUS_DATABASE},
    )
    # 保存到数据库（临时对象）
    config._saved_in_db = False
    return config


@pytest.fixture
async def milvus_provider(milvus_backend_config):
    """创建 Milvus Provider 实例"""
    if not MILVUS_HOST:
        pytest.skip("MILVUS_HOST not set")

    provider = await ProviderFactory.create(milvus_backend_config, use_cache=False)
    await provider.connect()
    yield provider

    # 清理测试索引
    try:
        await TestDocumentDense.drop_index(provider)
    except Exception:
        pass

    await provider.disconnect()


@pytest.fixture
async def milvus_dense_index(milvus_provider):
    """创建并初始化稠密索引"""
    if not MILVUS_HOST:
        pytest.skip("MILVUS_HOST not set")

    if await TestDocumentDense.index_exists(milvus_provider):
        await TestDocumentDense.drop_index(milvus_provider)

    await TestDocumentDense.create_index(milvus_provider)

    return milvus_provider


@pytest.fixture
async def milvus_extras_index(milvus_provider):
    """创建并初始化支持 extras 和 partition 的索引"""
    if not MILVUS_HOST:
        pytest.skip("MILVUS_HOST not set")

    # 确保不存在
    if await TestDocumentWithExtras.index_exists(milvus_provider):
        await TestDocumentWithExtras.drop_index(milvus_provider)

    # 创建 collection
    await TestDocumentWithExtras.create_index(milvus_provider)

    yield milvus_provider

    # 清理
    try:
        await TestDocumentWithExtras.drop_index(milvus_provider)
    except Exception:
        pass
