"""
Milvus Partition 和 JSON 字段测试

测试 Milvus Provider 的分区和 JSON 字段功能：
- Partition 字段支持
- JSON 字段（extras）的插入和检索
- JSON 字段的动态属性查询（如：extras["special_category"] == "special"）
- JSON 字段的范围查询
- JSON 字段的多值查询（terms）
- 同时使用 partition 和 JSON 字段查询
- JSON 字段支持多种数据类型
"""

import random
from datetime import datetime

import pytest

from ext.indexing.base import (
    SearchQuery,
    VectorSearchParam,
    QueryCondition,
    BoolQuery,
)
from tests.indexing.conftest import (
    skip_if_no_milvus,
    TestDocumentWithExtras,
)


# =============================================================================
# Partition 字段测试
# =============================================================================

@skip_if_no_milvus
async def test_partition_support(milvus_extras_index, milvus_provider):
    """测试 partition 字段支持"""
    docs = []
    for i in range(10):
        embedding = [random.random() for _ in range(1536)]
        doc = TestDocumentWithExtras(
            id=f"partition_test_{i}",
            title=f"分区测试文档 {i}",
            content=f"这是第 {i} 个测试文档",
            embedding=embedding,
            partition="partition_1" if i < 5 else "partition_2",
            extras=None,
            created_at=datetime.now(),
        )
        docs.append(doc)

    # 批量插入
    count = await TestDocumentWithExtras.bulk_insert(milvus_provider, docs)
    assert count == 10

    # 在 partition_1 中检索
    query_vector = docs[0].embedding
    vector_param = VectorSearchParam(
        vector=query_vector,
        k=5,
        metric_type="COSINE"
    )
    query = SearchQuery(
        vector_param=vector_param,
        partition_name="partition_1"
    )
    result = await TestDocumentWithExtras.search(milvus_provider, query)

    # 验证结果都在 partition_1 中
    assert len(result.documents) > 0
    for doc in result.documents:
        assert doc.partition == "partition_1"


@skip_if_no_milvus
async def test_partition_search_with_different_partitions(milvus_extras_index, milvus_provider):
    """测试在不同分区中搜索"""
    docs = []
    partitions = ["partition_1", "partition_2", "partition_3"]

    for i, partition in enumerate(partitions):
        embedding = [random.random() for _ in range(1536)]
        doc = TestDocumentWithExtras(
            id=f"multi_partition_{i}",
            title=f"多分区测试 {i}",
            content="测试内容",
            embedding=embedding,
            partition=partition,
            extras=None,
            created_at=datetime.now(),
        )
        docs.append(doc)

    # 批量插入
    count = await TestDocumentWithExtras.bulk_insert(milvus_provider, docs)
    assert count == 3

    # 在每个分区中分别检索
    for partition in partitions:
        query = SearchQuery(
            vector_param=VectorSearchParam(
                vector=docs[0].embedding,
                k=10,
                metric_type="COSINE"
            ),
            partition_name=partition
        )
        result = await TestDocumentWithExtras.search(milvus_provider, query)

        # 验证结果只在指定分区中
        if result.documents:
            for doc in result.documents:
                assert doc.partition == partition


# =============================================================================
# JSON 字段插入和检索测试
# =============================================================================

@skip_if_no_milvus
async def test_json_field_insert(milvus_extras_index, milvus_provider):
    """测试 JSON 字段的插入和检索"""
    embedding = [random.random() for _ in range(1536)]

    # 创建带 JSON 字段的文档
    doc = TestDocumentWithExtras(
        id="json_test_001",
        title="JSON 字段测试",
        content="测试 JSON 字段功能",
        embedding=embedding,
        partition="partition_1",
        extras={
            "special_category": "special",
            "priority": 5,
            "tags": ["important", "test"],
            "metadata": {
                "source": "manual",
                "verified": True
            }
        },
        created_at=datetime.now(),
    )

    # 插入
    result = await doc.insert(milvus_provider)
    assert result is True

    # 通过 ID 检索
    retrieved = await TestDocumentWithExtras.get_by_id(milvus_provider, "json_test_001")
    assert retrieved is not None
    assert retrieved.extras is not None
    assert retrieved.extras.get("special_category") == "special"
    assert retrieved.extras.get("priority") == 5


@skip_if_no_milvus
async def test_json_field_with_different_types(milvus_extras_index, milvus_provider):
    """测试 JSON 字段支持多种数据类型"""
    embedding = [random.random() for _ in range(1536)]

    doc = TestDocumentWithExtras(
        id="json_types_001",
        title="JSON 类型测试",
        content="测试 JSON 字段支持多种类型",
        embedding=embedding,
        partition="partition_1",
        extras={
            "string_field": "hello",
            "number_field": 42,
            "float_field": 3.14,
            "boolean_field": True,
            "null_field": None,
            "array_field": [1, 2, 3],
            "nested_object": {
                "level1": "value1",
                "level2": {
                    "level3": "value3"
                }
            }
        },
        created_at=datetime.now(),
    )

    # 插入
    result = await doc.insert(milvus_provider)
    assert result is True

    # 检索并验证
    retrieved = await TestDocumentWithExtras.get_by_id(milvus_provider, "json_types_001")
    assert retrieved is not None
    assert retrieved.extras is not None
    assert retrieved.extras.get("string_field") == "hello"
    assert retrieved.extras.get("number_field") == 42
    assert retrieved.extras.get("float_field") == 3.14
    assert retrieved.extras.get("boolean_field") is True
    assert retrieved.extras.get("null_field") is None
    assert isinstance(retrieved.extras.get("array_field"), list)
    assert retrieved.extras.get("nested_object").get("level1") == "value1"


# =============================================================================
# JSON 字段动态属性查询测试
# =============================================================================

@skip_if_no_milvus
async def test_json_field_dynamic_query(milvus_extras_index, milvus_provider):
    """测试 JSON 字段动态属性查询"""
    docs = []
    for i in range(5):
        embedding = [random.random() for _ in range(1536)]
        doc = TestDocumentWithExtras(
            id=f"json_query_{i}",
            title=f"JSON 查询测试 {i}",
            content="测试内容",
            embedding=embedding,
            partition="partition_1",
            extras={
                "special_category": "special" if i < 3 else "normal",
                "priority": i,
            },
            created_at=datetime.now(),
        )
        docs.append(doc)

    # 批量插入
    count = await TestDocumentWithExtras.bulk_insert(milvus_provider, docs)
    assert count == 5

    # 查询 extras["special_category"] == "special"
    bool_query = BoolQuery(
        filter=[
            QueryCondition(field='extras["special_category"]', value="special", match_type="term")
        ]
    )

    query_vector = docs[0].embedding
    vector_param = VectorSearchParam(
        vector=query_vector,
        k=10,
        metric_type="COSINE",
        filter=bool_query
    )

    query = SearchQuery(vector_param=vector_param)
    result = await TestDocumentWithExtras.search(milvus_provider, query)

    # 验证结果中所有文档的 special_category 都是 "special"
    assert len(result.documents) > 0
    for doc in result.documents:
        assert doc.extras.get("special_category") == "special"


# =============================================================================
# JSON 字段范围查询测试
# =============================================================================

@skip_if_no_milvus
async def test_json_field_range_query(milvus_extras_index, milvus_provider):
    """测试 JSON 字段范围查询"""
    docs = []
    for i in range(10):
        embedding = [random.random() for _ in range(1536)]
        doc = TestDocumentWithExtras(
            id=f"json_range_{i}",
            title=f"JSON 范围查询测试 {i}",
            content="测试内容",
            embedding=embedding,
            partition="partition_1",
            extras={
                "priority": i,
                "special_category": "test",
            },
            created_at=datetime.now(),
        )
        docs.append(doc)

    # 批量插入
    count = await TestDocumentWithExtras.bulk_insert(milvus_provider, docs)
    assert count == 10

    # 查询 extras["priority"] >= 5
    bool_query = BoolQuery(
        filter=[
            QueryCondition(field='extras["priority"]', range_gte=5, match_type="range")
        ]
    )

    query_vector = docs[0].embedding
    vector_param = VectorSearchParam(
        vector=query_vector,
        k=10,
        metric_type="COSINE",
        filter=bool_query
    )

    query = SearchQuery(vector_param=vector_param)
    result = await TestDocumentWithExtras.search(milvus_provider, query)

    # 验证结果中所有文档的 priority 都 >= 5
    assert len(result.documents) > 0
    for doc in result.documents:
        assert doc.extras.get("priority") >= 5


@skip_if_no_milvus
async def test_json_field_range_query_with_both_sides(milvus_extras_index, milvus_provider):
    """测试 JSON 字段双边范围查询"""
    docs = []
    for i in range(20):
        embedding = [random.random() for _ in range(1536)]
        doc = TestDocumentWithExtras(
            id=f"json_range_both_{i}",
            title=f"双边范围查询测试 {i}",
            content="测试内容",
            embedding=embedding,
            partition="partition_1",
            extras={
                "score": i * 10,
            },
            created_at=datetime.now(),
        )
        docs.append(doc)

    # 批量插入
    count = await TestDocumentWithExtras.bulk_insert(milvus_provider, docs)
    assert count == 20

    # 查询 extras["score"] >= 50 and extras["score"] <= 100
    bool_query = BoolQuery(
        filter=[
            QueryCondition(
                field='extras["score"]',
                range_gte=50,
                range_lte=100,
                match_type="range"
            )
        ]
    )

    query_vector = docs[0].embedding
    vector_param = VectorSearchParam(
        vector=query_vector,
        k=20,
        metric_type="COSINE",
        filter=bool_query
    )

    query = SearchQuery(vector_param=vector_param)
    result = await TestDocumentWithExtras.search(milvus_provider, query)

    # 验证结果中所有文档的 score 都在 [50, 100] 范围内
    assert len(result.documents) > 0
    for doc in result.documents:
        score = doc.extras.get("score")
        assert score >= 50
        assert score <= 100


# =============================================================================
# JSON 字段多值查询（terms）测试
# =============================================================================

@skip_if_no_milvus
async def test_json_field_terms_query(milvus_extras_index, milvus_provider):
    """测试 JSON 字段多值查询（terms）"""
    docs = []
    categories = ["cat1", "cat2", "cat3", "cat4", "cat5"]
    for i, category in enumerate(categories):
        embedding = [random.random() for _ in range(1536)]
        doc = TestDocumentWithExtras(
            id=f"json_terms_{i}",
            title=f"JSON terms 查询测试 {i}",
            content="测试内容",
            embedding=embedding,
            partition="partition_1",
            extras={
                "special_category": category,
            },
            created_at=datetime.now(),
        )
        docs.append(doc)

    # 批量插入
    count = await TestDocumentWithExtras.bulk_insert(milvus_provider, docs)
    assert count == 5

    # 查询 extras["special_category"] in ["cat1", "cat3", "cat5"]
    bool_query = BoolQuery(
        filter=[
            QueryCondition(field='extras["special_category"]', values=["cat1", "cat3", "cat5"], match_type="terms")
        ]
    )

    query_vector = docs[0].embedding
    vector_param = VectorSearchParam(
        vector=query_vector,
        k=10,
        metric_type="COSINE",
        filter=bool_query
    )

    query = SearchQuery(vector_param=vector_param)
    result = await TestDocumentWithExtras.search(milvus_provider, query)

    # 验证结果中的 special_category 在指定列表中
    assert len(result.documents) > 0
    for doc in result.documents:
        assert doc.extras.get("special_category") in ["cat1", "cat3", "cat5"]


# =============================================================================
# 同时使用 Partition 和 JSON 字段查询测试
# =============================================================================

@skip_if_no_milvus
async def test_partition_with_json_query(milvus_extras_index, milvus_provider):
    """测试同时使用 partition 和 JSON 字段查询"""
    docs = []
    for i in range(15):
        embedding = [random.random() for _ in range(1536)]
        partition = "partition_1" if i < 5 else ("partition_2" if i < 10 else "partition_3")
        category = "special" if i % 3 == 0 else "normal"
        doc = TestDocumentWithExtras(
            id=f"combined_test_{i}",
            title=f"组合测试 {i}",
            content="测试内容",
            embedding=embedding,
            partition=partition,
            extras={
                "special_category": category,
                "priority": i,
            },
            created_at=datetime.now(),
        )
        docs.append(doc)

    # 批量插入
    count = await TestDocumentWithExtras.bulk_insert(milvus_provider, docs)
    assert count == 15

    # 在 partition_2 中查询 special_category == "special"
    bool_query = BoolQuery(
        filter=[
            QueryCondition(field='extras["special_category"]', value="special", match_type="term")
        ]
    )

    query_vector = docs[0].embedding
    vector_param = VectorSearchParam(
        vector=query_vector,
        k=10,
        metric_type="COSINE",
        filter=bool_query
    )

    query = SearchQuery(
        vector_param=vector_param,
        partition_name="partition_2"
    )
    result = await TestDocumentWithExtras.search(milvus_provider, query)

    # 验证结果都在 partition_2 且 special_category == "special"
    assert len(result.documents) > 0
    for doc in result.documents:
        assert doc.partition == "partition_2"
        assert doc.extras.get("special_category") == "special"


@skip_if_no_milvus
async def test_partition_with_json_range_query(milvus_extras_index, milvus_provider):
    """测试同时使用 partition 和 JSON 范围查询"""
    docs = []
    for i in range(20):
        embedding = [random.random() for _ in range(1536)]
        partition = "partition_1" if i < 10 else "partition_2"
        doc = TestDocumentWithExtras(
            id=f"partition_range_test_{i}",
            title=f"分区范围查询测试 {i}",
            content="测试内容",
            embedding=embedding,
            partition=partition,
            extras={
                "score": i * 5,
            },
            created_at=datetime.now(),
        )
        docs.append(doc)

    # 批量插入
    count = await TestDocumentWithExtras.bulk_insert(milvus_provider, docs)
    assert count == 20

    # 在 partition_1 中查询 score >= 30
    bool_query = BoolQuery(
        filter=[
            QueryCondition(
                field='extras["score"]',
                range_gte=30,
                match_type="range"
            )
        ]
    )

    query_vector = docs[0].embedding
    vector_param = VectorSearchParam(
        vector=query_vector,
        k=20,
        metric_type="COSINE",
        filter=bool_query
    )

    query = SearchQuery(
        vector_param=vector_param,
        partition_name="partition_1"
    )
    result = await TestDocumentWithExtras.search(milvus_provider, query)

    # 验证结果都在 partition_1 且 score >= 30
    assert len(result.documents) > 0
    for doc in result.documents:
        assert doc.partition == "partition_1"
        assert doc.extras.get("score") >= 30


# =============================================================================
# 复杂查询测试
# =============================================================================

@skip_if_no_milvus
async def test_complex_json_query_with_multiple_conditions(milvus_extras_index, milvus_provider):
    """测试复杂的 JSON 查询（多个条件组合）"""
    docs = []
    for i in range(30):
        embedding = [random.random() for _ in range(1536)]
        partition = "partition_1" if i < 10 else ("partition_2" if i < 20 else "partition_3")
        category = "important" if i % 3 == 0 else "normal"
        status = "active" if i % 2 == 0 else "inactive"
        doc = TestDocumentWithExtras(
            id=f"complex_test_{i}",
            title=f"复杂查询测试 {i}",
            content="测试内容",
            embedding=embedding,
            partition=partition,
            extras={
                "special_category": category,
                "status": status,
                "priority": i % 10,
            },
            created_at=datetime.now(),
        )
        docs.append(doc)

    # 批量插入
    count = await TestDocumentWithExtras.bulk_insert(milvus_provider, docs)
    assert count == 30

    # 查询：partition_2 中，special_category == "important" 且 priority >= 5
    bool_query = BoolQuery(
        filter=[
            QueryCondition(field='extras["special_category"]', value="important", match_type="term"),
            QueryCondition(
                field='extras["priority"]',
                range_gte=5,
                match_type="range"
            )
        ]
    )

    query_vector = docs[0].embedding
    vector_param = VectorSearchParam(
        vector=query_vector,
        k=30,
        metric_type="COSINE",
        filter=bool_query
    )

    query = SearchQuery(
        vector_param=vector_param,
        partition_name="partition_2"
    )
    result = await TestDocumentWithExtras.search(milvus_provider, query)

    # 验证所有结果都满足条件
    assert len(result.documents) > 0
    for doc in result.documents:
        assert doc.partition == "partition_2"
        assert doc.extras.get("special_category") == "important"
        assert doc.extras.get("priority") >= 5


@skip_if_no_milvus
async def test_json_nested_field_query(milvus_extras_index, milvus_provider):
    """测试 JSON 嵌套字段查询"""
    docs = []
    for i in range(5):
        embedding = [random.random() for _ in range(1536)]
        doc = TestDocumentWithExtras(
            id=f"nested_test_{i}",
            title=f"嵌套查询测试 {i}",
            content="测试内容",
            embedding=embedding,
            partition="partition_1",
            extras={
                "metadata": {
                    "level1": f"value_{i}",
                    "level2": {
                        "level3": f"deep_value_{i}"
                    }
                }
            },
            created_at=datetime.now(),
        )
        docs.append(doc)

    # 批量插入
    count = await TestDocumentWithExtras.bulk_insert(milvus_provider, docs)
    assert count == 5

    # 查询嵌套字段（注意：Milvus JSON 查询语法）
    query = SearchQuery(
        vector_param=VectorSearchParam(
            vector=docs[0].embedding,
            k=10,
            metric_type="COSINE"
        ),
        partition_name="partition_1"
    )
    result = await TestDocumentWithExtras.search(milvus_provider, query)

    # 验证结果
    assert len(result.documents) > 0
    for doc in result.documents:
        assert doc.extras is not None
        assert "metadata" in doc.extras
