"""测试知识库 Collection 管理 API"""

import pytest
from unittest.mock import AsyncMock

# 存储创建的资源 ID，用于后续测试
collection_id = None
external_collection_id = None
embedding_model_id = None


# =============================================================================
# EmbeddingModelConfig Setup (for Collection tests)
# =============================================================================


def test_create_embedding_model_for_collection(client):
    """测试创建 Embedding 模型配置（为 Collection 测试准备）"""
    global embedding_model_id
    response = client.post(
        "/v1/config/embedding-model",
        json={
            "name": "test-collection-embedding",
            "type": "openai",
            "model_name": "text-embedding-3-small",
            "api_key": "sk-embedding-test",
            "base_url": "https://api.openai.com/v1",
            "dimension": 1536,
            "max_chunk_length": 8192,
            "batch_size": 100,
            "extra_config": {},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    embedding_model_id = data.get("data", {}).get("id")


# =============================================================================
# Collection Tests
# =============================================================================


@pytest.fixture
async def bind_mock_providers():
    """Bind mock providers to index models for testing"""
    from ext.indexing.models import (
        DocumentContentDenseIndex,
        DocumentContentSparseIndex,
        DocumentGenerateFAQDenseIndex,
    )

    mock_provider = AsyncMock()
    mock_provider.bulk_upsert = AsyncMock(return_value=[{"id": f"mock-{i}"} for i in range(100)])
    mock_provider.delete = AsyncMock()
    mock_provider.disconnect = AsyncMock()

    DocumentContentDenseIndex.Meta.provider = mock_provider  # type: ignore
    DocumentContentSparseIndex.Meta.provider = mock_provider  # type: ignore
    DocumentGenerateFAQDenseIndex.Meta.provider = mock_provider  # type: ignore

    yield mock_provider

    DocumentContentDenseIndex.Meta.provider = None  # type: ignore
    DocumentContentSparseIndex.Meta.provider = None  # type: ignore
    DocumentGenerateFAQDenseIndex.Meta.provider = None  # type: ignore



def test_create_collection_basic(client):
    """测试创建基础 Collection"""
    global collection_id
    response = client.post(
        "/v1/collection",
        json={
            "name": "test-collection",
            "description": "测试知识库集合",
            "is_public": True,
            "is_temp": False,
            "is_external": False,
            "embedding_model_config_id": embedding_model_id,
            "workflow_template": {
                "parse_document": {
                    "input": {"document_id": 0},
                    "execute_params": {"task_name": "workflow_document.DocumentParseTask"},
                    "depends_on": [],
                },
                "chunk_document": {
                    "input": {"document_id": 0, "strategy": "auto"},
                    "execute_params": {"task_name": "workflow_document.DocumentChunkTask"},
                    "depends_on": ["parse_document"],
                },
            },
            "external_config": {},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    assert "data" in data
    assert data["data"]["name"] == "test-collection"
    assert data["data"]["description"] == "测试知识库集合"
    assert data["data"]["is_public"] is True
    collection_id = data.get("data", {}).get("id")


def test_create_collection_with_workflow_subset(client):
    """测试创建 Collection - 使用 workflow 部分活动"""
    response = client.post(
        "/v1/collection",
        json={
            "name": "test-collection-subset-workflow",
            "description": "测试使用部分workflow活动的集合",
            "is_public": False,
            "is_external": False,
            "embedding_model_config_id": embedding_model_id,
            "workflow_template": {
                "parse_document": {
                    "input": {"document_id": 0},
                    "execute_params": {"task_name": "workflow_document.DocumentParseTask"},
                    "depends_on": [],
                },
            },
            "external_config": {},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    assert data["data"]["name"] == "test-collection-subset-workflow"


def test_create_external_collection(client):
    """测试创建外部 Collection"""
    global external_collection_id
    response = client.post(
        "/v1/collection",
        json={
            "name": "test-external-collection",
            "description": "测试外部知识库",
            "is_public": False,
            "is_external": True,
            "external_config": {
                "endpoint": "https://external-kb.example.com",
                "api_key": "external-key-123",
            },
            "workflow_template": {},
            "embedding_model_config_id": None,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    assert data["data"]["is_external"] is True
    external_collection_id = data.get("data", {}).get("id")


def test_list_collections(client):
    """测试获取 Collection 列表"""
    response = client.get("/v1/collection")
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    assert "data" in data
    assert "items" in data["data"]
    assert len(data["data"]["items"]) >= 3


def test_list_collections_with_filters(client):
    """测试带过滤条件的 Collection 列表查询"""
    response = client.get(
        "/v1/collection",
        params={
            "name__icontains": "test",
            "is_public": True,
            "is_external": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    assert "data" in data
    items = data["data"]["items"]
    for item in items:
        assert "test" in item["name"].lower()
        assert item["is_public"] is True
        assert item["is_external"] is False


def test_get_collection_detail(client):
    """测试获取 Collection 详情"""
    global collection_id
    if not collection_id:
        pytest.skip("未创建 Collection")

    response = client.get(f"/v1/collection/{collection_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    assert data["data"]["id"] == collection_id
    assert data["data"]["name"] == "test-collection"
    assert "embedding_model_config" in data["data"]


def test_update_collection_basic(client):
    """测试更新 Collection 基本信息"""
    global collection_id
    if not collection_id:
        pytest.skip("未创建 Collection")

    response = client.put(
        f"/v1/collection/{collection_id}",
        json={
            "name": "updated-test-collection",
            "description": "更新后的描述",
            "is_public": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0

    # 验证更新成功
    response = client.get(f"/v1/collection/{collection_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["data"]["name"] == "updated-test-collection"
    assert data["data"]["description"] == "更新后的描述"
    assert data["data"]["is_public"] is False


def test_update_collection_workflow(client):
    """测试更新 Collection 的 workflow_template"""
    global collection_id
    if not collection_id:
        pytest.skip("未创建 Collection")

    response = client.put(
        f"/v1/collection/{collection_id}",
        json={
            "workflow_template": {
                "parse_document": {
                    "input": {"document_id": 0, "engine": "pymupdf"},
                    "execute_params": {"task_name": "workflow_document.DocumentParseTask"},
                    "depends_on": [],
                },
                "chunk_document": {
                    "input": {"document_id": 0, "strategy": "length"},
                    "execute_params": {"task_name": "workflow_document.DocumentChunkTask"},
                    "depends_on": ["parse_document"],
                },
                "index_chunks": {
                    "input": {"document_id": 0, "batch_size": 200},
                    "execute_params": {"task_name": "workflow_document.IndexChunkTask"},
                    "depends_on": ["chunk_document"],
                },
            },
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0


def test_update_external_collection_config(client):
    """测试更新外部 Collection 的 external_config"""
    global external_collection_id
    if not external_collection_id:
        pytest.skip("未创建外部 Collection")

    response = client.put(
        f"/v1/collection/{external_collection_id}",
        json={
            "description": "更新后的外部知识库",
            "external_config": {
                "endpoint": "https://new-endpoint.example.com",
                "api_key": "new-api-key-456",
                "timeout": 30,
            },
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0


def test_update_collection_is_temp(client):
    """测试更新 Collection 的临时标志"""
    global collection_id
    if not collection_id:
        pytest.skip("未创建 Collection")

    response = client.put(
        f"/v1/collection/{collection_id}",
        json={
            "is_temp": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0

    # 验证更新成功
    response = client.get(f"/v1/collection/{collection_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["data"]["is_temp"] is True


def test_delete_collection_with_no_documents(client, bind_mock_providers):
    """测试删除没有文档的 Collection"""
    # 先创建一个临时 Collection 用于删除测试
    response = client.post(
        "/v1/collection",
        json={
            "name": "test-collection-to-delete",
            "description": "待删除的测试集合",
            "is_public": False,
            "is_external": False,
            "embedding_model_config_id": embedding_model_id,
            "workflow_template": {
                "parse_document": {
                    "input": {"document_id": 0},
                    "execute_params": {"task_name": "workflow_document.DocumentParseTask"},
                    "depends_on": [],
                },
            },
            "external_config": {},
        },
    )
    assert response.status_code == 200
    data = response.json()
    temp_collection_id = data.get("data", {}).get("id")

    # 删除 Collection
    response = client.delete(f"/v1/collection/{temp_collection_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    assert data["data"]["deleted"] == 1

    # 验证已删除
    response = client.get(f"/v1/collection/{temp_collection_id}")
    assert response.status_code == 200
    data = response.json()
    # 应该返回对象不存在或空数据
    assert data["code"] != 0 or data.get("data") is None


def test_delete_external_collection(client):
    """测试删除外部 Collection"""
    # 先创建一个临时外部 Collection
    response = client.post(
        "/v1/collection",
        json={
            "name": "test-external-collection-to-delete",
            "description": "待删除的外部知识库",
            "is_public": False,
            "is_external": True,
            "external_config": {
                "endpoint": "https://external-test.example.com",
            },
            "workflow_template": {},
            "embedding_model_config_id": None,
        },
    )
    assert response.status_code == 200
    data = response.json()
    temp_external_id = data.get("data", {}).get("id")

    # 删除外部 Collection
    response = client.delete(f"/v1/collection/{temp_external_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    assert data["data"]["deleted"] == 1


def test_collection_search_by_name(client):
    """测试按名称搜索 Collection"""
    response = client.get(
        "/v1/collection",
        params={
            "name__icontains": "external",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    items = data["data"]["items"]
    # 至少应该包含我们创建的外部collection
    assert len(items) >= 1
    for item in items:
        assert "external" in item["name"].lower()


def test_collection_pagination(client):
    """测试 Collection 列表分页"""
    response = client.get(
        "/v1/collection",
        params={
            "page": 1,
            "page_size": 10,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    assert "data" in data
    assert "items" in data["data"]
    assert "total_count" in data["data"]["page_info"]
    assert "total_page" in data["data"]["page_info"]
    assert "page" in data["data"]["page_info"]
    assert "size" in data["data"]["page_info"]


def test_create_collection_minimal(client):
    """测试创建最小配置的 Collection"""
    response = client.post(
        "/v1/collection",
        json={
            "name": "minimal-collection",
            "is_external": False,
            "embedding_model_config_id": embedding_model_id,
            "workflow_template": {
                "parse_document": {
                    "input": {"document_id": 0},
                    "execute_params": {"task_name": "workflow_document.DocumentParseTask"},
                    "depends_on": [],
                },
            },
            "external_config": {},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    assert data["data"]["name"] == "minimal-collection"


def test_create_public_collection_with_empty_user_id(client):
    """测试创建公开 Collection 且 user_id 为空"""
    response = client.post(
        "/v1/collection",
        json={
            "name": "public-collection-no-user",
            "description": "公共知识库，无特定用户",
            "is_public": True,
            "user_id": None,
            "tenant_id": None,
            "role_id": None,
            "is_external": False,
            "embedding_model_config_id": embedding_model_id,
            "workflow_template": {
                "parse_document": {
                    "input": {"document_id": 0},
                    "execute_params": {"task_name": "workflow_document.DocumentParseTask"},
                    "depends_on": [],
                },
            },
            "external_config": {},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    assert data["data"]["is_public"] is True
