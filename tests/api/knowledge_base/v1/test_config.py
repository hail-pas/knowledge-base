"""测试知识库配置管理 API"""

import pytest

# 存储创建的资源 ID，用于后续测试
file_source_id = None
llm_model_id = None
indexing_backend_id = None
embedding_model_id = None


# =============================================================================
# FileSource Tests
# =============================================================================


def test_create_file_source_local(client):
    """测试创建本地文件源"""
    global file_source_id
    response = client.post(
        "/v1/config/file-source",
        json={
            "name": "test-local-files",
            "type": "local_file",
            "storage_location": "/tmp/test-files",
            "description": "测试本地文件源",
            "extra_config": {},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    file_source_id = data.get("data", {}).get("id")


def test_create_file_source_s3(client):
    """测试创建 S3 文件源"""
    response = client.post(
        "/v1/config/file-source",
        json={
            "name": "test-s3",
            "type": "s3",
            "access_key": "test-access-key",
            "secret_key": "test-secret-key",
            "storage_location": "test-bucket",
            "endpoint": "https://s3.amazonaws.com",
            "region": "us-east-1",
            "extra_config": {
                "signature_version": "s3v4",
            },
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0


def test_list_file_sources(client):
    """测试获取文件源列表"""
    response = client.get("/v1/config/file-source")
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    assert "data" in data
    assert "items" in data["data"]


def test_get_file_source_detail(client):
    """测试获取文件源详情"""
    global file_source_id
    if not file_source_id:
        pytest.skip("未创建文件源")
    response = client.get(f"/v1/config/file-source/{file_source_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0


# =============================================================================
# LLMModelConfig Tests
# =============================================================================


def test_create_llm_model_openai(client):
    """测试创建 OpenAI LLM 模型配置"""
    global llm_model_id
    response = client.post(
        "/v1/config/llm-model",
        json={
            "name": "test-openai",
            "type": "openai",
            "model_name": "gpt-4",
            "api_key": "sk-test-key",
            "base_url": "https://api.openai.com/v1",
            "max_tokens": 4096,
            "extra_config": {
                "organization": "test-org",
            },
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    llm_model_id = data.get("data", {}).get("id")


def test_create_llm_model_deepseek(client):
    """测试创建 DeepSeek LLM 模型配置"""
    response = client.post(
        "/v1/config/llm-model",
        json={
            "name": "test-deepseek",
            "type": "deepseek",
            "model_name": "deepseek-chat",
            "api_key": "sk-deepseek-test",
            "base_url": "https://api.deepseek.com/v1",
            "extra_config": {},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0


def test_list_llm_models(client):
    """测试获取 LLM 模型配置列表"""
    response = client.get("/v1/config/llm-model")
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    assert "data" in data
    assert "items" in data["data"]


def test_get_llm_model_detail(client):
    """测试获取 LLM 模型配置详情"""
    global llm_model_id
    if not llm_model_id:
        pytest.skip("未创建 LLM 模型配置")
    response = client.get(f"/v1/config/llm-model/{llm_model_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0


# =============================================================================
# IndexingBackendConfig Tests
# =============================================================================


def test_create_indexing_backend_elasticsearch(client):
    """测试创建 Elasticsearch 索引后端配置"""
    global indexing_backend_id
    response = client.post(
        "/v1/config/indexing-backend",
        json={
            "name": "test-es",
            "type": "elasticsearch",
            "host": "localhost",
            "port": 9200,
            "username": "elastic",
            "password": "password",
            "extra_config": {
                "number_of_shards": 3,
                "number_of_replicas": 2,
            },
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    indexing_backend_id = data.get("data", {}).get("id")


def test_create_indexing_backend_milvus(client):
    """测试创建 Milvus 索引后端配置"""
    response = client.post(
        "/v1/config/indexing-backend",
        json={
            "name": "test-milvus",
            "type": "milvus",
            "host": "localhost",
            "port": 19530,
            "username": "root",
            "password": "milvus",
            "extra_config": {
                "index_type": "HNSW",
                "metric_type": "IP",
            },
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0


def test_list_indexing_backends(client):
    """测试获取索引后端配置列表"""
    response = client.get("/v1/config/indexing-backend")
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    assert "data" in data
    assert "items" in data["data"]


def test_get_indexing_backend_detail(client):
    """测试获取索引后端配置详情"""
    global indexing_backend_id
    if not indexing_backend_id:
        pytest.skip("未创建索引后端配置")
    response = client.get(f"/v1/config/indexing-backend/{indexing_backend_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0


# =============================================================================
# EmbeddingModelConfig Tests
# =============================================================================


def test_create_embedding_model_openai(client):
    """测试创建 OpenAI Embedding 模型配置"""
    global embedding_model_id
    response = client.post(
        "/v1/config/embedding-model",
        json={
            "name": "test-openai-embedding",
            "type": "openai",
            "model_name": "text-embedding-3-small",
            "api_key": "sk-embedding-test",
            "base_url": "https://api.openai.com/v1",
            "dimension": 1536,
            "max_chunk_length": 8192,
            "extra_config": {
                "encoding_format": "float",
            },
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    embedding_model_id = data.get("data", {}).get("id")


def test_list_embedding_models(client):
    """测试获取 Embedding 模型配置列表"""
    response = client.get("/v1/config/embedding-model")
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    assert "data" in data
    assert "items" in data["data"]


def test_get_embedding_model_detail(client):
    """测试获取 Embedding 模型配置详情"""
    global embedding_model_id
    if not embedding_model_id:
        pytest.skip("未创建 Embedding 模型配置")
    response = client.get(f"/v1/config/embedding-model/{embedding_model_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
