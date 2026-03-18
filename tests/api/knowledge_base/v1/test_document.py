"""测试 Document 管理 API"""
import time
import pytest
from io import BytesIO
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch


# =============================================================================
# Setup: Embedding Model, File Source, Collection
# =============================================================================


@pytest.fixture
def embedding_model_setup(client, request):
    """测试创建 Embedding 模型配置、File Source 和 Collection（为 Document 测试准备）"""
    import uuid

    test_id = str(uuid.uuid4())[:8]

    response = client.post(
        "/v1/config/embedding-model",
        json={
            "name": f"test-document-embedding-{test_id}",
            "type": "openai",
            "model_name": "text-embedding-3-small",
            "api_key": "sk-doc-test",
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
    embedding_model_id = data["data"]["id"]

    # 创建 file_source
    response = client.post(
        "/v1/config/file-source",
        json={
            "name": f"test-document-file-source-{test_id}",
            "type": "local_file",
            "storage_location": f"/tmp/test_documents_{test_id}",
            "is_enabled": True,
            "is_default": False,
            "extra_config": {},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    file_source_id = data["data"]["id"]

    # 创建 collection
    response = client.post(
        "/v1/collection",
        json={
            "name": f"test-document-collection-{test_id}",
            "description": "测试文档集合",
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
            "external_config": {
                "endpoint": "https://example.com",
                "authorization": "Bearer test-token",
                "collection_id": "test-collection-id",
            },
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    collection_id = data["data"]["id"]

    return {
        "embedding_model_id": embedding_model_id,
        "file_source_id": file_source_id,
        "collection_id": collection_id,
    }



@pytest.fixture
async def bind_mock_providers():
    """Bind mock providers to index models for testing"""
    from ext.indexing.models import (
        DocumentContentDenseIndex,
        DocumentContentSparseIndex,
        DocumentFAQDenseIndex,
    )

    mock_provider = AsyncMock()
    mock_provider.bulk_upsert = AsyncMock(return_value=[{"id": f"mock-{i}"} for i in range(100)])
    mock_provider.delete = AsyncMock()
    mock_provider.disconnect = AsyncMock()

    DocumentContentDenseIndex.Meta.provider = mock_provider  # type: ignore
    DocumentContentSparseIndex.Meta.provider = mock_provider  # type: ignore
    DocumentFAQDenseIndex.Meta.provider = mock_provider  # type: ignore

    yield mock_provider

    DocumentContentDenseIndex.Meta.provider = None  # type: ignore
    DocumentContentSparseIndex.Meta.provider = None  # type: ignore
    DocumentFAQDenseIndex.Meta.provider = None  # type: ignore


@pytest.fixture(autouse=True)
def mock_process_document():
    """Mock document workflow scheduling in API tests."""
    with patch(
        "api.knowledge_base.v1.document.process_document",
        new=AsyncMock(return_value="00000000-0000-0000-0000-000000000001"),
    ):
        yield


# =============================================================================
# Document Creation Tests
# =============================================================================


@pytest.fixture
def mock_file_source_upload():
    """Mock file source upload operations"""
    from ext.file_source.base import FileMetadata

    async def mock_upload(uri, content, content_type=None):
        return FileMetadata(
            uri=uri,
            file_name="test.pdf",
            file_size=len(content),
            last_modified=datetime.now(UTC),
            etag="test-etag-123",
            content_type=content_type,
        )

    provider_mock = AsyncMock()
    provider_mock.upload_file = mock_upload
    provider_mock.delete_file = AsyncMock(return_value=True)

    with patch("ext.file_source.FileSourceFactory.create", return_value=provider_mock):
        yield provider_mock


def test_create_document_by_upload(client, mock_file_source_upload, embedding_model_setup):
    """测试通过文件上传创建文档"""
    collection_id = embedding_model_setup["collection_id"]
    file_source_id = embedding_model_setup["file_source_id"]

    file_content = b"Test PDF content for document creation"
    files = {
        "file": ("test.pdf", BytesIO(file_content), "application/pdf"),
    }
    data = {
        "collection_id": collection_id,
        "file_source_id": file_source_id,
        "display_name": "测试文档",
    }

    response = client.post("/v1/document", files=files, data=data)
    assert response.status_code == 200
    result = response.json()
    assert result["code"] == 0
    assert "data" in result
    assert result["data"]["file_name"] == "test.pdf"
    assert result["data"]["display_name"] == "测试文档"
    assert result["data"]["status"] == "pending"
    assert result["data"]["extension"] == ".pdf"


def test_create_document_by_upload_without_display_name(client, mock_file_source_upload, embedding_model_setup):
    """测试通过文件上传创建文档（不指定 display_name）"""
    collection_id = embedding_model_setup["collection_id"]
    file_source_id = embedding_model_setup["file_source_id"]

    file_content = b"Another test PDF content"
    files = {
        "file": ("sample.pdf", BytesIO(file_content), "application/pdf"),
    }
    data = {
        "collection_id": collection_id,
        "file_source_id": file_source_id,
    }

    response = client.post("/v1/document", files=files, data=data)
    assert response.status_code == 200
    result = response.json()
    assert result["code"] == 0
    assert result["data"]["display_name"] == "sample.pdf"  # 使用原始文件名


def test_create_document_by_upload_txt_file(client, mock_file_source_upload, embedding_model_setup):
    """测试上传 TXT 文件创建文档"""
    collection_id = embedding_model_setup["collection_id"]
    file_source_id = embedding_model_setup["file_source_id"]

    file_content = b"This is a plain text document"
    files = {
        "file": ("test.txt", BytesIO(file_content), "text/plain"),
    }
    data = {
        "collection_id": collection_id,
        "file_source_id": file_source_id,
    }

    response = client.post("/v1/document", files=files, data=data)
    assert response.status_code == 200
    result = response.json()
    assert result["code"] == 0
    assert result["data"]["extension"] == ".txt"
    assert result["data"]["file_name"] == "test.txt"


def test_create_document_by_upload_with_extension(client, mock_file_source_upload, embedding_model_setup):
    """测试上传带有扩展名的文件"""
    collection_id = embedding_model_setup["collection_id"]
    file_source_id = embedding_model_setup["file_source_id"]

    file_content = b"Markdown content"
    files = {
        "file": ("README.md", BytesIO(file_content), "text/markdown"),
    }
    data = {
        "collection_id": collection_id,
        "file_source_id": file_source_id,
    }

    response = client.post("/v1/document", files=files, data=data)
    assert response.status_code == 200
    result = response.json()
    assert result["code"] == 0
    assert result["data"]["extension"] == ".md"


@pytest.fixture
def mock_file_source_metadata():
    """Mock file source metadata retrieval"""
    from ext.file_source.base import FileMetadata

    provider_mock = AsyncMock()
    provider_mock.get_file_metadata = AsyncMock(
        return_value=FileMetadata(
            uri="/path/to/document.pdf",
            file_name="remote-document.pdf",
            file_size=1024000,
            last_modified=datetime.now(UTC),
            etag="remote-etag-456",
            content_type="application/pdf",
            extra={"custom_field": "custom_value"},
        ),
    )

    with patch("ext.file_source.FileSourceFactory.create", return_value=provider_mock):
        yield provider_mock


def test_create_document_by_uri_non_http(client, mock_file_source_metadata, embedding_model_setup):
    """测试通过 URI（非 HTTP）创建文档"""
    collection_id = embedding_model_setup["collection_id"]
    file_source_id = embedding_model_setup["file_source_id"]

    data = {
        "collection_id": collection_id,
        "file_source_id": file_source_id,
        "uri": "/path/to/document.pdf",
        "display_name": "远程文档",
    }

    response = client.post("/v1/document", data=data)
    assert response.status_code == 200
    result = response.json()
    assert result["code"] == 0
    assert result["data"]["uri"] == "/path/to/document.pdf"
    assert result["data"]["display_name"] == "远程文档"
    assert result["data"]["file_size"] == 1024000
    assert result["data"]["source_version_key"] == "remote-etag-456"
    assert result["data"]["source_meta"]["custom_field"] == "custom_value"


def test_create_document_by_http_url(client, embedding_model_setup):
    """测试通过 HTTP URL 创建文档（跳过 metadata）"""
    collection_id = embedding_model_setup["collection_id"]
    file_source_id = embedding_model_setup["file_source_id"]

    data = {
        "collection_id": collection_id,
        "file_source_id": file_source_id,
        "uri": "https://example.com/document.pdf",
    }

    timestamp = int(time.time())

    response = client.post("/v1/document", data=data)
    assert response.status_code == 200
    result = response.json()
    assert result["code"] == 0
    assert result["data"]["uri"] == "https://example.com/document.pdf"
    assert result["data"]["file_size"] is None  # HTTP URL 跳过 metadata
    assert int(result["data"]["source_version_key"]) >= timestamp


def test_create_document_by_http_url_with_display_name(client, embedding_model_setup):
    """测试通过 HTTP URL 创建文档并指定 display_name"""
    collection_id = embedding_model_setup["collection_id"]
    file_source_id = embedding_model_setup["file_source_id"]

    data = {
        "collection_id": collection_id,
        "file_source_id": file_source_id,
        "uri": "https://example.com/report.pdf",
        "display_name": "远程报告",
    }

    response = client.post("/v1/document", data=data)
    assert response.status_code == 200
    result = response.json()
    assert result["code"] == 0
    assert result["data"]["display_name"] == "远程报告"


# =============================================================================
# Validation Tests
# =============================================================================


def test_create_document_missing_both_file_and_uri(client, embedding_model_setup):
    """测试创建文档时同时缺少 file 和 uri"""
    collection_id = embedding_model_setup["collection_id"]
    file_source_id = embedding_model_setup["file_source_id"]

    data = {
        "collection_id": collection_id,
        "file_source_id": file_source_id,
    }

    response = client.post("/v1/document", data=data)
    assert response.status_code == 200
    result = response.json()
    assert result["code"] != 0  # 应该返回错误
    assert "必须提供 file 或 uri" in result["message"]


def test_create_document_both_file_and_uri(client, embedding_model_setup):
    """测试创建文档时同时提供 file 和 uri（应该失败）"""
    collection_id = embedding_model_setup["collection_id"]
    file_source_id = embedding_model_setup["file_source_id"]

    file_content = b"Test content"
    files = {
        "file": ("test.pdf", BytesIO(file_content), "application/pdf"),
    }
    data = {
        "collection_id": collection_id,
        "file_source_id": file_source_id,
        "uri": "/path/to/file.pdf",
    }

    response = client.post("/v1/document", files=files, data=data)
    assert response.status_code == 200
    result = response.json()
    assert result["code"] != 0  # 应该返回错误


def test_create_document_nonexistent_collection(client, embedding_model_setup):
    """测试创建文档时指定不存在的 collection"""
    file_source_id = embedding_model_setup["file_source_id"]

    file_content = b"Test content"
    files = {
        "file": ("test.pdf", BytesIO(file_content), "application/pdf"),
    }
    data = {
        "collection_id": 999999,
        "file_source_id": file_source_id,
    }

    response = client.post("/v1/document", files=files, data=data)
    assert response.status_code == 200
    result = response.json()
    assert result["code"] != 0
    assert "Collection不存在" in result["message"]


def test_create_document_nonexistent_file_source(client, embedding_model_setup):
    """测试创建文档时指定不存在的 file_source"""
    collection_id = embedding_model_setup["collection_id"]

    data = {
        "collection_id": collection_id,
        "file_source_id": 999999,
        "uri": "/path/to/file.pdf",
    }

    response = client.post("/v1/document", data=data)
    assert response.status_code == 200
    result = response.json()
    assert result["code"] != 0
    assert "文件源不存在" in result["message"]


# =============================================================================
# Document List and Detail Tests
# =============================================================================


def test_list_documents(client):
    """测试获取文档列表"""
    response = client.get("/v1/document")
    assert response.status_code == 200
    result = response.json()
    assert result["code"] == 0
    assert "data" in result
    assert "items" in result["data"]
    assert len(result["data"]["items"]) > 0


def test_list_documents_with_collection_filter(client, embedding_model_setup):
    """测试按 collection_id 过滤文档列表"""
    collection_id = embedding_model_setup["collection_id"]

    response = client.get("/v1/document", params={"collection_id": collection_id})
    assert response.status_code == 200
    result = response.json()
    assert result["code"] == 0
    items = result["data"]["items"]
    for item in items:
        assert item["collection_id"] == collection_id


def test_list_documents_with_status_filter(client):
    """测试按 status 过滤文档列表"""
    response = client.get("/v1/document", params={"status": "pending"})
    assert response.status_code == 200
    result = response.json()
    assert result["code"] == 0
    items = result["data"]["items"]
    for item in items:
        assert item["status"] == "pending"


def test_list_documents_with_search(client):
    """测试搜索文档列表"""
    response = client.get("/v1/document", params={"search": "test"})
    assert response.status_code == 200
    result = response.json()
    assert result["code"] == 0


def test_list_documents_with_pagination(client):
    """测试文档列表分页"""
    response = client.get(
        "/v1/document",
        params={
            "page": 1,
            "page_size": 10,
        },
    )
    assert response.status_code == 200
    result = response.json()
    assert result["code"] == 0
    assert "page_info" in result["data"]
    page_info = result["data"]["page_info"]
    assert "total_count" in page_info
    assert "total_page" in page_info
    assert "page" in page_info
    assert "size" in page_info


def test_get_document_detail(client, mock_file_source_upload, embedding_model_setup):
    """测试获取文档详情"""
    collection_id = embedding_model_setup["collection_id"]
    file_source_id = embedding_model_setup["file_source_id"]

    # 先创建一个测试文档
    file_content = b"Test PDF content for detail"
    files = {
        "file": ("test_detail.pdf", BytesIO(file_content), "application/pdf"),
    }
    data = {
        "collection_id": collection_id,
        "file_source_id": file_source_id,
        "display_name": "详情测试文档",
    }

    response = client.post("/v1/document", files=files, data=data)
    assert response.status_code == 200
    result = response.json()
    assert result["code"] == 0
    document_id = result["data"]["id"]

    response = client.get(f"/v1/document/{document_id}")
    assert response.status_code == 200
    result = response.json()
    assert result["code"] == 0
    assert result["data"]["id"] == document_id
    assert result["data"]["file_name"] == "test_detail.pdf"


def test_get_document_detail_nonexistent(client):
    """测试获取不存在的文档详情"""
    response = client.get("/v1/document/999999")
    assert response.status_code == 200
    result = response.json()
    assert result["code"] != 0


# =============================================================================
# Document Delete Tests
# =============================================================================


@pytest.mark.asyncio
async def test_delete_document_success_status(client, embedding_model_setup, bind_mock_providers):
    """测试删除状态为 success 的文档"""
    from ext.ext_tortoise.models.knowledge_base import Document
    from ext.ext_tortoise.enums import DocumentStatusEnum

    collection_id = embedding_model_setup["collection_id"]
    file_source_id = embedding_model_setup["file_source_id"]

    # 创建一个临时文档用于删除测试
    file_content = b"Test document for deletion"

    provider_mock = AsyncMock()
    provider_mock.upload_file = AsyncMock(
        return_value=type(
            "Metadata",
            (),
            {
                "uri": "/test/to_delete.pdf",
                "file_name": "to_delete.pdf",
                "file_size": len(file_content),
            },
        )(),
    )
    provider_mock.delete_file = AsyncMock(return_value=True)

    with patch("ext.file_source.FileSourceFactory.create", return_value=provider_mock):
        files = {
            "file": ("to_delete.pdf", BytesIO(file_content), "application/pdf"),
        }
        data = {
            "collection_id": collection_id,
            "file_source_id": file_source_id,
        }

        response = client.post("/v1/document", files=files, data=data)
        assert response.status_code == 200
        temp_doc_id = response.json()["data"]["id"]

        # 更新文档状态为 success
        doc = await Document.get(id=temp_doc_id)
        doc.status = DocumentStatusEnum.success
        await doc.save()

        # 删除文档
        response = client.delete(f"/v1/document/{temp_doc_id}")
        assert response.status_code == 200
        result = response.json()
        assert result["code"] == 0
        assert result["data"]["deleted"] == 1


@pytest.mark.asyncio
async def test_delete_document_failure_status(client, embedding_model_setup, bind_mock_providers):
    """测试删除状态为 failure 的文档"""
    from ext.ext_tortoise.models.knowledge_base import Document
    from ext.ext_tortoise.enums import DocumentStatusEnum

    collection_id = embedding_model_setup["collection_id"]
    file_source_id = embedding_model_setup["file_source_id"]

    # 创建一个临时文档
    file_content = b"Test document with failure status"

    provider_mock = AsyncMock()
    provider_mock.upload_file = AsyncMock(
        return_value=type(
            "Metadata",
            (),
            {
                "uri": "/test/failed_doc.pdf",
                "file_name": "failed_doc.pdf",
                "file_size": len(file_content),
            },
        )(),
    )
    provider_mock.delete_file = AsyncMock(return_value=True)

    with patch("ext.file_source.FileSourceFactory.create", return_value=provider_mock):
        files = {
            "file": ("failed_doc.pdf", BytesIO(file_content), "application/pdf"),
        }
        data = {
            "collection_id": collection_id,
            "file_source_id": file_source_id,
        }

        response = client.post("/v1/document", files=files, data=data)
        assert response.status_code == 200
        temp_doc_id = response.json()["data"]["id"]

        # 更新文档状态为 failure
        doc = await Document.get(id=temp_doc_id)
        doc.status = DocumentStatusEnum.failure
        await doc.save()

        # 删除文档
        response = client.delete(f"/v1/document/{temp_doc_id}")
        assert response.status_code == 200
        result = response.json()
        assert result["code"] == 0


def test_delete_document_pending_status_should_fail(client, mock_file_source_upload, embedding_model_setup):
    """测试删除状态为 pending 的文档（应该失败）"""
    collection_id = embedding_model_setup["collection_id"]
    file_source_id = embedding_model_setup["file_source_id"]

    # 先创建一个测试文档
    file_content = b"Test PDF content for pending status"
    files = {
        "file": ("test_pending.pdf", BytesIO(file_content), "application/pdf"),
    }
    data = {
        "collection_id": collection_id,
        "file_source_id": file_source_id,
    }

    response = client.post("/v1/document", files=files, data=data)
    assert response.status_code == 200
    result = response.json()
    assert result["code"] == 0
    document_id = result["data"]["id"]

    # document_id 对应的文档状态应该是 pending
    response = client.delete(f"/v1/document/{document_id}")
    assert response.status_code == 200
    result = response.json()
    assert result["code"] != 0
    assert "当前状态不支持删除" in result["message"]


def test_delete_nonexistent_document(client):
    """测试删除不存在的文档"""
    response = client.delete("/v1/document/999999")
    assert response.status_code == 200
    result = response.json()
    assert result["code"] != 0


# =============================================================================
# Additional Tests
# =============================================================================


def test_create_document_by_uri_without_extension(client, mock_file_source_metadata, embedding_model_setup):
    """测试通过 URI 创建文档（URI 不包含扩展名）"""
    collection_id = embedding_model_setup["collection_id"]
    file_source_id = embedding_model_setup["file_source_id"]

    data = {
        "collection_id": collection_id,
        "file_source_id": file_source_id,
        "uri": "https://example.com/document",
    }

    response = client.post("/v1/document", data=data)
    assert response.status_code == 200
    result = response.json()
    assert result["code"] == 0


def test_create_document_with_empty_display_name(client, mock_file_source_upload, embedding_model_setup):
    """测试创建文档时 display_name 为空字符串（应使用原始文件名）"""
    collection_id = embedding_model_setup["collection_id"]
    file_source_id = embedding_model_setup["file_source_id"]

    file_content = b"Test content"
    files = {
        "file": ("empty_display_name.pdf", BytesIO(file_content), "application/pdf"),
    }
    data = {
        "collection_id": collection_id,
        "file_source_id": file_source_id,
        "display_name": "",
    }

    response = client.post("/v1/document", files=files, data=data)
    assert response.status_code == 200
    result = response.json()
    assert result["code"] == 0
    # 空字符串应该被视为 None，使用原始文件名
    assert result["data"]["display_name"] == "empty_display_name.pdf"


@pytest.fixture
def created_document_id(client, mock_file_source_upload, embedding_model_setup):
    collection_id = embedding_model_setup["collection_id"]
    file_source_id = embedding_model_setup["file_source_id"]

    files = {
        "file": ("doc_api_test.pdf", BytesIO(b"document api test"), "application/pdf"),
    }
    data = {
        "collection_id": collection_id,
        "file_source_id": file_source_id,
    }
    response = client.post("/v1/document", files=files, data=data)
    assert response.status_code == 200
    result = response.json()
    assert result["code"] == 0
    return result["data"]["id"]


def test_document_stream_api(client, created_document_id):
    async def fake_stream(uri: str, chunk_size: int = 8192):
        yield b"stream-content"

    provider_mock = AsyncMock()
    provider_mock.get_file_stream = fake_stream

    with patch("ext.file_source.FileSourceFactory.create", return_value=provider_mock):
        response = client.get(f"/v1/document/{created_document_id}/stream")
        assert response.status_code == 200
        assert response.content == b"stream-content"
        assert "inline;" in response.headers.get("content-disposition", "")


@pytest.mark.asyncio
async def test_document_pages_api_with_page_number_and_parsed_fallback(client, created_document_id):
    from ext.ext_tortoise.models.knowledge_base import Document, DocumentPages

    document = await Document.get(id=created_document_id)
    document.parsed_uri = "/test/parsed.md"
    await document.save()

    await DocumentPages.create(
        document_id=created_document_id,
        page_number=1,
        content="",
        tables=[],
        images=[],
        metadata={},
    )
    await DocumentPages.create(
        document_id=created_document_id,
        page_number=2,
        content="page-2-content",
        tables=[],
        images=[],
        metadata={},
    )

    provider_mock = AsyncMock()
    provider_mock.get_file = AsyncMock(return_value=b"parsed-uri-content")

    with patch("ext.file_source.FileSourceFactory.create", return_value=provider_mock):
        response = client.get(f"/v1/document/{created_document_id}/pages", params={"page_number": 1})
        assert response.status_code == 200
        result = response.json()
        assert result["code"] == 0
        assert len(result["data"]) == 1
        assert result["data"][0]["page_number"] == 1
        assert result["data"][0]["content"] == "parsed-uri-content"


def test_document_chunk_crud_api(client, created_document_id):
    create_resp = client.post(
        f"/v1/document/{created_document_id}/chunks",
        json={
            "content": "chunk-content-v1",
            "pages": [1, 2],
            "start": {"page_number": 1, "offset": 0},
            "end": {"page_number": 2, "offset": 10},
            "metadata": {"source": "manual"},
            "manual_add": True,
        },
    )
    assert create_resp.status_code == 200
    create_result = create_resp.json()
    assert create_result["code"] == 0
    chunk_id = create_result["data"]["id"]

    list_resp = client.get(f"/v1/document/{created_document_id}/chunks", params={"page_number": 2})
    assert list_resp.status_code == 200
    list_result = list_resp.json()
    assert list_result["code"] == 0
    assert any(item["id"] == chunk_id for item in list_result["data"])

    delete_resp = client.delete(f"/v1/document/{created_document_id}/chunks/{chunk_id}")
    assert delete_resp.status_code == 200
    delete_result = delete_resp.json()
    assert delete_result["code"] == 0
    assert delete_result["data"]["deleted"] == 1


def test_document_generated_faq_crud_api(client, created_document_id):
    create_resp = client.post(
        f"/v1/document/{created_document_id}/faqs",
        json={
            "content": "faq-ref-content",
            "question": "Q1?",
            "answer": "A1",
            "manual_add": True,
            "enabled": True,
        },
    )
    assert create_resp.status_code == 200
    create_result = create_resp.json()
    assert create_result["code"] == 0
    faq_id = create_result["data"]["id"]

    list_resp = client.get(f"/v1/document/{created_document_id}/faqs")
    assert list_resp.status_code == 200
    list_result = list_resp.json()
    assert list_result["code"] == 0
    assert any(item["id"] == faq_id for item in list_result["data"])

    update_resp = client.put(
        f"/v1/document/{created_document_id}/faqs/{faq_id}",
        json={
            "question": "Q1-updated?",
            "answer": "A1-updated",
            "enabled": False,
        },
    )
    assert update_resp.status_code == 200
    assert update_resp.json()["code"] == 0

    delete_resp = client.delete(f"/v1/document/{created_document_id}/faqs/{faq_id}")
    assert delete_resp.status_code == 200
    delete_result = delete_resp.json()
    assert delete_result["code"] == 0
    assert delete_result["data"]["deleted"] == 1
