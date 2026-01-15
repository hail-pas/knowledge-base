"""Document API 测试共享配置和 Fixtures"""

import pytest
import time
import random
import string


def _generate_unique_id():
    """生成唯一标识符"""
    return f"{int(time.time() * 1000)}{''.join(random.choices(string.ascii_lowercase, k=4))}"


@pytest.fixture
def collection_data():
    """Collection 测试数据"""
    unique_id = _generate_unique_id()
    return {
        "name": f"test_collection_{unique_id}",
        "description": f"Test collection description {unique_id}",
        "user_id": None,
        "tenant_id": None,
        "role_id": None,
        "is_public": False,
        "is_temp": False,
    }


@pytest.fixture
def file_source_data():
    """FileSource 测试数据"""
    unique_id = _generate_unique_id()
    return {
        "name": f"test_file_source_{unique_id}",
        "type": "local_file",
        "config": {
            "base_path": "/tmp/kb_test_files",
        },
        "is_enabled": True,
        "is_default": False,
        "description": f"Test file source {unique_id}",
    }


@pytest.fixture
async def default_file_source():
    """创建并返回一个默认的FileSource用于测试"""
    from ext.ext_tortoise.models.knowledge_base import FileSource
    from ext.ext_tortoise.enums import FileSourceTypeEnum

    unique_id = _generate_unique_id()
    file_source = await FileSource.create(
        name=f"default_test_source_{unique_id}",
        type=FileSourceTypeEnum.local_file,
        config={"base_path": "/tmp/kb_test_default"},
        is_enabled=True,
        is_default=True,
        description="Default test file source",
    )
    yield file_source
    # Cleanup
    await file_source.delete()


@pytest.fixture
async def test_collection(collection_data):
    """创建并返回一个Collection用于测试"""
    from ext.ext_tortoise.models.knowledge_base import Collection

    collection = await Collection.create(**collection_data)
    yield collection
    # Cleanup
    await collection.delete()


@pytest.fixture
def document_create_by_upload_data(test_collection):
    """Document上传方式创建的测试数据"""
    unique_id = _generate_unique_id()
    return {
        "collection_id": test_collection.id,
        "display_name": f"Test Document {unique_id}",
        "short_summary": f"Test summary {unique_id}",
        "long_summary": f"This is a longer test summary for document {unique_id}",
        "status": "pending",
        "file_source_id": None,  # 不传，使用默认
    }


@pytest.fixture
def document_create_by_uri_data(test_collection, default_file_source):
    """Document URI方式创建的测试数据"""
    unique_id = _generate_unique_id()
    return {
        "collection_id": test_collection.id,
        "file_source_id": default_file_source.id,
        "uri": f"/tmp/test_file_{unique_id}.txt",
        "file_name": f"test_file_{unique_id}.txt",
        "display_name": f"URI Test Document {unique_id}",
        "extension": "txt",
        "file_size": 1024,
        "source_last_modified": None,
        "source_version_key": None,
        "short_summary": f"Test summary from URI {unique_id}",
        "long_summary": f"This is a longer test summary for URI document {unique_id}",
        "source_meta": None,
        "status": "pending",
    }
