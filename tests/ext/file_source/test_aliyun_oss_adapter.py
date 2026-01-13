"""AliyunOSSAdapter 单元测试（使用真实 OSS 配置）"""
import os
from datetime import datetime
from uuid import uuid4

import pytest

from ext.file_source.providers.aliyun_oss import AliyunOSSAdapter
from ext.file_source.base import FileItem


@pytest.fixture
def oss_config():
    """获取 OSS 配置（从环境变量）"""
    config = {
        "endpoint": os.environ.get("ALIYUN_ENDPOINT"),
        "bucket": os.environ.get("ALIYUN_BUCKET_NAME"),
        "access_key_id": os.environ.get("ALIYUN_ACCESS_KEY_ID"),
        "access_key_secret": os.environ.get("ALIYUN_ACCESS_KEY_SECRET"),
        "region": os.environ.get("ALIYUN_REGION"),
    }

    # 检查必要配置是否存在
    if not all([config["endpoint"], config["bucket"], config["access_key_id"], config["access_key_secret"]]):
        pytest.skip("OSS 配置不完整，请设置环境变量: ALIYUN_ENDPOINT, ALIYUN_BUCKET_NAME, ALIYUN_ACCESS_KEY_ID, ALIYUN_ACCESS_KEY_SECRET")

    return config


@pytest.fixture
def oss_adapter(oss_config):
    """创建 OSS 适配器实例"""
    return AliyunOSSAdapter(oss_config)


@pytest.fixture
def test_file_uri():
    """生成唯一的测试文件 URI"""
    return f"test/unit_test_{uuid4().hex}.txt"


@pytest.mark.asyncio
class TestAliyunOSSAdapter:
    """AliyunOSSAdapter 测试类"""

    async def test_validate_success(self, oss_adapter):
        """测试成功验证配置"""
        result = await oss_adapter.validate()
        assert result is True

    async def test_upload_and_get_file(self, oss_adapter, test_file_uri):
        """测试上传和获取文件内容"""
        # 准备测试内容
        expected_content = b"Hello, OSS! This is a unit test."

        # 上传文件
        upload_result = await oss_adapter.upload_file(
            uri=test_file_uri,
            content=expected_content,
            content_type="text/plain"
        )
        assert upload_result is True

        try:
            # 获取文件内容
            actual_content = await oss_adapter.get_file(test_file_uri)
            assert actual_content == expected_content
        finally:
            # 清理：删除测试文件
            await oss_adapter.delete_file(test_file_uri)

    async def test_get_file_not_found(self, oss_adapter):
        """测试获取不存在的文件"""
        non_existent_uri = f"test/non_existent_{uuid4().hex}.txt"

        with pytest.raises(FileNotFoundError, match="OSS 对象不存在"):
            await oss_adapter.get_file(non_existent_uri)

    async def test_get_file_meta(self, oss_adapter, test_file_uri):
        """测试获取文件元数据"""
        # 准备测试内容
        test_content = b"Test metadata content"

        # 上传文件
        await oss_adapter.upload_file(
            uri=test_file_uri,
            content=test_content,
            content_type="text/plain"
        )

        try:
            # 获取文件元数据
            file_item = await oss_adapter.get_file_meta(test_file_uri)

            assert isinstance(file_item, FileItem)
            assert file_item.uri == test_file_uri
            assert file_item.name == test_file_uri.split("/")[-1]
            assert file_item.size == len(test_content)
            assert file_item.content_type == "text/plain"
            assert isinstance(file_item.last_modified, datetime)
            assert "etag" in file_item.metadata
        finally:
            # 清理：删除测试文件
            await oss_adapter.delete_file(test_file_uri)

    async def test_get_file_meta_not_found(self, oss_adapter):
        """测试获取不存在文件的元数据"""
        non_existent_uri = f"test/non_existent_{uuid4().hex}.txt"

        with pytest.raises(FileNotFoundError, match="OSS 对象不存在"):
            await oss_adapter.get_file_meta(non_existent_uri)

    async def test_check_file_exists_true(self, oss_adapter, test_file_uri):
        """测试检查存在的文件"""
        # 上传测试文件
        await oss_adapter.upload_file(
            uri=test_file_uri,
            content=b"Test content",
            content_type="text/plain"
        )

        try:
            # 检查文件是否存在
            result = await oss_adapter.check_file_exists(test_file_uri)
            assert result is True
        finally:
            # 清理：删除测试文件
            await oss_adapter.delete_file(test_file_uri)

    async def test_check_file_exists_false(self, oss_adapter):
        """测试检查不存在的文件"""
        non_existent_uri = f"test/non_existent_{uuid4().hex}.txt"

        result = await oss_adapter.check_file_exists(non_existent_uri)
        assert result is False

    async def test_delete_file_success(self, oss_adapter, test_file_uri):
        """测试成功删除文件"""
        # 上传测试文件
        await oss_adapter.upload_file(
            uri=test_file_uri,
            content=b"Content to be deleted",
            content_type="text/plain"
        )

        # 确认文件存在
        assert await oss_adapter.check_file_exists(test_file_uri) is True

        # 删除文件
        result = await oss_adapter.delete_file(test_file_uri)
        assert result is True

        # 确认文件已被删除
        assert await oss_adapter.check_file_exists(test_file_uri) is False

    async def test_upload_file_with_custom_content_type(self, oss_adapter, test_file_uri):
        """测试上传文件时指定自定义 Content-Type"""
        test_content = b"JSON content"

        # 上传 JSON 文件
        await oss_adapter.upload_file(
            uri=test_file_uri,
            content=test_content,
            content_type="application/json"
        )

        try:
            # 获取元数据验证 Content-Type
            file_item = await oss_adapter.get_file_meta(test_file_uri)
            assert file_item.content_type == "application/json"
        finally:
            # 清理：删除测试文件
            await oss_adapter.delete_file(test_file_uri)
