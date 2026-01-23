"""
测试 Aliyun OSS File Source Provider

最小可用验证的单元测试
"""

import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ext.file_source.providers.aliyun_oss import AliyunOSSFileSourceProvider
from ext.file_source.types import AliyunOSSExtraConfig
from tests.ext.file_source.conftest import skip_if_no_aliyun_config


@skip_if_no_aliyun_config
class TestAliyunOSSFileSourceProvider:
    """测试 Aliyun OSS File Source Provider"""

    def test_create_provider(self, aliyun_oss_provider_config):
        """测试创建 Provider 实例"""
        provider = AliyunOSSFileSourceProvider(**aliyun_oss_provider_config)
        assert provider is not None
        assert provider.storage_location == aliyun_oss_provider_config["storage_location"]
        assert isinstance(provider.extra_config, AliyunOSSExtraConfig)

    @pytest.mark.asyncio
    async def test_validate_connection(self, aliyun_oss_provider_config):
        """测试连接验证"""
        provider = AliyunOSSFileSourceProvider(**aliyun_oss_provider_config)
        is_connected = await provider.validate_connection()
        assert is_connected is True

    @pytest.mark.asyncio
    async def test_upload_and_get_file(self, aliyun_oss_provider_config, sample_file_content, sample_file_name):
        """测试上传和获取文件"""
        provider = AliyunOSSFileSourceProvider(**aliyun_oss_provider_config)
        bucket_name = aliyun_oss_provider_config["storage_location"]
        test_key = f"test/{sample_file_name}"
        uri = f"oss://{bucket_name}/{test_key}"

        await provider.upload_file(uri, sample_file_content)

        content = await provider.get_file(uri)
        assert content == sample_file_content

        await provider.delete_file(uri)

    @pytest.mark.asyncio
    async def test_file_exists(self, aliyun_oss_provider_config, sample_file_content, sample_file_name):
        """测试文件存在检查"""
        provider = AliyunOSSFileSourceProvider(**aliyun_oss_provider_config)
        bucket_name = aliyun_oss_provider_config["storage_location"]
        test_key = f"test/{sample_file_name}"
        uri = f"oss://{bucket_name}/{test_key}"

        await provider.upload_file(uri, sample_file_content)

        exists = await provider.file_exists(uri)
        assert exists is True

        await provider.delete_file(uri)

        exists = await provider.file_exists(uri)
        assert exists is False

    @pytest.mark.asyncio
    async def test_get_file_metadata(self, aliyun_oss_provider_config, sample_file_content, sample_file_name):
        """测试获取文件元数据"""
        provider = AliyunOSSFileSourceProvider(**aliyun_oss_provider_config)
        bucket_name = aliyun_oss_provider_config["storage_location"]
        test_key = f"test/{sample_file_name}"
        uri = f"oss://{bucket_name}/{test_key}"

        await provider.upload_file(uri, sample_file_content)

        metadata = await provider.get_file_metadata(uri)
        assert metadata.file_name == sample_file_name
        assert metadata.file_size == len(sample_file_content)
        assert metadata.uri == uri

        await provider.delete_file(uri)

    @pytest.mark.asyncio
    async def test_list_files(self, aliyun_oss_provider_config, sample_file_content):
        """测试列出文件"""
        provider = AliyunOSSFileSourceProvider(**aliyun_oss_provider_config)
        bucket_name = aliyun_oss_provider_config["storage_location"]

        await provider.upload_file(f"oss://{bucket_name}/test/file1.txt", sample_file_content)
        await provider.upload_file(f"oss://{bucket_name}/test/file2.txt", b"Another file content")

        files = await provider.list_files(prefix="test/")
        assert len(files) >= 2

        await provider.delete_file(f"oss://{bucket_name}/test/file1.txt")
        await provider.delete_file(f"oss://{bucket_name}/test/file2.txt")

    @pytest.mark.asyncio
    async def test_get_file_stream(self, aliyun_oss_provider_config, sample_file_content, sample_file_name):
        """测试获取文件流"""
        provider = AliyunOSSFileSourceProvider(**aliyun_oss_provider_config)
        bucket_name = aliyun_oss_provider_config["storage_location"]
        test_key = f"test/{sample_file_name}"
        uri = f"oss://{bucket_name}/{test_key}"

        await provider.upload_file(uri, sample_file_content)

        chunks = []
        async for chunk in provider.get_file_stream(uri, chunk_size=10):
            chunks.append(chunk)

        content = b"".join(chunks)
        assert content == sample_file_content

        await provider.delete_file(uri)

    @pytest.mark.asyncio
    async def test_delete_file(self, aliyun_oss_provider_config, sample_file_content, sample_file_name):
        """测试删除文件"""
        provider = AliyunOSSFileSourceProvider(**aliyun_oss_provider_config)
        bucket_name = aliyun_oss_provider_config["storage_location"]
        test_key = f"test/{sample_file_name}"
        uri = f"oss://{bucket_name}/{test_key}"

        await provider.upload_file(uri, sample_file_content)
        assert await provider.file_exists(uri) is True

        deleted = await provider.delete_file(uri)
        assert deleted is True
        assert await provider.file_exists(uri) is False

    @pytest.mark.asyncio
    async def test_list_files_with_limit(self, aliyun_oss_provider_config):
        """测试限制文件列表数量"""
        provider = AliyunOSSFileSourceProvider(**aliyun_oss_provider_config)
        bucket_name = aliyun_oss_provider_config["storage_location"]

        await provider.upload_file(f"oss://{bucket_name}/test/limit1.txt", b"content1")
        await provider.upload_file(f"oss://{bucket_name}/test/limit2.txt", b"content2")
        await provider.upload_file(f"oss://{bucket_name}/test/limit3.txt", b"content3")

        files = await provider.list_files(prefix="test/limit", limit=2)
        assert len(files) <= 2

        await provider.delete_file(f"oss://{bucket_name}/test/limit1.txt")
        await provider.delete_file(f"oss://{bucket_name}/test/limit2.txt")
        await provider.delete_file(f"oss://{bucket_name}/test/limit3.txt")

    @pytest.mark.asyncio
    async def test_upload_with_content_type(self, aliyun_oss_provider_config, sample_file_content):
        """测试上传文件时指定内容类型"""
        provider = AliyunOSSFileSourceProvider(**aliyun_oss_provider_config)
        bucket_name = aliyun_oss_provider_config["storage_location"]
        uri = f"oss://{bucket_name}/test/content_type.txt"

        await provider.upload_file(uri, sample_file_content, content_type="text/plain")

        metadata = await provider.get_file_metadata(uri)
        assert metadata.content_type == "text/plain"

        await provider.delete_file(uri)
