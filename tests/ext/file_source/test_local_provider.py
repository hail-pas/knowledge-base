"""
测试 Local File Source Provider

最小可用验证的单元测试
"""

import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ext.file_source.providers.local import LocalFileSourceProvider
from ext.file_source.types import LocalFileSourceExtraConfig


class TestLocalFileSourceProvider:
    """测试 Local File Source Provider"""

    def test_create_provider(self, local_provider_config):
        """测试创建 Provider 实例"""
        provider = LocalFileSourceProvider(**local_provider_config)
        assert provider is not None
        assert provider.storage_location == local_provider_config["storage_location"]
        assert isinstance(provider.extra_config, LocalFileSourceExtraConfig)

    def test_validate_connection(self, local_provider_config):
        """测试连接验证"""
        provider = LocalFileSourceProvider(**local_provider_config)
        assert provider.storage_location is not None

    async def test_validate_connection_success(self, local_provider_config):
        """测试连接验证成功"""
        provider = LocalFileSourceProvider(**local_provider_config)
        is_connected = await provider.validate_connection()
        assert is_connected is True

    async def test_upload_and_get_file(
        self, local_provider_config, local_temp_dir, sample_file_content, sample_file_name
    ):
        """测试上传和获取文件"""
        provider = LocalFileSourceProvider(**local_provider_config)
        file_path = local_temp_dir / sample_file_name

        await provider.upload_file(str(file_path), sample_file_content)

        content = await provider.get_file(str(file_path))
        assert content == sample_file_content

    async def test_file_exists(self, local_provider_config, local_temp_dir, sample_file_content, sample_file_name):
        """测试文件存在检查"""
        provider = LocalFileSourceProvider(**local_provider_config)
        file_path = local_temp_dir / sample_file_name

        await provider.upload_file(str(file_path), sample_file_content)

        exists = await provider.file_exists(str(file_path))
        assert exists is True

        non_existent_file = local_temp_dir / "non_existent.txt"
        exists = await provider.file_exists(str(non_existent_file))
        assert exists is False

    async def test_get_file_metadata(
        self, local_provider_config, local_temp_dir, sample_file_content, sample_file_name
    ):
        """测试获取文件元数据"""
        provider = LocalFileSourceProvider(**local_provider_config)
        file_path = local_temp_dir / sample_file_name

        await provider.upload_file(str(file_path), sample_file_content)

        metadata = await provider.get_file_metadata(str(file_path))
        assert metadata.file_name == sample_file_name
        assert metadata.file_size == len(sample_file_content)
        assert metadata.uri == str(file_path)
        assert metadata.etag is not None

    async def test_list_files(self, local_provider_config, local_temp_dir, sample_file_content):
        """测试列出文件"""
        provider = LocalFileSourceProvider(**local_provider_config)

        await provider.upload_file(str(local_temp_dir / "file1.txt"), sample_file_content)
        await provider.upload_file(str(local_temp_dir / "file2.txt"), b"Another file content")

        files = await provider.list_files()
        assert len(files) >= 2
        file_names = [f.file_name for f in files]
        assert "file1.txt" in file_names or "file2.txt" in file_names

    async def test_get_file_stream(self, local_provider_config, local_temp_dir, sample_file_content, sample_file_name):
        """测试获取文件流"""
        provider = LocalFileSourceProvider(**local_provider_config)
        file_path = local_temp_dir / sample_file_name

        await provider.upload_file(str(file_path), sample_file_content)

        chunks = []
        async for chunk in provider.get_file_stream(str(file_path), chunk_size=10):
            chunks.append(chunk)

        content = b"".join(chunks)
        assert content == sample_file_content

    async def test_delete_file(self, local_provider_config, local_temp_dir, sample_file_content, sample_file_name):
        """测试删除文件"""
        provider = LocalFileSourceProvider(**local_provider_config)
        file_path = local_temp_dir / sample_file_name

        await provider.upload_file(str(file_path), sample_file_content)
        assert await provider.file_exists(str(file_path)) is True

        deleted = await provider.delete_file(str(file_path))
        assert deleted is True
        assert await provider.file_exists(str(file_path)) is False

    async def test_list_files_with_limit(self, local_provider_config, local_temp_dir):
        """测试限制文件列表数量"""
        provider = LocalFileSourceProvider(**local_provider_config)

        await provider.upload_file(str(local_temp_dir / "file1.txt"), b"content1")
        await provider.upload_file(str(local_temp_dir / "file2.txt"), b"content2")
        await provider.upload_file(str(local_temp_dir / "file3.txt"), b"content3")

        files = await provider.list_files(limit=2)
        assert len(files) <= 2
