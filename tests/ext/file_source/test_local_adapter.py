"""LocalAdapter 单元测试"""
import os
from pathlib import Path
from datetime import datetime
from tempfile import TemporaryDirectory

import pytest
from dateutil.tz import tzlocal

from ext.file_source.providers.local import LocalAdapter
from ext.file_source.base import FileItem


@pytest.fixture
def temp_dir():
    """临时目录 fixture"""
    with TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_file(temp_dir):
    """创建示例测试文件"""
    file_path = Path(temp_dir) / "test.txt"
    content = b"Hello, World!"
    file_path.write_bytes(content)
    return str(file_path), content


@pytest.fixture
def large_file(temp_dir):
    """创建大文件（用于测试流式读取）"""
    file_path = Path(temp_dir) / "large.txt"
    content = b"x" * 20000  # 20KB
    file_path.write_bytes(content)
    return str(file_path), content


@pytest.mark.asyncio
class TestLocalAdapter:
    """LocalAdapter 测试类"""

    async def test_validate(self):
        """测试配置验证"""
        adapter = LocalAdapter({})
        result = await adapter.validate()
        assert result is True

    async def test_get_file_success(self, sample_file):
        """测试成功获取文件内容"""
        file_path, expected_content = sample_file
        adapter = LocalAdapter({})

        content = await adapter.get_file(file_path)
        assert content == expected_content

    async def test_get_file_not_found(self, temp_dir):
        """测试获取不存在的文件"""
        adapter = LocalAdapter({})
        non_existent_path = str(Path(temp_dir) / "non_existent.txt")

        with pytest.raises(FileNotFoundError, match="文件不存在"):
            await adapter.get_file(non_existent_path)

    async def test_get_file_stream_success(self, large_file):
        """测试成功获取文件流"""
        file_path, expected_content = large_file
        adapter = LocalAdapter({})

        chunks = []
        async for chunk in adapter.get_file_stream(file_path):
            chunks.append(chunk)

        actual_content = b"".join(chunks)
        assert actual_content == expected_content

    async def test_get_file_stream_not_found(self, temp_dir):
        """测试流式读取不存在的文件"""
        adapter = LocalAdapter({})
        non_existent_path = str(Path(temp_dir) / "non_existent.txt")

        with pytest.raises(FileNotFoundError, match="文件不存在"):
            chunks = []
            async for chunk in adapter.get_file_stream(non_existent_path):
                chunks.append(chunk)

    async def test_get_file_meta_success(self, sample_file):
        """测试成功获取文件元数据"""
        file_path, content = sample_file
        adapter = LocalAdapter({})

        file_item = await adapter.get_file_meta(file_path)

        assert isinstance(file_item, FileItem)
        assert file_item.uri == file_path
        assert file_item.name == "test.txt"
        assert file_item.size == len(content)
        assert file_item.content_type == "text/plain"
        assert isinstance(file_item.last_modified, datetime)

    async def test_get_file_meta_not_found(self, temp_dir):
        """测试获取不存在文件的元数据"""
        adapter = LocalAdapter({})
        non_existent_path = str(Path(temp_dir) / "non_existent.txt")

        with pytest.raises(FileNotFoundError, match="文件不存在"):
            await adapter.get_file_meta(non_existent_path)

    async def test_upload_file_success(self, temp_dir):
        """测试成功上传文件（写入本地）"""
        adapter = LocalAdapter({})
        target_path = str(Path(temp_dir) / "uploaded.txt")
        content = b"Test upload content"

        result = await adapter.upload_file(target_path, content)

        assert result is True
        assert Path(target_path).exists()
        assert Path(target_path).read_bytes() == content

    async def test_delete_file_success(self, temp_dir):
        """测试成功删除文件"""
        file_path = str(Path(temp_dir) / "to_delete.txt")
        Path(file_path).write_bytes(b"content to delete")

        adapter = LocalAdapter({})
        result = await adapter.delete_file(file_path)

        assert result is True
        assert not Path(file_path).exists()

    async def test_delete_file_not_found(self, temp_dir):
        """测试删除不存在的文件"""
        adapter = LocalAdapter({})
        non_existent_path = str(Path(temp_dir) / "non_existent.txt")

        with pytest.raises(FileNotFoundError, match="文件不存在"):
            await adapter.delete_file(non_existent_path)

    async def test_check_file_exists_true(self, sample_file):
        """测试检查存在的文件"""
        file_path, _ = sample_file
        adapter = LocalAdapter({})

        result = await adapter.check_file_exists(file_path)
        assert result is True

    async def test_check_file_exists_false(self, temp_dir):
        """测试检查不存在的文件"""
        adapter = LocalAdapter({})
        non_existent_path = str(Path(temp_dir) / "non_existent.txt")

        result = await adapter.check_file_exists(non_existent_path)
        assert result is False
