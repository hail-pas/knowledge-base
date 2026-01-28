"""
Local File System Provider
"""

import os
import hashlib
from pathlib import Path
import asyncio
from datetime import datetime
import aiofiles

from ext.file_source.base import BaseFileSourceProvider, FileMetadata
from ext.file_source.types import LocalFileSourceExtraConfig
from loguru import logger


class LocalFileSourceProvider(BaseFileSourceProvider[LocalFileSourceExtraConfig]):
    """本地文件系统 Provider"""

    def _validate_config(self) -> None:
        if not self.storage_location:
            raise ValueError("storage_location (root_path) is required for local file source")

        path = Path(self.storage_location)
        if not path.exists():
            raise ValueError(f"Root path does not exist: {self.storage_location}")
        if not path.is_dir():
            raise ValueError(f"Root path is not a directory: {self.storage_location}")

    async def validate_connection(self) -> bool:
        """验证根路径是否存在且可访问"""
        try:
            path = Path(self.storage_location or "")
            return path.exists() and path.is_dir() and os.access(path, os.R_OK)
        except Exception as e:
            logger.error(f"Failed to validate local connection: {e}")
            return False

    async def list_files(
        self,
        prefix: str = "",
        recursive: bool = False,
        limit: int | None = None,
    ) -> list[FileMetadata]:
        """列出文件"""
        base_path = Path(self.storage_location or "")
        search_path = base_path / prefix if prefix else base_path

        files = []
        pattern = "**/*" if recursive else "*"

        count = 0
        for file_path in search_path.glob(pattern):
            if limit and count >= limit:
                break

            if not file_path.is_file():
                continue

            if self.extra_config.follow_symlinks and file_path.is_symlink():
                file_path = file_path.resolve()

            if self.extra_config.allowed_extensions:
                if file_path.suffix not in self.extra_config.allowed_extensions:
                    continue

            if self.extra_config.excluded_extensions:
                if file_path.suffix in self.extra_config.excluded_extensions:
                    continue

            stat = file_path.stat()
            if self.extra_config.max_file_size:
                if stat.st_size > self.extra_config.max_file_size:
                    continue

            if self.extra_config.require_readable and not os.access(file_path, os.R_OK):
                continue

            files.append(
                FileMetadata(  # type: ignore[call-arg]
                    uri=str(file_path),
                    file_name=file_path.name,
                    file_size=stat.st_size,
                    last_modified=datetime.fromtimestamp(stat.st_mtime),
                    etag=self._compute_etag(file_path),
                ),
            )
            count += 1

        return files

    async def get_file(self, uri: str) -> bytes:
        """获取文件内容"""
        async with aiofiles.open(uri, "rb") as f:
            return await f.read()

    async def get_file_stream(self, uri: str, chunk_size: int = 8192):  # type: ignore[misc]
        """获取文件流"""
        async with aiofiles.open(uri, "rb") as f:
            while True:
                chunk = await f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    async def get_file_metadata(self, uri: str) -> FileMetadata:
        """获取文件元数据"""
        path = Path(uri)
        stat = path.stat()
        return FileMetadata(  # type: ignore[call-arg]
            uri=uri,
            file_name=path.name,
            file_size=stat.st_size,
            last_modified=datetime.fromtimestamp(stat.st_mtime),
            etag=self._compute_etag(path),
        )

    async def file_exists(self, uri: str) -> bool:
        """检查文件是否存在"""
        return Path(uri).exists() and Path(uri).is_file()

    async def upload_file(self, uri: str, content: bytes, content_type: str | None = None) -> FileMetadata:
        """上传（写入）文件"""
        async with aiofiles.open(uri, "wb") as f:
            await f.write(content)

        return await self.get_file_metadata(uri)

    async def delete_file(self, uri: str) -> bool:
        """删除文件"""
        try:
            Path(uri).unlink()
            return True
        except FileNotFoundError:
            return False
        except Exception:
            return False

    def _compute_etag(self, path: Path) -> str:
        """计算文件 ETag（MD5 hash）"""
        return hashlib.md5(path.read_bytes()).hexdigest()
