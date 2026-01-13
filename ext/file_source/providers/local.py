import mimetypes
import re
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

from ext.file_source.base import FileSourceAdapter, FileItem


class LocalAdapter(FileSourceAdapter):
    """本地文件系统适配器

    对于本地文件，uri 直接是绝对路径，无需额外配置。
    """

    async def validate(self) -> bool:
        """验证配置是否正确

        本地文件适配器无需额外配置，始终返回 True。
        """
        return True

    async def get_file(self, uri: str) -> bytes:
        """获取单文件内容

        Args:
            uri: 文件的绝对路径

        Returns:
            文件内容（字节数组）

        Raises:
            FileNotFoundError: 文件不存在
        """
        path = Path(uri)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"文件不存在: {path}")
        return path.read_bytes()

    async def get_file_stream(self, uri: str) -> AsyncIterator[bytes]: # type: ignore
        """获取文件流（大文件）

        Args:
            uri: 文件的绝对路径

        Yields:
            文件内容分块（8KB）

        Raises:
            FileNotFoundError: 文件不存在
        """
        path = Path(uri)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"文件不存在: {path}")

        with open(path, "rb") as f:
            while chunk := f.read(8192):
                yield chunk

    async def get_file_meta(self, uri: str) -> FileItem:
        """获取文件元数据

        Args:
            uri: 文件的绝对路径

        Returns:
            文件元数据

        Raises:
            FileNotFoundError: 文件不存在
        """
        path = Path(uri)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"文件不存在: {path}")

        stat = path.stat()

        return FileItem(
            uri=uri,
            name=path.name,
            size=stat.st_size,
            content_type=mimetypes.guess_type(path)[0] or "application/octet-stream",
            last_modified=datetime.fromtimestamp(stat.st_mtime),
            metadata={}
        )

    async def list_files(self, prefix: str = "", filter=None) -> AsyncIterator[FileItem]: # type: ignore
        """列出文件（批量获取）

        Args:
            prefix: 路径前缀（如果是目录，则列出该目录下的文件）
            filter: 文件过滤条件（FileFilter 对象）

        Yields:
            FileItem: 文件项

        Raises:
            FileNotFoundError: 前缀路径不存在
            ValueError: 前缀路径不是目录
        """
        path = Path(prefix) if prefix else Path.cwd()

        if not path.exists():
            raise FileNotFoundError(f"路径不存在: {path}")

        if not path.is_dir():
            raise ValueError(f"路径不是目录: {path}")

        # 遍历目录
        for item in path.rglob("*"):
            if not item.is_file():
                continue

            # 应用过滤条件
            if filter:
                # 检查扩展名
                if filter.allowed_extensions:
                    ext = item.suffix.lower()
                    if ext not in [e.lower() for e in filter.allowed_extensions]:
                        continue

                if filter.blocked_extensions:
                    ext = item.suffix.lower()
                    if ext in [e.lower() for e in filter.blocked_extensions]:
                        continue

                # 检查文件大小
                stat = item.stat()
                if filter.min_size is not None and stat.st_size < filter.min_size:
                    continue

                if filter.max_size is not None and stat.st_size > filter.max_size:
                    continue

                # 检查文件名模式
                if filter.name_pattern:
                    if not re.search(filter.name_pattern, item.name):
                        continue

            # 获取文件元数据
            stat = item.stat()
            file_item = FileItem(
                uri=str(item.absolute()),
                name=item.name,
                size=stat.st_size,
                content_type=mimetypes.guess_type(item)[0] or "application/octet-stream",
                last_modified=datetime.fromtimestamp(stat.st_mtime),
                metadata={}
            )

            yield file_item

    async def check_file_exists(self, uri: str) -> bool:
        """检查文件是否存在

        Args:
            uri: 文件的绝对路径

        Returns:
            文件是否存在
        """
        path = Path(uri)
        return path.exists() and path.is_file()

    async def upload_file(
        self,
        uri: str,
        content: bytes,
        content_type: str = "application/octet-stream",
        metadata: dict | None = None
    ) -> bool:
        """上传文件到本地文件系统

        对于本地文件系统，"上传"意味着写入文件到指定路径。

        Args:
            uri: 文件的绝对路径（目标路径）
            content: 文件内容
            content_type: MIME 类型（忽略）
            metadata: 元数据（忽略）

        Returns:
            是否写入成功

        Raises:
            OSError: 写入文件失败（如权限不足、目录不存在等）
        """
        path = Path(uri)

        # 确保父目录存在
        path.parent.mkdir(parents=True, exist_ok=True)

        # 写入文件内容
        try:
            path.write_bytes(content)
            return True
        except Exception as e:
            raise OSError(f"写入文件失败: {uri}") from e

    async def delete_file(self, uri: str) -> bool:
        """删除本地文件

        Args:
            uri: 文件的绝对路径

        Returns:
            是否删除成功

        Raises:
            FileNotFoundError: 文件不存在
            OSError: 删除文件失败（如权限不足）
        """
        path = Path(uri)

        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        if not path.is_file():
            raise ValueError(f"路径不是文件: {path}")

        try:
            path.unlink()
            return True
        except Exception as e:
            raise OSError(f"删除文件失败: {uri}") from e
