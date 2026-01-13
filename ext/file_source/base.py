from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional
from dataclasses import dataclass, field
from datetime import datetime
import os
import mimetypes


@dataclass
class FileItem:
    """统一的文件项描述"""
    uri: str                    # 唯一标识（本地为绝对路径）
    name: str                   # 文件名
    size: int                   # 大小
    content_type: str           # MIME 类型
    last_modified: datetime     # 最后修改时间
    metadata: Optional[dict] = field(default=None)       # 其他元数据


@dataclass
class FileFilter:
    """文件过滤条件（预留，用于后续批量获取）"""
    allowed_extensions: Optional[list[str]] = field(default=None)   # 允许的扩展名
    blocked_extensions: Optional[list[str]] = field(default=None)   # 禁止的扩展名
    min_size: Optional[int] = field(default=None)                   # 最小文件大小
    max_size: Optional[int] = field(default=None)                   # 最大文件大小
    name_pattern: Optional[str] = field(default=None)               # 文件名匹配模式


class FileSourceAdapter(ABC):
    """文件源适配器抽象基类

    所有文件源适配器都需要实现此接口，提供统一的文件获取方式。
    """

    def __init__(self, config: dict):
        """初始化适配器

        Args:
            config: 文件源配置（JSON 格式）
        """
        self.config = config

    @abstractmethod
    async def validate(self) -> bool:
        """验证配置是否正确

        Returns:
            配置是否有效
        """
        pass

    @abstractmethod
    async def get_file(self, uri: str) -> bytes:
        """获取单文件内容

        Args:
            uri: 文件唯一标识（本地为绝对路径）

        Returns:
            文件内容（字节数组）

        """
        pass

    async def get_file_stream(self, uri: str) -> AsyncIterator[bytes]:
        """获取文件流（大文件可选实现）

        Args:
            uri: 文件唯一标识

        Yields:
            文件内容分块

        Raises:
            NotImplementedError: 如果适配器不支持流式读取
        """
        raise NotImplementedError("此适配器不支持流式读取")

    async def upload_file(
        self,
        uri: str,
        content: bytes,
        content_type: str = "application/octet-stream",
        metadata: dict | None = None
    ) -> bool:
        """上传文件到文件源（可选实现）

        Args:
            uri: 文件唯一标识（上传后的目标路径）
            content: 文件内容
            content_type: MIME 类型（可选）
            metadata: 元数据（可选）

        Returns:
            是否上传成功

        Raises:
            NotImplementedError: 如果适配器不支持上传
        """
        raise NotImplementedError("此适配器不支持上传文件")

    async def upload_file_from_path(
        self,
        uri: str,
        file_path: str,
        content_type: str | None = None,
        metadata: dict | None = None
    ) -> bool:
        """从本地路径上传文件到文件源（可选实现）

        Args:
            uri: 文件唯一标识（上传后的目标路径）
            file_path: 本地文件路径
            content_type: MIME 类型（可选，如果为 None 则自动检测）
            metadata: 元数据（可选）

        Returns:
            是否上传成功

        Raises:
            NotImplementedError: 如果适配器不支持上传
            FileNotFoundError: 本地文件不存在
        """


        if not os.path.exists(file_path):
            raise FileNotFoundError(f"本地文件不存在: {file_path}")

        # 读取文件内容
        with open(file_path, "rb") as f:
            content = f.read()

        # 自动检测 content_type
        if content_type is None:
            content_type, _ = mimetypes.guess_type(file_path)
            if content_type is None:
                content_type = "application/octet-stream"

        # 调用上传方法
        return await self.upload_file(uri, content, content_type, metadata)

    async def download_file_to_path(
        self,
        uri: str,
        file_path: str,
        overwrite: bool = False
    ) -> bool:
        """下载文件到指定路径（可选实现）

        Args:
            uri: 文件唯一标识
            file_path: 本地目标路径
            overwrite: 是否覆盖已存在的文件（默认 False）

        Returns:
            是否下载成功

        Raises:
            NotImplementedError: 如果适配器不支持下载
            FileExistsError: 文件已存在且 overwrite=False
            PermissionError: 无写入权限
        """


        # 检查文件是否已存在
        if os.path.exists(file_path) and not overwrite:
            raise FileExistsError(f"目标文件已存在: {file_path}")

        # 确保目录存在
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)

        # 获取文件内容
        content = await self.get_file(uri)

        # 写入文件
        with open(file_path, "wb") as f:
            f.write(content)

        return True

    async def delete_file(self, uri: str) -> bool:
        """删除文件（可选实现）

        Args:
            uri: 文件唯一标识

        Returns:
            是否删除成功

        Raises:
            NotImplementedError: 如果适配器不支持删除
        """
        raise NotImplementedError("此适配器不支持删除文件")

    async def get_file_meta(self, uri: str) -> FileItem:
        """获取文件元数据（可选实现）

        Args:
            uri: 文件唯一标识

        Returns:
            文件元数据

        Raises:
            NotImplementedError: 如果适配器不支持获取元数据
        """
        raise NotImplementedError("此适配器不支持获取元数据")

    # 以下方法为后续扩展预留（批量获取）

    async def list_files(
        self,
        prefix: str = "",
        filter: Optional[FileFilter] = None
    ) -> AsyncIterator[FileItem]:
        """列出文件（批量获取预留）

        Args:
            prefix: 文件路径前缀
            filter: 文件过滤条件

        Yields:
            文件项

        Raises:
            NotImplementedError: 如果适配器不支持批量列出
        """
        raise NotImplementedError("此适配器不支持批量列出文件")
