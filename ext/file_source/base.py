"""
File Source Provider 基类

定义统一的文件操作接口和基类实现
"""

import aiofiles
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Generic, TypeVar

from datetime import datetime
from loguru import logger
from pydantic import BaseModel, Field
from ext.ext_tortoise.models.knowledge_base import FileSource
from ext.file_source.types import BaseFileSourceExtraConfig

ExtraConfigT = TypeVar("ExtraConfigT")


class FileMetadata(BaseModel):
    """统一的文件元数据"""

    uri: str = Field(..., description="文件URI")
    file_name: str = Field(..., description="文件名")
    file_size: int = Field(..., description="文件大小(bytes)")
    last_modified: datetime | None = Field(None, description="最后修改时间")
    etag: str | None = Field(None, description="ETag标识")
    content_type: str | None = Field(None, description="MIME类型")
    extra: dict = Field(default_factory=dict, description="额外元数据")


class BaseFileSourceProvider(ABC, Generic[ExtraConfigT]):
    """
    文件源 Provider 基类（泛型）

    类型参数:
        ExtraConfigT: extra_config 的具体类型（必须继承 BaseFileSourceExtraConfig）

    设计原则:
        1. 定义统一的文件操作接口
        2. 提供默认的错误处理和重试逻辑
        3. 支持异步操作
        4. 提供连接验证和健康检查
    """

    extra_config: ExtraConfigT

    def __init__(
        self,
        access_key: str | None,
        secret_key: str | None,
        endpoint: str | None,
        region: str | None,
        storage_location: str | None,
        use_ssl: bool,
        verify_ssl: bool,
        timeout: int,
        max_retries: int,
        concurrent_limit: int,
        max_connections: int,
        extra_config: dict,
    ) -> None:
        """
        初始化 Provider

        参数说明：
            - storage_location: 统一的存储位置字段
              * type=local_file: 本地路径
              * type=s3/minio/aliyun_oss: bucket 名称
        """
        self.access_key = access_key
        self.secret_key = secret_key
        self.endpoint = endpoint
        self.region = region
        self.storage_location = storage_location
        self.use_ssl = use_ssl
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.max_retries = max_retries
        self.concurrent_limit = concurrent_limit
        self.max_connections = max_connections

        extra_config = extra_config or {}
        self.extra_config: ExtraConfigT = self._convert_extra_config(extra_config)

        self._validate_config()

    def _convert_extra_config(self, extra_config_dict: dict) -> ExtraConfigT:
        """将 dict 转换成具体的 Pydantic model 类型"""
        extra_config_cls: type[BaseFileSourceExtraConfig] = self._get_extra_config_cls()  # type: ignore[assignment]
        return extra_config_cls.from_dict(extra_config_dict)  # type: ignore[return-value]

    def _get_extra_config_cls(self) -> type:
        """从泛型参数提取 extra_config 类型"""
        if hasattr(self, "__orig_bases__"):  # type: ignore[attr-defined]
            for base in self.__orig_bases__:  # type: ignore[attr-defined]
                if hasattr(base, "__args__") and base.__args__:
                    extra_config_type = base.__args__[0]
                    if isinstance(extra_config_type, type):
                        return extra_config_type

        logger.warning("无法从泛型参数提取 extra_config 类型，使用默认类型 BaseFileSourceExtraConfig")
        return BaseFileSourceExtraConfig

    def _validate_config(self) -> None:
        """验证配置（子类可覆盖）"""

    @abstractmethod
    async def validate_connection(self) -> bool:
        """验证连接是否有效"""

    @abstractmethod
    async def list_files(
        self,
        prefix: str = "",
        recursive: bool = False,
        limit: int | None = None,
    ) -> list[FileMetadata]:
        """列出文件

        Args:
            prefix: 路径前缀
            recursive: 是否递归列出
            limit: 最大返回数量

        Returns:
            文件元数据列表
        """

    @abstractmethod
    async def get_file(self, uri: str) -> bytes:
        """获取文件内容"""

    @abstractmethod
    async def get_file_stream(self, uri: str, chunk_size: int = 8192) -> AsyncIterator[bytes]:
        """获取文件流（用于大文件）"""

    @abstractmethod
    async def get_file_metadata(self, uri: str) -> FileMetadata:
        """获取文件元数据"""

    @abstractmethod
    async def file_exists(self, uri: str) -> bool:
        """检查文件是否存在"""

    async def upload_file(self, uri: str, content: bytes, content_type: str | None = None) -> FileMetadata:
        """上传文件（可选）"""
        raise NotImplementedError(f"{self.__class__.__name__} does not support upload")

    async def delete_file(self, uri: str) -> bool:
        """删除文件（可选）"""
        raise NotImplementedError(f"{self.__class__.__name__} does not support delete")

    async def upload_from_local(self, local_path: str, uri: str, content_type: str | None = None) -> FileMetadata:
        """从本地文件上传

        Args:
            local_path: 本地文件路径
            uri: 目标URI
            content_type: MIME类型

        Returns:
            上传后的文件元数据
        """

        async with aiofiles.open(local_path, "rb") as f:
            content = await f.read()
        return await self.upload_file(uri, content, content_type)

    async def download_to_local(self, uri: str, local_path: str) -> None:
        """下载文件到本地

        Args:
            uri: 源文件URI
            local_path: 本地文件路径
        """
        local_path_obj = Path(local_path)
        local_path_obj.parent.mkdir(parents=True, exist_ok=True)

        content = await self.get_file(uri)
        async with aiofiles.open(local_path, "wb") as f:
            await f.write(content)

    async def health_check(self) -> dict[str, str | bool]:
        """健康检查，返回连接状态和性能指标"""
        try:
            is_connected: bool = await self.validate_connection()
            return {
                "status": "healthy" if is_connected else "unhealthy",
                "connected": is_connected,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
