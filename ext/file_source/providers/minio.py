"""
MinIO Provider
"""

from typing import Any

from minio import Minio
from minio.error import S3Error
import asyncio

from ext.file_source.base import BaseFileSourceProvider, FileMetadata
from ext.file_source.types import MinIOExtraConfig
from loguru import logger


class MinIOFileSourceProvider(BaseFileSourceProvider[MinIOExtraConfig]):
    """MinIO Provider（S3 兼容）"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        super().__init__(*args, **kwargs)
        self._client = None

    @property
    def client(self) -> Minio:
        """懒加载 MinIO client"""
        if self._client is None:
            secure = self.use_ssl
            if self.extra_config.cert_check is not None:
                secure = self.extra_config.cert_check

            self._client = Minio(  # type: ignore[arg-type,assignment]
                endpoint=self.endpoint or "",
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=secure,
                region=self.extra_config.region or self.region,
            )
        return self._client  # type: ignore[return-value]

    def _validate_config(self) -> None:
        if not self.access_key or not self.secret_key:
            raise ValueError("access_key and secret_key are required for MinIO")
        if not self.endpoint:
            raise ValueError("endpoint is required for MinIO")
        if not self.storage_location:
            raise ValueError("storage_location (bucket_name) is required for MinIO")

    async def validate_connection(self) -> bool:
        """验证连接"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: self.client.bucket_exists(self.storage_location or ""))  # type: ignore[arg-type]
            return True
        except S3Error as e:
            logger.error(f"MinIO connection failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to validate MinIO connection: {e}")
            return False

    async def list_files(
        self,
        prefix: str = "",
        recursive: bool = False,
        limit: int | None = None,
    ) -> list[FileMetadata]:
        """列出文件"""
        loop = asyncio.get_event_loop()
        objects = await loop.run_in_executor(
            None,
            lambda: self.client.list_objects(self.storage_location or "", prefix=prefix, recursive=recursive),  # type: ignore[arg-type]
        )

        files = []
        count = 0
        for obj in objects:
            if limit and count >= limit:
                break
            if obj.is_dir:
                continue

            files.append(
                FileMetadata(  # type: ignore[call-arg]
                    uri=f"minio://{self.storage_location}/{obj.object_name}",  # type: ignore[union-attr]
                    file_name=obj.object_name.split("/")[-1],  # type: ignore[union-attr]
                    file_size=obj.size,
                    last_modified=obj.last_modified,
                    etag=obj.etag,
                )
            )
            count += 1

        return files

    async def get_file(self, uri: str) -> bytes:
        """获取文件内容"""
        key = self._extract_key(uri)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: self.client.get_object(self.storage_location or "", key))  # type: ignore[arg-type]
        return response.read()

    async def get_file_stream(self, uri: str, chunk_size: int = 8192):  # type: ignore[misc]
        """获取文件流"""
        key = self._extract_key(uri)
        loop = asyncio.get_event_loop()

        def get_object():  # type: ignore[misc]
            return self.client.get_object(self.storage_location or "", key)  # type: ignore[arg-type]

        response = await loop.run_in_executor(None, get_object)

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            yield chunk

    async def get_file_metadata(self, uri: str) -> FileMetadata:
        """获取文件元数据"""
        key = self._extract_key(uri)
        loop = asyncio.get_event_loop()
        stat = await loop.run_in_executor(None, lambda: self.client.stat_object(self.storage_location or "", key))  # type: ignore[arg-type]

        return FileMetadata(  # type: ignore[call-arg]
            uri=uri,
            file_name=key.split("/")[-1],
            file_size=stat.size or 0,
            last_modified=stat.last_modified,
            etag=stat.etag,
            content_type=stat.content_type,
        )

    async def file_exists(self, uri: str) -> bool:
        """检查文件是否存在"""
        try:
            await self.get_file_metadata(uri)
            return True
        except Exception:
            return False

    async def upload_file(self, uri: str, content: bytes, content_type: str | None = None) -> FileMetadata:
        """上传文件"""
        key = self._extract_key(uri)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.client.put_object(self.storage_location or "", key, content, len(content)),  # type: ignore[arg-type]
        )
        return await self.get_file_metadata(uri)

    def _extract_key(self, uri: str) -> str:
        """从 URI 提取 object key"""
        if uri.startswith(f"minio://{self.storage_location}/"):
            return uri[len(f"minio://{self.storage_location}/") :]
        return uri
