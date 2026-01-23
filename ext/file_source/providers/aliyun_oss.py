"""
Aliyun OSS Provider

支持 CryptoBucket 和普通 Bucket，支持 RSA 密钥对认证
"""

from typing import Any

import oss2
from oss2.crypto import RsaProvider
from oss2.exceptions import OssError, NoSuchKey
import asyncio
from loguru import logger

from ext.file_source.base import BaseFileSourceProvider, FileMetadata
from ext.file_source.types import AliyunOSSExtraConfig


class AliyunOSSFileSourceProvider(BaseFileSourceProvider[AliyunOSSExtraConfig]):
    """阿里云 OSS Provider（支持 CryptoBucket）"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        super().__init__(*args, **kwargs)
        self._bucket = None
        self._auth = None

    @property
    def auth(self):  # type: ignore[no-untyped-def]
        """懒加载认证对象"""
        if self._auth is None:
            self._auth = oss2.AuthV4(self.access_key, self.secret_key)
        return self._auth

    @property
    def bucket(self):  # type: ignore[no-untyped-def]
        """懒加载 Bucket 对象（支持加密）"""
        if self._bucket is None:
            bucket_name = self.storage_location

            if self.extra_config.private_key_content and self.extra_config.public_key_content:
                logger.info("Using CryptoBucket with RSA key pair for Aliyun OSS")

                key_pair = {
                    "private_key": self.extra_config.private_key_content,
                    "public_key": self.extra_config.public_key_content,
                }

                mat_desc = {self.extra_config.mat_desc_vendor: self.extra_config.mat_desc_vendor} if self.extra_config.mat_desc_vendor else {"kbService": "kbService"}

                crypto_provider = RsaProvider(key_pair, mat_desc=mat_desc)

                self._bucket = oss2.CryptoBucket(
                    auth=self.auth,
                    endpoint=self.endpoint,
                    bucket_name=bucket_name,
                    crypto_provider=crypto_provider,
                    region=self.region,
                    app_name=self.extra_config.app_name or "",
                    enable_crc=self.extra_config.enable_crc,
                )

            else:
                logger.info("Using regular Bucket for Aliyun OSS")

                self._bucket = oss2.Bucket(
                    auth=self.auth,
                    endpoint=self.endpoint,
                    bucket_name=bucket_name,
                    region=self.region,
                    app_name=self.extra_config.app_name or "",
                    enable_crc=self.extra_config.enable_crc,
                )

                if self.extra_config.is_encrypted_bucket:
                    if self.extra_config.kms_key_id:
                        assert self._bucket is not None
                        self._bucket.server_side_encryption = "KMS"  # type: ignore[attr-defined]
                        self._bucket.kms_key_id = self.extra_config.kms_key_id  # type: ignore[attr-defined]
                    else:
                        assert self._bucket is not None
                        self._bucket.server_side_encryption = "AES256"  # type: ignore[attr-defined]

        return self._bucket  # type: ignore[return-value]

    def _validate_config(self) -> None:
        if not self.storage_location:
            raise ValueError("storage_location (bucket name) is required for Aliyun OSS")
        if not self.endpoint:
            raise ValueError("endpoint is required for Aliyun OSS")
        if not self.region:
            raise ValueError("region is required for Aliyun OSS (V4 signature)")

        if self.extra_config.private_key_content or self.extra_config.public_key_content:
            if not self.access_key:
                raise ValueError("access_key (RAM AccessKey) is required when using RSA key pair")
            if not self.secret_key:
                raise ValueError("secret_key (RAM AccessKey Secret) is required when using RSA key pair")
        else:
            if not self.access_key or not self.secret_key:
                raise ValueError("access_key and secret_key are required for Aliyun OSS")

    async def validate_connection(self) -> bool:
        """验证连接"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: self.bucket.get_bucket_info())
            return True
        except OssError as e:
            logger.error(f"Aliyun OSS connection failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to validate Aliyun OSS connection: {e}")
            return False

    async def list_files(
        self,
        prefix: str = "",
        recursive: bool = False,
        limit: int | None = None,
    ) -> list[FileMetadata]:
        """列出文件"""
        loop = asyncio.get_event_loop()
        bucket_name = self.storage_location

        files = []
        count = 0

        def list_files_sync() -> None:  # type: ignore[misc]
            nonlocal count
            max_keys = limit if limit else 1000
            for obj in oss2.ObjectIterator(
                self.bucket, prefix=prefix, delimiter="" if recursive else "/", max_keys=max_keys
            ):
                if limit and count >= limit:
                    break

                if obj.is_prefix():
                    continue

                files.append(
                    FileMetadata(  # type: ignore[call-arg]
                        uri=f"oss://{bucket_name}/{obj.key}",
                        file_name=obj.key.split("/")[-1],
                        file_size=obj.size,
                        last_modified=obj.last_modified,
                        etag=obj.etag,
                    )
                )
                count += 1

        await loop.run_in_executor(None, list_files_sync)
        return files

    async def get_file(self, uri: str) -> bytes:
        """获取文件内容"""
        key = self._extract_key(uri)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: self.bucket.get_object(key))
        return result.read()  # type: ignore[no-any-return]

    async def get_file_stream(self, uri: str, chunk_size: int = 8192):  # type: ignore[misc]
        """获取文件流"""
        key = self._extract_key(uri)
        loop = asyncio.get_event_loop()

        result = await loop.run_in_executor(None, lambda: self.bucket.get_object(key))

        while True:
            chunk = result.read(chunk_size)
            if not chunk:
                break
            yield chunk

    async def get_file_metadata(self, uri: str) -> FileMetadata:
        """获取文件元数据"""
        key = self._extract_key(uri)
        loop = asyncio.get_event_loop()

        meta = await loop.run_in_executor(None, lambda: self.bucket.head_object(key))

        last_modified_val = meta.last_modified
        if isinstance(last_modified_val, int):
            from datetime import datetime

            last_modified_val = datetime.fromtimestamp(last_modified_val)

        return FileMetadata(
            uri=uri,
            file_name=key.split("/")[-1],
            file_size=int(meta.content_length or 0),
            last_modified=last_modified_val,  # type: ignore[arg-type]
            etag=meta.etag,
            content_type=meta.content_type,
        )

    async def file_exists(self, uri: str) -> bool:
        """检查文件是否存在"""
        try:
            await self.get_file_metadata(uri)
            return True
        except NoSuchKey:
            return False

    async def upload_file(self, uri: str, content: bytes, content_type: str | None = None) -> FileMetadata:
        """上传文件"""
        key = self._extract_key(uri)
        loop = asyncio.get_event_loop()

        headers = {}
        if content_type:
            headers["Content-Type"] = content_type

        await loop.run_in_executor(None, lambda: self.bucket.put_object(key, content, headers=headers))

        return await self.get_file_metadata(uri)

    async def delete_file(self, uri: str) -> bool:
        """删除文件"""
        try:
            key = self._extract_key(uri)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: self.bucket.delete_object(key))
            return True
        except Exception:
            return False

    def _extract_key(self, uri: str) -> str:
        """从 URI 提取 object key"""
        if uri.startswith(f"oss://{self.storage_location}/"):
            return uri[len(f"oss://{self.storage_location}/") :]
        return uri
