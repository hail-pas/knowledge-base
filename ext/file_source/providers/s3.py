"""
AWS S3 Provider (async with aiobotocore)
"""

from typing import Any

import aiobotocore
from aiobotocore.config import AioConfig
from aiobotocore.session import AioSession
from botocore.exceptions import ClientError

from ext.file_source.base import BaseFileSourceProvider, FileMetadata
from ext.file_source.types import S3ExtraConfig
from loguru import logger


class S3FileSourceProvider(BaseFileSourceProvider[S3ExtraConfig]):
    """AWS S3 Provider (async with aiobotocore)"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        super().__init__(*args, **kwargs)
        self._session = None
        self._client = None

    @property
    async def client(self) -> Any:
        """懒加载 aiobotocore client"""
        if self._client is None:
            s3_config = {}
            if self.extra_config.addressing_style:
                s3_config["addressing_style"] = self.extra_config.addressing_style
            if self.extra_config.payload_transfer_threshold is not None:
                s3_config["payload_transfer_threshold"] = self.extra_config.payload_transfer_threshold
            if self.extra_config.signature_version:
                signature_version_map = {"s3v4": "v4", "s3v2": "v2"}
                s3_config["signature_version"] = signature_version_map.get(
                    self.extra_config.signature_version.lower(), "v4",
                )

            config = AioConfig(
                connect_timeout=self.timeout,
                read_timeout=self.timeout,
                max_pool_connections=self.max_connections,
                retries={"max_attempts": self.max_retries},
                s3=s3_config if s3_config else None,
                **(self.extra_config.config or {}),
            )

            session = AioSession()  # type: ignore[name-defined]
            self._session = session

            client_creator = session.create_client(  # type: ignore[misc]
                "s3",
                region_name=self.region or "us-east-1",
                aws_secret_access_key=self.secret_key,
                aws_access_key_id=self.access_key,
                aws_session_token=self.extra_config.session_token,
                endpoint_url=self.endpoint,
                config=config,
                use_ssl=self.use_ssl,
                verify=self.verify_ssl,
            )
            self._client = await client_creator.__aenter__()
        return self._client

    def _validate_config(self) -> None:
        if not self.access_key or not self.secret_key:
            raise ValueError("access_key and secret_key are required for S3")
        if not self.storage_location:
            raise ValueError("storage_location (bucket_name) is required for S3")

    async def validate_connection(self) -> bool:
        """验证连接"""
        try:
            client = await self.client
            # 尝试列出 bucket 中的对象作为连接验证
            await client.list_objects_v2(Bucket=self.storage_location, MaxKeys=1)
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", str(e))
            if error_code == "404":
                logger.error(f"Bucket not found: {self.storage_location}")
            elif error_code == "403":
                logger.error(f"Access denied to bucket: {self.storage_location}")
            else:
                logger.error(f"Failed to validate S3 connection: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to validate S3 connection: {e}")
            return False

    async def list_files(
        self, prefix: str = "", recursive: bool = False, limit: int | None = None,
    ) -> list[FileMetadata]:
        """列出文件"""
        client = await self.client
        paginator = client.get_paginator("list_objects_v2")
        delimiter = "" if recursive else "/"

        files = []

        paginate_kwargs = {
            "Bucket": self.storage_location,
            "Prefix": prefix,
            "Delimiter": delimiter,
        }

        async for page in paginator.paginate(**paginate_kwargs):
            for obj in page.get("Contents", []):
                files.append(
                    FileMetadata(  # type: ignore[call-arg]
                        uri=f"s3://{self.storage_location}/{obj['Key']}",
                        file_name=obj["Key"].split("/")[-1],
                        file_size=obj["Size"],
                        last_modified=obj["LastModified"],
                        etag=obj["ETag"].strip('"'),
                    ),
                )
                if limit is not None and len(files) >= limit:
                    break
            if limit is not None and len(files) >= limit:
                break

        return files

    async def get_file(self, uri: str) -> bytes:
        """获取文件内容"""
        key = self._extract_key(uri)
        client = await self.client
        response = await client.get_object(Bucket=self.storage_location, Key=key)
        return await response["Body"].read()  # type: ignore[no-any-return]

    async def get_file_stream(self, uri: str, chunk_size: int = 8192):  # type: ignore[misc]
        """获取文件流"""
        key = self._extract_key(uri)
        client = await self.client
        response = await client.get_object(Bucket=self.storage_location, Key=key)

        while True:
            chunk = await response["Body"].read(chunk_size)
            if not chunk:
                break
            yield chunk

    async def get_file_metadata(self, uri: str) -> FileMetadata:
        """获取文件元数据"""
        key = self._extract_key(uri)
        client = await self.client
        response = await client.head_object(Bucket=self.storage_location, Key=key)

        return FileMetadata(
            uri=uri,
            file_name=key.split("/")[-1],
            file_size=response["ContentLength"],
            last_modified=response["LastModified"],
            etag=response["ETag"].strip('"'),
            content_type=response.get("ContentType"),
        )

    async def file_exists(self, uri: str) -> bool:
        """检查文件是否存在"""
        try:
            await self.get_file_metadata(uri)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    async def upload_file(self, uri: str, content: bytes, content_type: str | None = None) -> FileMetadata:
        """上传文件（支持大文件分片上传）"""
        key = self._extract_key(uri)
        client = await self.client

        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type

        threshold = self.extra_config.multipart_threshold or 8388608
        chunk_size = self.extra_config.multipart_chunksize or 8388608

        if len(content) > threshold:
            logger.info(f"Using multipart upload for {key} (size: {len(content)})")
            mpu = await client.create_multipart_upload(Bucket=self.storage_location, Key=key, **extra_args)
            parts = []

            for part_num, start in enumerate(range(0, len(content), chunk_size), 1):
                end = min(start + chunk_size, len(content))
                part = await client.upload_part(
                    Bucket=self.storage_location,
                    Key=key,
                    PartNumber=part_num,
                    UploadId=mpu["UploadId"],
                    Body=content[start:end],
                )
                parts.append({"PartNumber": part_num, "ETag": part["ETag"]})

            await client.complete_multipart_upload(
                Bucket=self.storage_location,
                Key=key,
                UploadId=mpu["UploadId"],
                MultipartUpload={"Parts": parts},
            )
        else:
            await client.put_object(Bucket=self.storage_location, Key=key, Body=content, **extra_args)

        return await self.get_file_metadata(uri)

    async def delete_file(self, uri: str) -> bool:
        """删除文件"""
        try:
            key = self._extract_key(uri)
            client = await self.client
            await client.delete_object(Bucket=self.storage_location, Key=key)
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """关闭连接"""
        if self._client:
            await self._client.__aexit__(None, None, None)
            self._client = None
            self._session = None

    def _extract_key(self, uri: str) -> str:
        """从 URI 提取 object key"""
        if uri.startswith(f"s3://{self.storage_location}/"):
            return uri[len(f"s3://{self.storage_location}/") :]
        return uri
