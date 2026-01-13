from typing import AsyncIterator, Optional
from datetime import datetime
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError
from botocore.config import Config

from ext.file_source.base import FileSourceAdapter, FileItem


class S3Adapter(FileSourceAdapter):
    """S3/MinIO 适配器

    支持 AWS S3 及兼容 S3 API 的存储服务（如 MinIO、阿里云 OSS 等）。

    Config 格式:
    {
        "endpoint_url": "https://s3.amazonaws.com",  # 可选，AWS默认可省略，MinIO必填
        "region": "us-east-1",                       # 可选，默认 us-east-1
        "bucket": "my-bucket",                       # 必填
        "access_key": "",        # 必填
        "secret_key": "",  # 必填
        "session_token": "optional-session-token",   # 可选，临时凭证
    }

    MinIO 配置示例:
    {
        "endpoint_url": "http://localhost:9000",
        "region": "us-east-1",
        "bucket": "rag-documents",
        "access_key": "minioadmin",
        "secret_key": "minioadmin",
    }
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.endpoint_url = config.get("endpoint_url")
        self.region = config.get("region", "us-east-1")
        self.bucket = config.get("bucket")
        self.access_key = config.get("access_key")
        self.secret_key = config.get("secret_key")
        self.session_token = config.get("session_token")
        self._s3_client = None
        self._s3_resource = None

    def _create_s3_client(self):
        """创建 S3 客户端"""
        if self._s3_client is None:

            # 配置 S3 客户端，兼容阿里云 OSS、MinIO 等 S3 兼容存储
            config_params = {
                "region_name": self.region,
            }

            if self.endpoint_url:
                config_params["endpoint_url"] = self.endpoint_url

            # 阿里云 OSS 配置要求：
            # 1. 使用虚拟托管样式（virtual-hosted-style）
            # 2. 使用 v4 签名版本
            # 3. 禁用请求 checksum 计算以避免 aws-chunked 编码问题
            s3_config = Config(
                s3={"addressing_style": "virtual"},
                signature_version="s3v4",
                request_checksum_calculation="when_required",  # 只在必要时计算 checksum
            )

            self._s3_client = boto3.client(
                "s3",
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                aws_session_token=self.session_token,
                config=s3_config,
                **config_params
            )

        return self._s3_client

    # def _create_s3_resource(self):
    #     """创建 S3 资源对象"""
    #     if self._s3_resource is None:
    #         config_params = {
    #             "region_name": self.region,
    #         }

    #         if self.endpoint_url:
    #             config_params["endpoint_url"] = self.endpoint_url

    #         self._s3_resource = boto3.resource(
    #             "s3",
    #             aws_access_key_id=self.access_key,
    #             aws_secret_access_key=self.secret_key,
    #             aws_session_token=self.session_token,
    #             **config_params
    #         )

    #     return self._s3_resource

    async def validate(self) -> bool:
        """验证配置是否正确

        检查 bucket 是否存在且有访问权限
        """
        if not self.bucket:
            return False

        try:
            client = self._create_s3_client()
            client.head_bucket(Bucket=self.bucket)
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            # 404 = NoSuchBucket, 403 = AccessDenied
            if error_code in ["404", "NoSuchBucket", "403", "AccessDenied"]:
                return False
            return False
        except Exception as e:
            return False

    async def get_file(self, uri: str) -> bytes:
        """获取单文件内容

        Args:
            uri: S3 对象键（object key）
                  例如: "documents/manuals/API.pdf" 或 "folder/file.txt"

        Returns:
            文件内容（字节数组）

        Raises:
            FileNotFoundError: 文件不存在
            PermissionError: 无访问权限
        """
        try:
            client = self._create_s3_client()
            response = client.get_object(Bucket=self.bucket, Key=uri)
            return response["Body"].read()
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "NoSuchKey":
                raise FileNotFoundError(f"S3 对象不存在: {self.bucket}/{uri}")
            elif error_code == "AccessDenied":
                raise PermissionError(f"无权限访问 S3 对象: {self.bucket}/{uri}")
            raise RuntimeError(f"获取 S3 对象失败: {e}")

    async def get_file_stream(self, uri: str) -> AsyncIterator[bytes]: # type: ignore
        """获取文件流（大文件）

        Args:
            uri: S3 对象键

        Yields:
            文件内容分块（8KB）
        """
        try:
            client = self._create_s3_client()
            response = client.get_object(Bucket=self.bucket, Key=uri)

            for chunk in response["Body"].iter_chunks(chunk_size=8192):
                yield chunk
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "NoSuchKey":
                raise FileNotFoundError(f"S3 对象不存在: {self.bucket}/{uri}")
            elif error_code == "AccessDenied":
                raise PermissionError(f"无权限访问 S3 对象: {self.bucket}/{uri}")
            raise RuntimeError(f"获取 S3 对象失败: {e}")

    async def get_file_meta(self, uri: str) -> FileItem:
        """获取文件元数据

        Args:
            uri: S3 对象键

        Returns:
            文件元数据

        Raises:
            FileNotFoundError: 文件不存在
        """
        try:
            client = self._create_s3_client()
            response = client.head_object(Bucket=self.bucket, Key=uri)

            metadata = response.get("Metadata", {})
            content_type = response.get("ContentType", "application/octet-stream")
            last_modified = response.get("LastModified")
            size = response.get("ContentLength", 0)

            # 从对象键中提取文件名
            name = uri.split("/")[-1] if "/" in uri else uri

            return FileItem(
                uri=uri,
                name=name,
                size=size,
                content_type=content_type,
                last_modified=last_modified,
                metadata={
                    "etag": response.get("ETag", "").strip('"'),
                    "storage_class": response.get("StorageClass"),
                    "user_metadata": metadata,
                }
            )
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "404" or error_code == "NoSuchKey":
                raise FileNotFoundError(f"S3 对象不存在: {self.bucket}/{uri}")
            raise RuntimeError(f"获取 S3 对象元数据失败: {e}")

    async def list_files(self, prefix: str = "", filter=None) -> AsyncIterator[FileItem]: # type: ignore
        """列出文件（批量获取）

        Args:
            prefix: 对象键前缀，用于过滤文件夹
                    例如: "documents/manuals/" 只列出该文件夹下的文件
            filter: 文件过滤条件（暂未实现）

        Yields:
            文件项
        """
        try:
            client = self._create_s3_client()
            paginator = client.get_paginator("list_objects_v2")

            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                contents = page.get("Contents", [])
                for obj in contents:
                    key = obj["Key"]
                    # 跳过以 "/" 结尾的文件夹标记
                    if key.endswith("/"):
                        continue

                    # 应用文件扩展名过滤
                    if filter and filter.allowed_extensions:
                        if not any(key.endswith(ext) for ext in filter.allowed_extensions):
                            continue

                    if filter and filter.blocked_extensions:
                        if any(key.endswith(ext) for ext in filter.blocked_extensions):
                            continue

                    yield FileItem(
                        uri=key,
                        name=key.split("/")[-1] if "/" in key else key,
                        size=obj.get("Size", 0),
                        content_type="application/octet-stream",  # 列表时不返回内容类型
                        last_modified=obj.get("LastModified"),
                        metadata={
                            "etag": obj.get("ETag", "").strip('"'),
                            "storage_class": obj.get("StorageClass"),
                        }
                    )
        except ClientError as e:
            raise RuntimeError(f"列出 S3 对象失败: {e}")

    async def check_file_exists(self, uri: str) -> bool:
        """检查文件是否存在

        Args:
            uri: S3 对象键

        Returns:
            文件是否存在
        """
        try:
            client = self._create_s3_client()
            client.head_object(Bucket=self.bucket, Key=uri)
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ["404", "NoSuchKey"]:
                return False
            raise
        except Exception:
            return False

    async def upload_file(
        self,
        uri: str,
        content: bytes,
        content_type: str = "application/octet-stream",
        metadata: dict | None = None
    ) -> bool:
        """上传文件到 S3

        Args:
            uri: S3 对象键（上传后的目标路径）
            content: 文件内容
            content_type: MIME 类型
            metadata: 元数据

        Returns:
            是否上传成功
        """
        try:
            client = self._create_s3_client()

            # 准备上传参数
            params = {
                "Bucket": self.bucket,
                "Key": uri,
                "Body": content,
                "ContentType": content_type,
            }

            # 添加元数据
            if metadata:
                params["Metadata"] = metadata

            client.put_object(**params)
            return True

        except ClientError as e:
            raise RuntimeError(f"上传文件到 S3 失败: {e}")

    async def delete_file(self, uri: str) -> bool:
        """删除 S3 中的文件

        Args:
            uri: S3 对象键

        Returns:
            是否删除成功
        """
        try:
            client = self._create_s3_client()
            client.delete_object(Bucket=self.bucket, Key=uri)
            return True
        except ClientError as e:
            raise RuntimeError(f"删除 S3 对象失败: {e}")

    async def get_presigned_url(
        self,
        uri: str,
        expiration: int = 3600
    ) -> str:
        """生成预签名 URL（用于直接访问或下载）

        Args:
            uri: S3 对象键
            expiration: URL 有效期（秒），默认 1 小时

        Returns:
            预签名 URL
        """
        try:
            client = self._create_s3_client()
            return client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": uri},
                ExpiresIn=expiration,
            )
        except Exception as e:
            raise RuntimeError(f"生成预签名 URL 失败: {e}")
