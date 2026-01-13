from typing import AsyncIterator, Optional, cast

from ext.file_source.base import FileSourceAdapter, FileItem
import oss2
from oss2.crypto import RsaProvider


class AliyunOSSAdapter(FileSourceAdapter):
    """阿里云 OSS 适配器（支持客户端加密）

    支持阿里云 OSS 的客户端加密功能，使用 RSA 密钥对进行加密。

    Config 格式:
    {
        "endpoint": "https://oss-cn-shanghai.aliyuncs.com",
        "bucket": "my-bucket",
        "access_key_id": "LTAI5t...",
        "access_key_secret": "...",
        "enable_encryption": true,  # 是否启用客户端加密
        "private_key_path": "/path/to/private_key.pem",  # RSA 私钥路径
        "public_key_path": "/path/to/public_key.pem",   # RSA 公钥路径
        "mat_desc": {"key": "value"},  # 加密材料描述（可选）
    }

    使用示例:
    ```python
    config = {
        "endpoint": "https://oss-cn-shanghai.aliyuncs.com",
        "bucket": "my-bucket",
        "access_key_id": "LTAI5t...",
        "access_key_secret": "...",
        "enable_encryption": True,
        "private_key_path": "/path/to/private_key.pem",
        "public_key_path": "/path/to/public_key.pem",
    }
    adapter = AliyunOSSAdapter(config)
    ```
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.endpoint = config.get("endpoint")
        self.bucket = config.get("bucket")
        self.access_key_id = config.get("access_key_id")
        self.access_key_secret = config.get("access_key_secret")
        self.enable_encryption = config.get("enable_encryption", False)
        self.private_key_path = config.get("private_key_path")
        self.public_key_path = config.get("public_key_path")
        self.mat_desc = config.get("mat_desc", {"desc": "aliyun-oss-encryption"})
        self._bucket = None
        self._auth = None

    def _create_auth(self):
        """创建 OSS 认证对象"""
        if self._auth is None:
            self._auth = oss2.Auth(self.access_key_id, self.access_key_secret)

        return self._auth

    def _create_bucket(self):
        """创建 OSS Bucket 对象"""
        if self._bucket is None:
            auth = self._create_auth()

            if self.enable_encryption:
                # 客户端加密模式
                if not self.private_key_path or not self.public_key_path:
                    raise ValueError(
                        "启用加密时必须提供 private_key_path 和 public_key_path"
                    )

                # 读取 RSA 密钥对
                with open(self.private_key_path, "r") as f:
                    private_key = f.read()
                with open(self.public_key_path, "r") as f:
                    public_key = f.read()

                key_pair = {
                    'private_key': private_key,
                    'public_key': public_key
                }

                # 创建加密 Bucket
                crypto_provider = RsaProvider(key_pair, mat_desc=self.mat_desc)
                self._bucket = oss2.CryptoBucket(
                    auth,
                    self.endpoint,
                    self.bucket,
                    crypto_provider=crypto_provider
                )
            else:
                # 普通 Bucket
                self._bucket = oss2.Bucket(auth, self.endpoint, self.bucket)

        return self._bucket

    async def validate(self) -> bool:
        """验证配置是否正确"""
        if not self.bucket or not self.endpoint:
            return False

        try:
            bucket = self._create_bucket()
            # 尝试获取 bucket 信息
            bucket.get_bucket_info()
            return True
        except Exception:
            return False

    async def get_file(self, uri: str) -> bytes:
        """获取文件内容

        Args:
            uri: OSS 对象键

        Returns:
            文件内容
        """
        try:
            bucket = self._create_bucket()
            result = bucket.get_object(uri)
            return cast(bytes, result.read())
        except Exception as e:
            if "NoSuchKey" in str(e) or "404" in str(e):
                raise FileNotFoundError(f"OSS 对象不存在: {self.bucket}/{uri}")
            raise RuntimeError(f"获取 OSS 对象失败: {e}")

    async def upload_file(
        self,
        uri: str,
        content: bytes,
        content_type: str = "application/octet-stream",
        metadata: dict | None = None
    ) -> bool:
        """上传文件到 OSS

        Args:
            uri: OSS 对象键
            content: 文件内容
            content_type: MIME 类型
            metadata: 元数据

        Returns:
            是否上传成功
        """
        try:
            bucket = self._create_bucket()

            headers = {}
            if content_type:
                headers['Content-Type'] = content_type

            bucket.put_object(uri, content, headers=headers)

            return True
        except Exception as e:
            raise RuntimeError(f"上传文件到 OSS 失败: {e}")

    async def delete_file(self, uri: str) -> bool:
        """删除 OSS 中的文件

        Args:
            uri: OSS 对象键

        Returns:
            是否删除成功
        """
        try:
            bucket = self._create_bucket()
            bucket.delete_object(uri)
            return True
        except Exception as e:
            raise RuntimeError(f"删除 OSS 对象失败: {e}")

    async def check_file_exists(self, uri: str) -> bool:
        """检查文件是否存在

        Args:
            uri: OSS 对象键

        Returns:
            文件是否存在
        """
        try:
            bucket = self._create_bucket()
            bucket.head_object(uri)
            return True
        except Exception:
            return False

    async def get_file_meta(self, uri: str) -> FileItem:
        """获取文件元数据

        Args:
            uri: OSS 对象键

        Returns:
            文件元数据
        """
        try:
            bucket = self._create_bucket()
            result = bucket.head_object(uri)

            # 从对象键中提取文件名
            name = uri.split("/")[-1] if "/" in uri else uri

            # 获取 content_type，优先从 headers 中获取
            content_type = None
            if hasattr(result, 'content_type') and result.content_type:
                content_type = result.content_type
            elif hasattr(result, 'headers') and 'Content-Type' in result.headers:
                content_type = result.headers['Content-Type']
            else:
                content_type = "application/octet-stream"

            # 将时间戳转换为 datetime 对象
            from datetime import datetime as dt
            last_modified_dt = result.last_modified if isinstance(result.last_modified, dt) else dt.fromtimestamp(result.last_modified)

            return FileItem(
                uri=uri,
                name=name,
                size=result.content_length,
                content_type=content_type,
                last_modified=last_modified_dt,
                metadata={
                    "etag": result.etag.strip('"') if result.etag else None,
                }
            )
        except Exception as e:
            if "NoSuchKey" in str(e) or "404" in str(e):
                raise FileNotFoundError(f"OSS 对象不存在: {self.bucket}/{uri}")
            raise RuntimeError(f"获取 OSS 对象元数据失败: {e}")

    async def list_files(self, prefix: str = "", filter=None) -> AsyncIterator[FileItem]: # type: ignore
        """列出文件

        Args:
            prefix: 对象键前缀
            filter: 文件过滤条件

        Yields:
            文件项
        """
        try:
            bucket = self._create_bucket()

            for obj in oss2.ObjectIterator(bucket, prefix=prefix):
                key = obj.key

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

                # 将时间戳转换为 datetime 对象
                from datetime import datetime as dt
                last_modified_dt = obj.last_modified if isinstance(obj.last_modified, dt) else dt.fromtimestamp(obj.last_modified)

                yield FileItem(
                    uri=key,
                    name=key.split("/")[-1] if "/" in key else key,
                    size=obj.size,
                    content_type="application/octet-stream",
                    last_modified=last_modified_dt,
                    metadata={
                        "etag": obj.etag.strip('"') if obj.etag else None,
                    }
                )
        except Exception as e:
            raise RuntimeError(f"列出 OSS 对象失败: {e}")
