"""
File Source Extra Config 类型定义

定义各种 provider 的 extra_config 类型
"""

from typing import Any
from pydantic import BaseModel, Field


class BaseFileSourceExtraConfig(BaseModel):
    """
    File Source Extra Config 基础类型

    所有 provider 的 extra_config 都应继承此类
    提供通用的字段转换方法

    注意：通用字段（timeout, use_ssl, verify_ssl, region 等）已在 FileSource 表中定义
    此处只定义真正 provider 特有的配置
    """

    def to_dict(self) -> dict[str, Any]:
        """转换为字典（用于存储到数据库）"""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseFileSourceExtraConfig":
        """从字典创建实例"""
        valid_data = {k: v for k, v in data.items() if k in cls.model_fields}
        return cls.model_validate(valid_data)


class LocalFileSourceExtraConfig(BaseFileSourceExtraConfig):
    """本地文件存储特定配置"""

    allowed_extensions: list[str] | None = Field(
        default=None, description="允许的文件扩展名列表（如 ['.txt', '.pdf']）",
    )
    excluded_extensions: list[str] | None = Field(default=None, description="排除的文件扩展名列表")
    max_file_size: int | None = Field(default=None, description="最大文件大小(bytes)，None=无限制")
    follow_symlinks: bool = Field(default=False, description="是否跟随符号链接")
    require_readable: bool = Field(default=True, description="要求文件可读")


class S3CompatibleExtraConfig(BaseFileSourceExtraConfig):
    """S3兼容存储（S3/MinIO）通用配置"""

    signature_version: str | None = Field(default="s3v4", description="签名版本（s3v4/s3v2）")
    multipart_threshold: int | None = Field(default=8388608, description="启用分片上传的阈值(bytes)")
    multipart_chunksize: int | None = Field(default=8388608, description="分片大小(bytes)")
    addressing_style: str | None = Field(default="path", description="地址模式（path/virtual-hosted）")
    payload_transfer_threshold: int | None = Field(
        default=0, description="禁用 aws-chunked encoding 的阈值(bytes), 0 表示禁用",
    )


class MinIOExtraConfig(S3CompatibleExtraConfig):
    """MinIO 特定配置"""

    region: str | None = Field(default=None, description="MinIO region字符串（覆盖公共region字段）")
    cert_check: bool | None = Field(default=None, description="是否检查证书（覆盖公共use_ssl/verify_ssl字段）")


class S3ExtraConfig(S3CompatibleExtraConfig):
    """AWS S3 特定配置"""

    config: dict[str, Any] | None = Field(
        default_factory=dict, description="boto3 Config对象额外参数（如max_pool_connections等）",
    )
    session_token: str | None = Field(default=None, description="STS临时会话令牌")


class AliyunOSSExtraConfig(BaseFileSourceExtraConfig):
    """阿里云 OSS 特定配置"""

    is_encrypted_bucket: bool = Field(default=True, description="是否使用加密bucket")
    kms_key_id: str | None = Field(default=None, description="KMS密钥ID（用于SSE-KMS）")
    encryption_algorithm: str | None = Field(default="AES256", description="加密算法（AES256/KMS）")
    private_key_content: str | None = Field(default=None, description="RSA私钥内容（用于证书认证，存储为字符串）")
    public_key_content: str | None = Field(default=None, description="RSA公钥内容（用于证书认证，存储为字符串）")
    mat_desc_vendor: str | None = Field(default="kbService", description="材质描述的vendor字段（如 'kbService'）")
    enable_crc: bool = Field(default=True, description="启用CRC数据校验（推荐）")
    app_name: str | None = Field(default=None, description="自定义应用名称（用于User-Agent）")
