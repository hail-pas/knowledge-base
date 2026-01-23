"""
File Source 模块

提供统一的文件存储服务抽象，支持多种文件源类型
"""

from ext.file_source.base import BaseFileSourceProvider, FileMetadata
from ext.file_source.types import (
    BaseFileSourceExtraConfig,
    LocalFileSourceExtraConfig,
    S3CompatibleExtraConfig,
    MinIOExtraConfig,
    S3ExtraConfig,
    AliyunOSSExtraConfig,
)
from ext.file_source.factory import FileSourceFactory

__all__ = [
    # Core
    "BaseFileSourceProvider",
    "FileMetadata",
    # Config types
    "BaseFileSourceExtraConfig",
    "LocalFileSourceExtraConfig",
    "S3CompatibleExtraConfig",
    "MinIOExtraConfig",
    "S3ExtraConfig",
    "AliyunOSSExtraConfig",
    # Factory
    "FileSourceFactory",
]
