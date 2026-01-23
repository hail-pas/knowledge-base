"""
File Source Providers 初始化

自动注册所有 file source providers
"""

from loguru import logger

from ext.file_source.factory import FileSourceFactory
from ext.ext_tortoise.enums import FileSourceTypeEnum

try:
    from ext.file_source.providers.local import LocalFileSourceProvider

    FileSourceFactory.register(FileSourceTypeEnum.local_file, LocalFileSourceProvider)
    logger.info("Registered Local File Source provider")
except Exception as e:
    logger.warning(f"Failed to register Local provider: {e}")

try:
    from ext.file_source.providers.s3 import S3FileSourceProvider

    FileSourceFactory.register(FileSourceTypeEnum.s3, S3FileSourceProvider)
    logger.info("Registered S3 File Source provider")
except Exception as e:
    logger.warning(f"Failed to register S3 provider: {e}")

try:
    from ext.file_source.providers.minio import MinIOFileSourceProvider

    FileSourceFactory.register(FileSourceTypeEnum.minio, MinIOFileSourceProvider)
    logger.info("Registered MinIO File Source provider")
except Exception as e:
    logger.warning(f"Failed to register MinIO provider: {e}")

try:
    from ext.file_source.providers.aliyun_oss import AliyunOSSFileSourceProvider

    FileSourceFactory.register(FileSourceTypeEnum.aliyun_oss, AliyunOSSFileSourceProvider)
    logger.info("Registered Aliyun OSS File Source provider")
except Exception as e:
    logger.warning(f"Failed to register Aliyun OSS provider: {e}")

__all__ = ["LocalFileSourceProvider", "S3FileSourceProvider", "MinIOFileSourceProvider", "AliyunOSSFileSourceProvider"]
