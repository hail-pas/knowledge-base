"""
FileSource Factory - 文件源适配器工厂类
"""

from typing import Dict, Type, Optional
import asyncio

from ext.file_source.base import FileSourceAdapter
from ext.file_source.exceptions import (
    FileSourceConfigError,
    FileSourceTypeError,
)

from ext.ext_tortoise.enums import FileSourceTypeEnum
from ext.ext_tortoise.models.knowledge_base import FileSource


class FileSourceAdapterFactory:
    """文件源适配器工厂类

    管理不同文件源类型（Local、S3、OSS、SharePoint等）的适配器实例创建和缓存。
    """

    # 文件源类型到适配器类的映射
    _adapters: Dict[FileSourceTypeEnum, Type[FileSourceAdapter]] = {}

    # 适配器实例缓存（source_id -> Adapter 实例）
    _instances: Dict[int, FileSourceAdapter] = {}

    # 锁，用于防止并发创建同一实例
    _locks: Dict[int, asyncio.Lock] = {}

    @classmethod
    def register(cls, source_type: FileSourceTypeEnum, adapter_class: Type[FileSourceAdapter]) -> None:
        """注册新的文件源适配器类型

        Args:
            source_type: 文件源类型标识（如 local_file, s3, aliyun_oss）
            adapter_class: 实现 FileSourceAdapter 的类
        """
        cls._adapters[source_type] = adapter_class

    @classmethod
    async def create(cls, source: FileSource, use_cache: bool = True) -> FileSourceAdapter:
        """创建适配器实例

        Args:
            source: FileSource 数据库实例（必须已保存到数据库）
            use_cache: 是否使用缓存

        Returns:
            FileSourceAdapter 实例

        Raises:
            FileSourceConfigError: 配置无效或未启用
            FileSourceTypeError: 不支持的文件源类型或创建失败
        """
        # 验证配置是否启用
        if not source.is_enabled:
            raise FileSourceConfigError(
                f"File source config is disabled. "
                f"Config: {source.name} (id={source.id})"
            )

        # 获取适配器类
        adapter_cls = cls._adapters.get(source.type)
        if not adapter_cls:
            available_types = ", ".join([t.value for t in cls._adapters.keys()])
            raise FileSourceTypeError(
                f"不支持的文件源类型: {source.type.value}, "
                f"可用类型: {available_types}"
            )

        # 如果不使用缓存或临时对象，直接创建新实例
        if not use_cache or not source._saved_in_db:
            adapter = adapter_cls(source.config)
            return adapter

        # 检查缓存
        if source.id in cls._instances:
            return cls._instances[source.id]

        # 获取或创建锁
        if source.id not in cls._locks:
            cls._locks[source.id] = asyncio.Lock()

        # 使用锁防止并发创建
        async with cls._locks[source.id]:
            # 再次检查缓存（可能在等待锁时已被其他协程创建）
            if source.id in cls._instances:
                return cls._instances[source.id]

            # 创建新实例并缓存
            adapter = adapter_cls(source.config)
            cls._instances[source.id] = adapter
            return adapter

    @classmethod
    def clear_cache(cls, source_id: Optional[int] = None) -> None:
        """清除适配器实例缓存

        Args:
            source_id: 要清除的文件源 ID，如果为 None 则清除所有缓存
        """
        if source_id is None:
            cls._instances.clear()
            cls._locks.clear()
        else:
            # 清除缓存前先处理（如果实例存在）
            instance = cls._instances.pop(source_id, None)
            if instance is not None:
                # 这里只是从缓存中移除，实际的连接清理由用户管理
                pass
            cls._locks.pop(source_id, None)

    @classmethod
    def has_adapter(cls, source_type: FileSourceTypeEnum) -> bool:
        """检查文件源类型是否已注册

        Args:
            source_type: 文件源类型

        Returns:
            是否已注册
        """
        return source_type in cls._adapters

    @classmethod
    def get_registered_source_types(cls) -> list[FileSourceTypeEnum]:
        """获取所有已注册的文件源类型

        Returns:
            已注册文件源类型列表
        """
        return list(cls._adapters.keys())

    @classmethod
    async def get_default_adapter(cls) -> Optional[FileSourceAdapter]:
        """获取默认的适配器实例

        Returns:
            默认的适配器实例，如果没有默认配置则返回 None

        Raises:
            FileSourceConfigError: 默认配置无效
            FileSourceTypeError: 创建失败
        """
        # 查找默认配置
        default_source = await FileSource.filter(
            is_enabled=True,
            is_default=True
        ).first()

        if default_source is None:
            return None

        return await cls.create(default_source)


# 注册内置适配器（导入时自动注册）
from ext.file_source.providers.local import LocalAdapter
from ext.file_source.providers.s3 import S3Adapter
from ext.file_source.providers.aliyun_oss import AliyunOSSAdapter
from ext.file_source.providers.sharepoint import SharePointAdapter

FileSourceAdapterFactory.register(FileSourceTypeEnum.local_file, LocalAdapter)
FileSourceAdapterFactory.register(FileSourceTypeEnum.s3, S3Adapter)
FileSourceAdapterFactory.register(FileSourceTypeEnum.aliyun_oss, AliyunOSSAdapter)
FileSourceAdapterFactory.register(FileSourceTypeEnum.sharepoint, SharePointAdapter)
# 后续扩展：
# FileSourceAdapterFactory.register(FileSourceTypeEnum.api, APIAdapter)
