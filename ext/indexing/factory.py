"""
Provider Factory - 索引后端 Provider 工厂类
"""

from typing import Dict, Type, Optional
import asyncio

from ext.indexing.base import BaseProvider
from ext.indexing.exceptions import (
    IndexingConfigError,
    IndexingBackendError,
)

from ext.ext_tortoise.enums import IndexingBackendTypeEnum
from ext.ext_tortoise.models.knowledge_base import IndexingBackendConfig


class ProviderFactory:
    """索引后端 Provider 工厂类

    管理不同后端（Elasticsearch、Milvus等）的 Provider 实例创建和缓存。
    """

    # 后端类型到 Provider 类的映射
    _providers: Dict[IndexingBackendTypeEnum, Type[BaseProvider]] = {}

    # Provider 实例缓存（config_id -> Provider 实例）
    _instances: Dict[int, BaseProvider] = {}

    # 锁，用于防止并发创建同一实例
    _locks: Dict[int, asyncio.Lock] = {}

    @classmethod
    def register(cls, backend_type: IndexingBackendTypeEnum, provider_class: Type[BaseProvider]) -> None:
        """注册新的 Provider 类型

        Args:
            backend_type: 后端类型标识（如 elasticsearch, milvus）
            provider_class: 实现 BaseProvider 的类
        """
        cls._providers[backend_type] = provider_class

    @classmethod
    async def create(cls, config: IndexingBackendConfig, use_cache: bool = True) -> BaseProvider:
        """创建 Provider 实例

        Args:
            config: IndexingBackendConfig 数据库实例（必须已保存到数据库）
            use_cache: 是否使用缓存

        Returns:
            BaseProvider 实例

        Raises:
            IndexingConfigError: 配置无效或未启用
            IndexingBackendError: 不支持的后端类型或创建失败
        """
        # 验证配置是否启用
        if not config.is_enabled:
            raise IndexingConfigError(
                f"Indexing backend config is disabled. "
                f"Config: {config.name} (id={config.id})"
            )

        # 获取 Provider 类
        provider_cls = cls._providers.get(config.type)
        if not provider_cls:
            available_types = ", ".join([t.value for t in cls._providers.keys()])
            raise IndexingBackendError(
                f"不支持的索引后端类型: {config.type.value}, "
                f"可用类型: {available_types}"
            )

        # 构建 provider 配置字典
        provider_config = cls._build_provider_config(config)

        # 如果不使用缓存或临时对象，直接创建新实例
        if not use_cache or not config._saved_in_db:
            provider = provider_cls(backend_type=config.type, config=provider_config)
            return provider

        # 检查缓存
        if config.id in cls._instances:
            return cls._instances[config.id]

        # 获取或创建锁
        if config.id not in cls._locks:
            cls._locks[config.id] = asyncio.Lock()

        # 使用锁防止并发创建
        async with cls._locks[config.id]:
            # 再次检查缓存（可能在等待锁时已被其他协程创建）
            if config.id in cls._instances:
                return cls._instances[config.id]

            # 创建新实例并缓存
            # Provider 内部会自动管理连接，这里只负责创建
            provider = provider_cls(backend_type=config.type, config=provider_config)
            cls._instances[config.id] = provider
            return provider

    @classmethod
    def _build_provider_config(cls, config: IndexingBackendConfig) -> Dict[str, any]:
        """构建 Provider 配置字典

        将数据库配置对象转换为 Provider 需要的配置字典。

        Args:
            config: IndexingBackendConfig 数据库实例

        Returns:
            Provider 配置字典
        """
        provider_config = {
            "name": config.name,
            "host": config.host,
            "port": config.port,
            "secure": config.secure,
        }

        # 可选字段
        if config.username:
            provider_config["username"] = config.username
        if config.password:
            provider_config["password"] = config.password
        if config.api_key:
            provider_config["api_key"] = config.api_key

        # 合并额外的配置参数
        if config.config:
            provider_config.update(config.config)

        return provider_config

    @classmethod
    def clear_cache(cls, config_id: Optional[int] = None) -> None:
        """清除 Provider 实例缓存

        Args:
            config_id: 要清除的配置 ID，如果为 None 则清除所有缓存
        """
        if config_id is None:
            cls._instances.clear()
            cls._locks.clear()
        else:
            # 清除缓存前先断开连接（如果实例存在）
            instance = cls._instances.pop(config_id, None)
            if instance is not None:
                # 异步断开连接需要在事件循环中执行
                # 这里只是从缓存中移除，实际的连接清理由用户管理
                pass
            cls._locks.pop(config_id, None)

    @classmethod
    def has_provider(cls, backend_type: IndexingBackendTypeEnum) -> bool:
        """检查后端类型是否已注册

        Args:
            backend_type: 后端类型

        Returns:
            是否已注册
        """
        return backend_type in cls._providers

    @classmethod
    def get_registered_backend_types(cls) -> list[IndexingBackendTypeEnum]:
        """获取所有已注册的后端类型

        Returns:
            已注册后端类型列表
        """
        return list(cls._providers.keys())

    @classmethod
    async def get_default_provider(cls) -> Optional[BaseProvider]:
        """获取默认的 Provider 实例

        Returns:
            默认的 Provider 实例，如果没有默认配置则返回 None

        Raises:
            IndexingConfigError: 默认配置无效
            IndexingBackendError: 创建失败
        """
        from ext.ext_tortoise.models.knowledge_base import IndexingBackendConfig

        # 查找默认配置
        default_config = await IndexingBackendConfig.filter(
            is_enabled=True,
            is_default=True
        ).first()

        if default_config is None:
            return None

        return await cls.create(default_config)


# 注册 provider
from ext.indexing.providers import ElasticsearchProvider, MilvusProvider

ProviderFactory.register(IndexingBackendTypeEnum.elasticsearch, ElasticsearchProvider)
ProviderFactory.register(IndexingBackendTypeEnum.milvus, MilvusProvider)
