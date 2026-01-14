"""
LLM 模型工厂类

提供统一的接口，根据数据库配置动态创建 LLM 模型实例。
支持模型注册、实例缓存、并发安全等功能。
"""

import asyncio
from typing import Dict, Type, Optional, Any
from loguru import logger

from ext.llm.base import LLMModel, ModelCapabilities
from ext.llm.exceptions import (
    LLMConfigError,
    LLMModelNotFoundError,
)

from ext.ext_tortoise.enums import LLMModelTypeEnum
from ext.ext_tortoise.models.knowledge_base import LLMModelConfig


class LLMModelFactory:
    """LLM 模型工厂类

    负责根据数据库配置动态创建 LLM 模型实例。
    支持多种 LLM 服务提供商，提供统一的接口。

    使用示例:
        >>> config = await LLMModelConfig.get(name="gpt-4o-config")
        >>> model = await LLMModelFactory.create(config)
        >>> pydantic_model = model.get_model_for_agent()
        >>> agent = Agent(pydantic_model, result_type=str)
    """

    # 模型类型到实现类的映射
    _models: Dict[LLMModelTypeEnum, Type[LLMModel]] = {}

    # 模型实例缓存
    _instances: Dict[int, LLMModel] = {}

    # 锁，用于防止并发创建同一实例
    _locks: Dict[int, asyncio.Lock] = {}

    @classmethod
    def register(cls, model_type: LLMModelTypeEnum, model_class: Type[LLMModel]) -> None:
        """注册新的 LLM 模型类型

        Args:
            model_type: 模型类型标识（如 "openai", "azure_openai", "deepseek"）
            model_class: 实现 LLMModel 的类

        Raises:
            ValueError: 如果模型类型已注册

        Example:
            >>> LLMModelFactory.register(LLMModelTypeEnum.openai, OpenAIModel)
        """
        if model_type in cls._models:
            import warnings
            warnings.warn(
                f"模型类型 {model_type.value} 已注册，将被覆盖",
                stacklevel=2
            )
        cls._models[model_type] = model_class

    @classmethod
    async def create(
        cls,
        config: LLMModelConfig,
        use_cache: bool = True,
        validate: bool = True
    ) -> LLMModel:
        """创建 LLM 模型实例

        Args:
            config: LLMModelConfig 数据库实例（必须已保存到数据库）
            use_cache: 是否使用缓存
            validate: 是否验证模型配置和能力

        Returns:
            LLMModel 实例

        Raises:
            LLMConfigError: 配置错误或模型未启用
            LLMModelNotFoundError: 不支持的模型类型

        Example:
            >>> config = await LLMModelConfig.filter(name="gpt-4o").first()
            >>> model = await LLMModelFactory.create(config)
        """
        # 验证配置是否启用
        if not config.is_enabled:
            raise LLMConfigError(
                f"LLM 模型配置未启用。"
                f"配置: {config.name} (id={config.id})"
            )

        # 获取模型类
        model_cls = cls._models.get(config.type)
        if not model_cls:
            available_types = ", ".join([t.value for t in cls._models.keys()])
            raise LLMModelNotFoundError(
                f"不支持的模型类型: {config.type.value}, "
                f"可用类型: {available_types}"
            )

        # 解析能力配置
        capabilities_data = config.capabilities or {}
        capabilities = ModelCapabilities.from_dict(capabilities_data)

        # 如果不使用缓存，直接创建新实例
        # 临时对象（未保存到数据库）不使用缓存
        if not use_cache or not config._saved_in_db:
            model = cls._create_instance(
                model_cls,
                config,
                capabilities,
                validate
            )
            return model

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
            model = cls._create_instance(
                model_cls,
                config,
                capabilities,
                validate
            )
            cls._instances[config.id] = model

            return model

    @classmethod
    def _create_instance(
        cls,
        model_cls: Type[LLMModel],
        config: LLMModelConfig,
        capabilities: ModelCapabilities,
        validate: bool
    ) -> LLMModel:
        """创建模型实例的内部方法

        Args:
            model_cls: 模型类
            config: 配置对象
            capabilities: 能力配置
            validate: 是否验证

        Returns:
            LLMModel 实例
        """
        # 创建模型实例
        model = model_cls(
            model_name=config.model_name,
            model_type=config.type.value,
            config=config.config,
            max_tokens=config.max_tokens,
            capabilities=capabilities,
            max_retries=config.max_retries,
            timeout=config.timeout,
            rate_limit=config.rate_limit,
        )

        # 可选验证
        if validate:
            # 验证基本配置
            model.validate_config(cls._get_required_keys(config.type))

        return model

    @classmethod
    def _get_required_keys(cls, model_type: LLMModelTypeEnum) -> list[str]:
        """获取特定模型类型所需的配置键

        Args:
            model_type: 模型类型

        Returns:
            必需的配置键列表
        """
        # 根据模型类型返回不同的必需配置
        if model_type == LLMModelTypeEnum.openai:
            return ["api_key"]
        elif model_type == LLMModelTypeEnum.azure_openai:
            return ["api_key", "endpoint"]
        elif model_type == LLMModelTypeEnum.deepseek:
            return ["api_key"]
        else:
            # 默认至少需要 api_key
            return ["api_key"]

    @classmethod
    def clear_cache(cls, config_id: Optional[int] = None) -> None:
        """清除模型实例缓存

        Args:
            config_id: 要清除的配置 ID，如果为 None 则清除所有缓存

        Example:
            >>> # 清除单个配置的缓存
            >>> LLMModelFactory.clear_cache(config_id=1)
            >>> # 清除所有缓存
            >>> LLMModelFactory.clear_cache()
        """
        if config_id is None:
            cls._instances.clear()
            cls._locks.clear()
        else:
            cls._instances.pop(config_id, None)
            cls._locks.pop(config_id, None)

    @classmethod
    def has_model(cls, model_type: LLMModelTypeEnum) -> bool:
        """检查模型类型是否已注册

        Args:
            model_type: 模型类型

        Returns:
            是否已注册

        Example:
            >>> LLMModelFactory.has_model(LLMModelTypeEnum
.openai)
            True
        """
        return model_type in cls._models

    @classmethod
    def get_registered_model_types(cls) -> list[LLMModelTypeEnum]:
        """获取所有已注册的模型类型

        Returns:
            已注册模型类型列表

        Example:
            >>> types = LLMModelFactory.get_registered_model_types()
            >>> [t.value for t in types]
            ['openai', 'azure_openai', 'deepseek']
        """
        return list(cls._models.keys())

    @classmethod
    async def create_by_name(
        cls,
        name: str,
        use_cache: bool = True,
        validate: bool = True
    ) -> LLMModel:
        """根据配置名称创建模型实例

        Args:
            name: 配置名称
            use_cache: 是否使用缓存
            validate: 是否验证

        Returns:
            LLMModel 实例

        Raises:
            LLMModelNotFoundError: 配置不存在

        Example:
            >>> model = await LLMModelFactory.create_by_name("gpt-4o-config")
        """
        from ext.ext_tortoise.models.knowledge_base import LLMModelConfig

        config = await LLMModelConfig.filter(name=name, is_enabled=True).first()
        if not config:
            raise LLMModelNotFoundError(
                f"未找到名称为 '{name}' 的已启用 LLM 配置"
            )

        return await cls.create(config, use_cache=use_cache, validate=validate)

    @classmethod
    async def create_default(
        cls,
        use_cache: bool = True,
        validate: bool = True
    ) -> LLMModel:
        """创建默认的 LLM 模型实例

        Args:
            use_cache: 是否使用缓存
            validate: 是否验证

        Returns:
            LLMModel 实例

        Raises:
            LLMModelNotFoundError: 没有配置默认模型

        Example:
            >>> model = await LLMModelFactory.create_default()
        """
        from ext.ext_tortoise.models.knowledge_base import LLMModelConfig

        config = await LLMModelConfig.filter(
            is_enabled=True,
            is_default=True
        ).first()

        if not config:
            # 如果没有默认配置，尝试获取第一个启用的配置
            config = await LLMModelConfig.filter(
                is_enabled=True
            ).first()

        if not config:
            raise LLMModelNotFoundError(
                "未找到可用的 LLM 模型配置"
            )

        return await cls.create(config, use_cache=use_cache, validate=validate)

    @classmethod
    async def create_by_capability(
        cls,
        capability: str,
        use_cache: bool = True,
        validate: bool = True
    ) -> LLMModel:
        """根据能力要求创建模型实例

        Args:
            capability: 需要的能力（如 "function_calling", "vision"）
            use_cache: 是否使用缓存
            validate: 是否验证

        Returns:
            LLMModel 实例

        Raises:
            LLMModelNotFoundError: 没有支持该能力的模型

        Example:
            >>> model = await LLMModelFactory.create_by_capability("function_calling")
        """
        from ext.ext_tortoise.models.knowledge_base import LLMModelConfig

        # 查询支持该能力的配置
        configs = await LLMModelConfig.filter(
            is_enabled=True
        ).all()

        for config in configs:
            capabilities_data = config.capabilities or {}
            if capabilities_data.get(capability, False):
                return await cls.create(config, use_cache=use_cache, validate=validate)

        raise LLMModelNotFoundError(
            f"未找到支持 '{capability}' 能力的 LLM 模型配置"
        )

    @classmethod
    def get_cache_info(cls) -> Dict[str, Any]:
        """获取缓存信息

        Returns:
            缓存信息字典

        Example:
            >>> info = LLMModelFactory.get_cache_info()
            >>> info
            {'cached_count': 3, 'cached_ids': [1, 2, 3]}
        """
        return {
            "cached_count": len(cls._instances),
            "cached_ids": list(cls._instances.keys()),
            "registered_models": [t.value for t in cls._models.keys()],
        }

    @classmethod
    async def close_all(cls) -> None:
        """关闭所有缓存的模型实例

        用于应用关闭时清理资源

        Example:
            >>> await LLMModelFactory.close_all()
        """
        for model in cls._instances.values():
            try:
                await model.close()
            except Exception as e:
                # 记录错误但继续清理其他模型
                logger.error(f"关闭模型实例时出错: {e}", exc_info=True)

        cls.clear_cache()


# 在模块加载时自动注册已知的模型提供者
# 这些导入放在这里是为了确保在导入模块时就完成注册
try:
    from ext.llm.providers.openai import OpenAIModelWrapper
    LLMModelFactory.register(LLMModelTypeEnum.openai, OpenAIModelWrapper)
except ImportError:
    pass

try:
    from ext.llm.providers.azure_openai import AzureOpenAIModelWrapper
    LLMModelFactory.register(LLMModelTypeEnum.azure_openai, AzureOpenAIModelWrapper)
except ImportError:
    pass

try:
    from ext.llm.providers.deepseek import DeepSeekModelWrapper
    LLMModelFactory.register(LLMModelTypeEnum.deepseek, DeepSeekModelWrapper)
except ImportError:
    pass
