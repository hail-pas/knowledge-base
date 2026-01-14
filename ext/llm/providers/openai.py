"""
OpenAI LLM 模型实现

使用 pydantic_ai.models 封装 OpenAI API，提供 OpenAI 模型的动态创建接口。
"""

from typing import Any, Dict
from loguru import logger

from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel

from ext.llm.base import LLMModel, ModelCapabilities
from ext.llm.exceptions import LLMConfigError


class OpenAIModelWrapper(LLMModel):
    """OpenAI 模型实现

    封装 pydantic_ai.models.OpenAIModel，支持 OpenAI API 的所有功能。

    支持的模型:
        - GPT-4o: gpt-4o, gpt-4o-mini
        - GPT-4 Turbo: gpt-4-turbo, gpt-4-turbo-2024-04-09
        - GPT-3.5 Turbo: gpt-3.5-turbo, gpt-3.5-turbo-0125
        - o1 系列: o1-preview, o1-mini
    """

    # OpenAI 模型的默认能力配置
    DEFAULT_CAPABILITIES = {
        "function_calling": True,
        "json_output": True,
        "multimodal": True,
        "streaming": True,
        "vision": True,
        "audio_input": False,
        "audio_output": False,
        "tools": True,
        "structured_output": True,
    }

    # 支持多模态的模型列表
    MULTIMODAL_MODELS = {
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4-vision-preview",
    }

    def __init__(
        self,
        model_name: str,
        model_type: str,
        config: Dict[str, Any],
        max_tokens: int = 4096,
        capabilities: ModelCapabilities | None = None,
        max_retries: int = 3,
        timeout: int = 60,
        rate_limit: int = 60,
    ):
        """
        初始化 OpenAI 模型

        Args:
            model_name: 模型名称（如 gpt-4o, gpt-3.5-turbo）
            model_type: 模型类型（固定为 "openai"）
            config: 配置参数，需要包含:
                - api_key: OpenAI API 密钥
                - base_url: (可选) 自定义 API 基础 URL
                - organization: (可选) 组织 ID
            max_tokens: 最大输出 token 数
            capabilities: 模型能力配置
            max_retries: 最大重试次数
            timeout: 请求超时时间（秒）
            rate_limit: 每分钟最大请求次数
        """
        # 根据模型名称自动调整能力配置
        if capabilities is None:
            capabilities_data = self._auto_detect_capabilities(model_name, config)
            capabilities = ModelCapabilities.from_dict(capabilities_data)

        super().__init__(
            model_name=model_name,
            model_type=model_type,
            config=config,
            max_tokens=max_tokens,
            capabilities=capabilities,
            max_retries=max_retries,
            timeout=timeout,
            rate_limit=rate_limit,
        )

        # 验证必需的配置
        self.validate_config(["api_key"])

        # 根据模型名称调整默认的 max_tokens
        self._adjust_max_tokens_for_model()

    def _auto_detect_capabilities(
        self,
        model_name: str,
        config: Dict[str, Any]
    ) -> Dict[str, bool]:
        """
        根据模型名称自动检测能力

        Args:
            model_name: 模型名称
            config: 用户配置

        Returns:
            能力配置字典
        """
        # 如果用户在 config 中提供了 capabilities，优先使用
        if "capabilities" in config and config["capabilities"]:
            return config["capabilities"]

        # 否则根据模型名称推断能力
        capabilities = self.DEFAULT_CAPABILITIES.copy()

        # 检查是否是多模态模型
        model_lower = model_name.lower()
        is_multimodal = any(
            model_lower.startswith(prefix)
            for prefix in ["gpt-4o", "gpt-4-vision", "gpt-4-turbo"]
        )

        capabilities["multimodal"] = is_multimodal
        capabilities["vision"] = is_multimodal

        # 检查是否支持 function calling
        # 大多数 GPT 模型都支持，除了一些旧模型
        if model_name.startswith("gpt-3.5-turbo-0301"):
            capabilities["function_calling"] = False
            capabilities["tools"] = False
            capabilities["structured_output"] = False

        return capabilities

    def _adjust_max_tokens_for_model(self) -> None:
        """根据模型限制调整 max_tokens"""
        # OpenAI 不同模型的 token 限制
        model_limits = {
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-4-turbo": 128000,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-3.5-turbo": 16385,
            "gpt-3.5-turbo-16k": 16385,
            "o1-preview": 128000,
            "o1-mini": 128000,
        }

        # 获取模型的上下文限制
        context_limit = model_limits.get(
            self.model_name,
            model_limits.get("gpt-4o", 128000)  # 默认使用 GPT-4o 的限制
        )

        # 如果配置的 max_tokens 超过模型限制，进行警告和调整
        if self.max_tokens > context_limit:
            logger.warning(
                f"配置的 max_tokens ({self.max_tokens}) 超过了模型 "
                f"{self.model_name} 的上下文限制 ({context_limit})，"
                f"将自动调整为 {context_limit - 1000}"
            )
            self.max_tokens = context_limit - 1000  # 留出一些余量给输入

    def _create_pydantic_model(self) -> Any:
        """
        创建 pydantic_ai OpenAIModel 实例

        Returns:
            OpenAIModel 实例

        Raises:
            LLMConfigError: 配置错误
        """
        try:
            # 提取配置参数
            api_key = self.config["api_key"]
            base_url = self.config.get("base_url")

            # 创建 OpenAI Provider
            provider = OpenAIProvider(
                api_key=api_key,
                base_url=base_url,
            )

            # 创建 OpenAI Model
            # 注意：这里创建的是 pydantic_ai.models.OpenAIModel
            model = OpenAIModel(
                model_name=self.model_name,
                provider=provider,
            )

            logger.info(
                f"成功创建 OpenAI 模型: {self.model_name}, "
                f"base_url: {base_url or 'default'}, "
                f"max_tokens: {self.max_tokens}"
            )

            return model

        except Exception as e:
            logger.error(f"创建 OpenAI 模型失败: {e}", exc_info=True)
            raise LLMConfigError(
                f"创建 OpenAI 模型失败: {str(e)}"
            ) from e

    def get_model_settings(self) -> Dict[str, Any]:
        """
        获取模型设置，用于创建 Agent 时传递额外参数

        Returns:
            模型设置字典
        """
        settings = super().get_model_settings()

        # 添加温度等参数（如果配置中有）
        if "temperature" in self.config:
            settings["temperature"] = self.config["temperature"]

        if "top_p" in self.config:
            settings["top_p"] = self.config["top_p"]

        if "frequency_penalty" in self.config:
            settings["frequency_penalty"] = self.config["frequency_penalty"]

        if "presence_penalty" in self.config:
            settings["presence_penalty"] = self.config["presence_penalty"]

        return settings

    def __repr__(self) -> str:
        base_repr = super().__repr__()
        base_url = self.config.get("base_url", "default")
        return (
            f"{base_repr[:-1]}, "
            f"base_url={base_url})"
        )
