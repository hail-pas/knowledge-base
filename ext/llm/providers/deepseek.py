"""
DeepSeek LLM 模型实现

使用 pydantic_ai.models 封装 DeepSeek API，提供 DeepSeek 模型的动态创建接口。
DeepSeek 使用 OpenAI 兼容的 API 接口。
"""

from typing import Any, Dict
from loguru import logger

from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.deepseek import DeepSeekProvider
from pydantic_ai.models.openai import OpenAIModel

from ext.llm.base import LLMModel, ModelCapabilities
from ext.llm.exceptions import LLMConfigError


class DeepSeekModelWrapper(LLMModel):
    """DeepSeek 模型实现

    封装 pydantic_ai.models.OpenAIModel，支持 DeepSeek API 的所有功能。
    DeepSeek 使用 OpenAI 兼容的 API 接口。

    支持的模型:
        - DeepSeek V3: deepseek-chat (通用对话模型)
        - DeepSeek Coder V2: deepseek-coder (代码生成模型)

    DeepSeek 特点:
        - 使用 OpenAI 兼容的 API
        - 基础 URL: https://api.deepseek.com
        - 支持函数调用和工具
        - 支持结构化输出
        - 支持流式输出
    """

    # DeepSeek 模型的默认能力配置
    DEFAULT_CAPABILITIES = {
        "function_calling": False,
        "json_output": False,
        "multimodal": False,  # DeepSeek 目前不支持多模态
        "streaming": True,
        "vision": False,
        "audio_input": False,
        "audio_output": False,
        "tools": False,
        "structured_output": True,
    }

    # DeepSeek API 基础 URL
    DEFAULT_BASE_URL = "https://api.deepseek.com"

    # DeepSeek 模型的上下文限制
    MODEL_LIMITS = {
        "deepseek-chat": 128000,
        "deepseek-coder": 128000,
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
        初始化 DeepSeek 模型

        Args:
            model_name: 模型名称（如 deepseek-chat, deepseek-coder）
            model_type: 模型类型（固定为 "deepseek"）
            config: 配置参数，需要包含:
                - api_key: DeepSeek API 密钥
                可选参数:
                - base_url: 自定义 API 基础 URL（默认使用官方地址）
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

        # 设置默认的 base_url
        if "base_url" not in self.config:
            self.config["base_url"] = self.DEFAULT_BASE_URL

        # 根据 model_name 调整默认的 max_tokens
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

        # DeepSeek 的能力配置相对固定
        return self.DEFAULT_CAPABILITIES.copy()

    def _adjust_max_tokens_for_model(self) -> None:
        """根据模型限制调整 max_tokens"""
        # 获取模型的上下文限制
        context_limit = self.MODEL_LIMITS.get(
            self.model_name,
            self.MODEL_LIMITS.get("deepseek-chat", 128000)  # 默认使用 deepseek-chat 的限制
        )

        # 如果配置的 max_tokens 超过模型限制，进行警告和调整
        if self.max_tokens > context_limit:
            logger.warning(
                f"配置的 max_tokens ({self.max_tokens}) 超过了 DeepSeek 模型 "
                f"{self.model_name} 的上下文限制 ({context_limit})，"
                f"将自动调整为 {context_limit - 1000}"
            )
            self.max_tokens = context_limit - 1000  # 留出一些余量给输入

    def _create_pydantic_model(self) -> Any:
        """
        创建 pydantic_ai Model 实例（配置用于 DeepSeek）

        Returns:
            OpenAIModel 实例（配置为使用 DeepSeek API）

        Raises:
            LLMConfigError: 配置错误
        """
        try:
            # 提取配置参数
            api_key = self.config["api_key"]
            base_url = self.config.get("base_url", self.DEFAULT_BASE_URL)

            provider = OpenAIProvider(
                api_key=api_key,
                base_url=base_url,
            )

            # 创建 OpenAI Model
            model = OpenAIModel(
                model_name=self.model_name,
                provider=provider,
            )

            logger.info(
                f"成功创建 DeepSeek 模型: {self.model_name}, "
                f"base_url: {base_url}, "
                f"max_tokens: {self.max_tokens}"
            )

            return model

        except Exception as e:
            logger.error(f"创建 DeepSeek 模型失败: {e}", exc_info=True)
            raise LLMConfigError(
                f"创建 DeepSeek 模型失败: {str(e)}"
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

    def supports_function_calling(self) -> bool:
        """检查模型是否支持函数调用"""
        return self.capabilities.function_calling

    def supports_json_output(self) -> bool:
        """检查模型是否支持 JSON 输出"""
        return self.capabilities.json_output

    def supports_structured_output(self) -> bool:
        """检查模型是否支持结构化输出（Pydantic 模型）"""
        return self.capabilities.structured_output

    def supports_multimodal(self) -> bool:
        """检查模型是否支持多模态（DeepSeek 不支持）"""
        return False

    def supports_vision(self) -> bool:
        """检查模型是否支持视觉能力（DeepSeek 不支持）"""
        return False

    def get_api_base_url(self) -> str:
        """获取 API 基础 URL"""
        return self.config.get("base_url", self.DEFAULT_BASE_URL)

    def __repr__(self) -> str:
        base_repr = super().__repr__()
        base_url = self.config.get("base_url", self.DEFAULT_BASE_URL)
        return (
            f"{base_repr[:-1]}, "
            f"base_url={base_url})"
        )
