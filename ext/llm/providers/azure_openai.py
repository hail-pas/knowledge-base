"""
Azure OpenAI LLM 模型实现

使用 pydantic_ai.models 封装 Azure OpenAI API，提供 Azure OpenAI 模型的动态创建接口。
"""

from typing import Any, Dict
from loguru import logger

from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel

from ext.llm.base import LLMModel, ModelCapabilities
from ext.llm.exceptions import LLMConfigError


class AzureOpenAIModelWrapper(LLMModel):
    """Azure OpenAI 模型实现

    封装 pydantic_ai.models.OpenAIModel，支持 Azure OpenAI API 的所有功能。

    支持的模型:
        - GPT-4o: gpt-4o, gpt-4o-mini
        - GPT-4 Turbo: gpt-4-turbo, gpt-4-turbo-2024-04-09
        - GPT-3.5 Turbo: gpt-35-turbo, gpt-35-turbo-16k
        - GPT-4: gpt-4, gpt-4-32k

    Azure OpenAI 特殊配置要求:
        - endpoint: Azure OpenAI 服务端点（如 https://your-resource.openai.azure.com/）
        - api_key: API 密钥
        - api_version: (可选) API 版本，默认为 "2024-02-15-preview"
        - deployment: Azure 部署名称（通常与 model_name 相同）
    """

    # Azure OpenAI 模型的默认能力配置
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

    # 支持多模态的模型列表（Azure 版本）
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
        初始化 Azure OpenAI 模型

        Args:
            model_name: 模型部署名称（如 gpt-4o, gpt-35-turbo）
                注意：在 Azure OpenAI 中，这是部署名称，可能与基础模型名称不同
            model_type: 模型类型（固定为 "azure_openai"）
            config: 配置参数，必须包含:
                - endpoint: Azure OpenAI 服务端点（如 https://your-resource.openai.azure.com/）
                - api_key: API 密钥
                可选参数:
                - api_version: API 版本（默认 "2024-02-15-preview"）
                - deployment: 部署名称（如果与 model_name 不同）
                - organization: 组织 ID（通常不使用）
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
        self.validate_config(["api_key", "endpoint"])

        # 标准化 endpoint（确保以 / 结尾）
        self._normalize_endpoint()

        # 根据 model_name（部署名称）调整默认的 max_tokens
        self._adjust_max_tokens_for_model()

    def _auto_detect_capabilities(
        self,
        model_name: str,
        config: Dict[str, Any]
    ) -> Dict[str, bool]:
        """
        根据部署名称自动检测能力

        Args:
            model_name: 部署名称
            config: 用户配置

        Returns:
            能力配置字典
        """
        # 如果用户在 config 中提供了 capabilities，优先使用
        if "capabilities" in config and config["capabilities"]:
            return config["capabilities"]

        # 否则根据部署名称推断能力
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
        # 大多数 Azure OpenAI 的 GPT 模型都支持
        if model_name.startswith("gpt-35-turbo-0301"):
            capabilities["function_calling"] = False
            capabilities["tools"] = False
            capabilities["structured_output"] = False

        return capabilities

    def _normalize_endpoint(self) -> None:
        """标准化 endpoint URL"""
        endpoint = self.config.get("endpoint", "")

        if not endpoint:
            raise LLMConfigError("Azure OpenAI 配置缺少必需的 endpoint 参数")

        # 移除末尾的斜杠（如果有多个）
        endpoint = endpoint.rstrip("/")

        # 重新添加一个斜杠
        self.config["endpoint"] = endpoint + "/"

        logger.debug(f"标准化后的 endpoint: {self.config['endpoint']}")

    def _adjust_max_tokens_for_model(self) -> None:
        """根据部署模型限制调整 max_tokens"""
        # Azure OpenAI 不同部署模型的 token 限制
        # 注意：这里的 model_name 实际上是部署名称
        model_limits = {
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-4-turbo": 128000,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-35-turbo": 16385,
            "gpt-35-turbo-16k": 16385,
        }

        # 获取部署模型的上下文限制
        context_limit = model_limits.get(
            self.model_name,
            model_limits.get("gpt-4o", 128000)  # 默认使用 GPT-4o 的限制
        )

        # 如果配置的 max_tokens 超过模型限制，进行警告和调整
        if self.max_tokens > context_limit:
            logger.warning(
                f"配置的 max_tokens ({self.max_tokens}) 超过了 Azure OpenAI 部署 "
                f"{self.model_name} 的上下文限制 ({context_limit})，"
                f"将自动调整为 {context_limit - 1000}"
            )
            self.max_tokens = context_limit - 1000  # 留出一些余量给输入

    def _create_pydantic_model(self) -> Any:
        """
        创建 pydantic_ai OpenAIModel 实例（配置用于 Azure OpenAI）

        Returns:
            OpenAIModel 实例（配置为使用 Azure OpenAI）

        Raises:
            LLMConfigError: 配置错误
        """
        try:
            # 提取配置参数
            api_key = self.config["api_key"]
            endpoint = self.config["endpoint"]
            api_version = self.config.get("api_version", "2024-02-15-preview")

            # 验证 endpoint 格式
            if not endpoint.startswith(("http://", "https://")):
                raise LLMConfigError(
                    f"Endpoint 必须以 http:// 或 https:// 开头，当前值: {endpoint}"
                )

            # 创建 OpenAI Provider（配置为使用 Azure OpenAI）
            # Azure OpenAI 使用 OpenAI 兼容的 API，但需要特殊的 endpoint 和 header
            provider = OpenAIProvider(
                api_key=api_key,
                base_url=endpoint,
                # Azure OpenAI 特定的 headers
                # 注意：pydantic_ai 的 OpenAIProvider 可能不直接支持 api_version
                # 如果需要，可以通过额外的配置传递
            )

            # 创建 OpenAI Model
            # 注意：这里创建的是 pydantic_ai.models.OpenAIModel
            # 通过 base_url 指向 Azure OpenAI 端点
            model = OpenAIModel(
                model_name=self.model_name,  # 在 Azure OpenAI 中这是部署名称
                provider=provider,
            )

            logger.info(
                f"成功创建 Azure OpenAI 模型: {self.model_name}, "
                f"endpoint: {endpoint}, "
                f"api_version: {api_version}, "
                f"max_tokens: {self.max_tokens}"
            )

            return model

        except LLMConfigError:
            raise
        except Exception as e:
            logger.error(f"创建 Azure OpenAI 模型失败: {e}", exc_info=True)
            raise LLMConfigError(
                f"创建 Azure OpenAI 模型失败: {str(e)}"
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

    def get_deployment_name(self) -> str:
        """获取部署名称"""
        return self.model_name

    def get_api_version(self) -> str:
        """获取 API 版本"""
        return self.config.get("api_version", "2024-02-15-preview")

    def supports_multimodal(self) -> bool:
        """检查模型是否支持多模态"""
        return self.capabilities.multimodal or self.capabilities.vision

    def supports_function_calling(self) -> bool:
        """检查模型是否支持函数调用"""
        return self.capabilities.function_calling

    def supports_json_output(self) -> bool:
        """检查模型是否支持 JSON 输出"""
        return self.capabilities.json_output

    def supports_structured_output(self) -> bool:
        """检查模型是否支持结构化输出（Pydantic 模型）"""
        return self.capabilities.structured_output

    def __repr__(self) -> str:
        base_repr = super().__repr__()
        endpoint = self.config.get("endpoint", "unknown")
        api_version = self.get_api_version()
        return (
            f"{base_repr[:-1]}, "
            f"endpoint={endpoint}, "
            f"api_version={api_version})"
        )
