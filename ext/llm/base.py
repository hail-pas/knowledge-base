"""
LLM 模型抽象基类

提供统一的 LLM 接口，封装 pydantic_ai.models，支持动态切换不同的 LLM 服务提供商。
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List, Union, TYPE_CHECKING
from loguru import logger


class ModelCapabilities:
    """模型能力配置

    定义 LLM 模型支持的各种能力，用于验证模型是否满足特定功能需求。
    """

    def __init__(
        self,
        function_calling: bool = False,
        json_output: bool = False,
        multimodal: bool = False,
        streaming: bool = True,
        vision: bool = False,
        audio_input: bool = False,
        audio_output: bool = False,
        tools: bool = False,
        structured_output: bool = False,
    ):
        """
        初始化模型能力配置

        Args:
            function_calling: 是否支持函数调用
            json_output: 是否支持 JSON 格式输出
            multimodal: 是否支持多模态输入
            streaming: 是否支持流式输出
            vision: 是否支持视觉能力
            audio_input: 是否支持音频输入
            audio_output: 是否支持音频输出
            tools: 是否支持工具调用
            structured_output: 是否支持结构化输出（Pydantic模型）
        """
        self.function_calling = function_calling
        self.json_output = json_output
        self.multimodal = multimodal
        self.streaming = streaming
        self.vision = vision
        self.audio_input = audio_input
        self.audio_output = audio_output
        self.tools = tools
        self.structured_output = structured_output

    def to_dict(self) -> Dict[str, bool]:
        """转换为字典格式"""
        return {
            "function_calling": self.function_calling,
            "json_output": self.json_output,
            "multimodal": self.multimodal,
            "streaming": self.streaming,
            "vision": self.vision,
            "audio_input": self.audio_input,
            "audio_output": self.audio_output,
            "tools": self.tools,
            "structured_output": self.structured_output,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelCapabilities":
        """从字典创建能力配置"""
        return cls(
            function_calling=data.get("function_calling", False),
            json_output=data.get("json_output", False),
            multimodal=data.get("multimodal", False),
            streaming=data.get("streaming", True),
            vision=data.get("vision", False),
            audio_input=data.get("audio_input", False),
            audio_output=data.get("audio_output", False),
            tools=data.get("tools", False),
            structured_output=data.get("structured_output", False),
        )

    def requires_capability(
        self,
        capability: str,
        raise_error: bool = True
    ) -> bool:
        """
        检查模型是否支持特定能力

        Args:
            capability: 能力名称（如 "function_calling", "vision" 等）
            raise_error: 如果不支持是否抛出异常

        Returns:
            是否支持该能力

        Raises:
            LLMCapabilityError: 当 raise_error=True 且不支持该能力时
        """
        has_capability = getattr(self, capability, False)

        if not has_capability and raise_error:
            from ext.llm.exceptions import LLMCapabilityError
            raise LLMCapabilityError(
                f"模型不支持 {capability} 能力。"
                f"当前能力: {self.to_dict()}"
            )

        return has_capability

    def __repr__(self) -> str:
        enabled = [k for k, v in self.to_dict().items() if v]
        return f"ModelCapabilities({', '.join(enabled) if enabled else 'none'})"


class LLMModel(ABC):
    """LLM 模型抽象基类

    所有 LLM 模型实现必须继承此类并实现核心方法。
    提供统一的接口，封装 pydantic_ai.models，支持动态切换不同的 LLM 服务提供商。

    注意：此类封装的是 pydantic_ai.models.Model，不是 Agent。
    Agent 应当在外部使用 Model 实例创建。
    """

    def __init__(
        self,
        model_name: str,
        model_type: str,
        config: Dict[str, Any],
        max_tokens: int = 4096,
        capabilities: Optional[ModelCapabilities] = None,
        max_retries: int = 3,
        timeout: int = 60,
        rate_limit: int = 60,
    ):
        """
        初始化 LLM 模型

        Args:
            model_name: 模型标识符（如 gpt-4o, claude-3-opus, deepseek-chat 等）
            model_type: 模型类型（openai, azure_openai, deepseek 等）
            config: 模型配置参数（如 api_key, base_url, temperature 等）
            max_tokens: 模型最大输出 token 数
            capabilities: 模型能力配置
            max_retries: 最大重试次数
            timeout: 请求超时时间（秒）
            rate_limit: 每分钟最大请求次数（0表示无限制）
        """
        self.model_name = model_name
        self.model_type = model_type
        self.config = config or {}
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout = timeout
        self.rate_limit = rate_limit

        # 设置能力配置
        if capabilities is None:
            # 从 config 中读取能力配置，如果没有则使用默认值
            capabilities_data = self.config.get("capabilities", {})
            self.capabilities = ModelCapabilities.from_dict(capabilities_data)
        else:
            self.capabilities = capabilities

        # pydantic_ai Model 实例（延迟创建）
        self._model: Optional[Any] = None

    @abstractmethod
    def _create_pydantic_model(self) -> "KnownModel":
        """
        创建 pydantic_ai Model 实例（由子类实现）

        Returns:
            pydantic_ai.models.KnownModel 实例

        Raises:
            LLMConfigError: 配置错误
            LLMModelNotFoundError: 模型未找到
        """
        pass

    @property
    def model(self) -> "KnownModel":
        """
        获取 pydantic_ai Model 实例（懒加载）

        Returns:
            pydantic_ai.models.KnownModel 实例
        """
        if self._model is None:
            self._model = self._create_pydantic_model()
        return self._model

    def get_model_for_agent(self, **agent_kwargs) -> "KnownModel":
        """
        获取用于创建 Agent 的 Model 实例

        这个方法提供了一个便捷的接口，让外部可以直接使用返回的 Model 实例创建 pydantic_ai Agent。

        Args:
            **agent_kwargs: 额外的 Agent 配置参数（如 temperature, max_tokens 等）
                           这些参数会与模型的默认配置合并，agent_kwargs 的优先级更高。

        Returns:
            pydantic_ai.models.KnownModel 实例

        Example:
            >>> llm_model = await LLMModelFactory.create(config)
            >>> model = llm_model.get_model_for_agent(temperature=0.7)
            >>> agent = Agent(model, result_type=str)
        """
        # 获取基础 Model 实例
        model = self.model

        # 某些 Model 类型支持在获取时应用额外的设置
        # 这里可以根据具体的 pydantic_ai Model 类型进行扩展
        # 目前直接返回 model，具体的参数配置应该在创建 Model 时处理

        return model

    def validate_config(self, required_keys: List[str]) -> None:
        """
        验证配置是否包含必需的参数

        Args:
            required_keys: 必需的配置键列表

        Raises:
            LLMConfigError: 配置缺少必需参数
        """
        from ext.llm.exceptions import LLMConfigError

        missing_keys = [key for key in required_keys if key not in self.config or not self.config[key]]
        if missing_keys:
            raise LLMConfigError(
                f"缺少必需的配置参数: {', '.join(missing_keys)}. "
                f"模型类型: {self.model_type}, 模型名称: {self.model_name}"
            )

    def requires_capability(self, capability: str) -> bool:
        """
        检查模型是否支持特定能力

        Args:
            capability: 能力名称

        Returns:
            是否支持该能力
        """
        return self.capabilities.requires_capability(capability, raise_error=False)

    def ensure_capability(self, capability: str) -> None:
        """
        确保模型支持特定能力，如果不支持则抛出异常

        Args:
            capability: 能力名称

        Raises:
            LLMCapabilityError: 不支持该能力
        """
        self.capabilities.requires_capability(capability, raise_error=True)

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        获取配置参数

        Args:
            key: 配置键
            default: 默认值

        Returns:
            配置值
        """
        return self.config.get(key, default)

    def set_config(self, key: str, value: Any) -> None:
        """
        设置配置参数

        注意：此方法只更新内存中的配置，不会影响已创建的 Model 实例。
        如果需要重新创建 Model，可以调用 reset_model() 方法。

        Args:
            key: 配置键
            value: 配置值
        """
        self.config[key] = value

    def reset_model(self) -> None:
        """重置 Model 实例（下次访问时重新创建）"""
        self._model = None

    def get_model_settings(self) -> Dict[str, Any]:
        """
        获取模型设置（用于传递给 Agent 或 API）

        Returns:
            模型设置字典
        """
        return {
            "max_tokens": self.max_tokens,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name={self.model_name}, "
            f"model_type={self.model_type}, "
            f"capabilities={self.capabilities}, "
            f"max_tokens={self.max_tokens})"
        )

    async def close(self) -> None:
        """
        关闭模型，释放资源

        某些模型可能需要清理资源（如关闭连接池等）
        """
        # 默认实现：如果 model 有 close 方法则调用
        if self._model is not None:
            if hasattr(self._model, 'close'):
                await self._model.close()
            self._model = None

    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()
