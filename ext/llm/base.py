"""
LLM 模型泛型基类

提供 LLM 模型的抽象接口和默认实现
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Dict, Any, Optional, Type
from collections.abc import AsyncIterator
import httpx
from loguru import logger
from util.general import truncate_content

from ext.llm.types import (
    BaseExtraConfig,
    LLMRequest,
    LLMResponse,
    StreamChunk,
    CompletionRequest,
    CompletionResponse,
    ChatMessage,
    TokenUsage,
    ToolCall,
)

ExtraConfigT = TypeVar("ExtraConfigT", bound=BaseExtraConfig)


class BaseLLMModel(Generic[ExtraConfigT], ABC):
    """
    LLM 模型抽象基类（泛型）

    类型参数:
        ExtraConfigT: extra_config 的具体类型

    设计原则:
        1. 提供大量默认实现，子类只需在必要时覆盖
        2. 通过配置化处理 provider 差异
        3. 核心方法（chat, chat_stream）由子类实现
        4. 使用全局 httpx client（对于非SDK实现）
    """

    extra_config: ExtraConfigT

    def __init__(
        self,
        model_name: str,
        model_type: str,
        max_tokens: int,
        api_key: str | None,
        base_url: str | None,
        supports_chat: bool,
        supports_completion: bool,
        supports_streaming: bool,
        supports_function_calling: bool,
        supports_vision: bool,
        default_temperature: float,
        default_top_p: float,
        max_retries: int,
        timeout: int,
        extra_config: dict[str, Any],
    ):
        """
        初始化 LLM 模型

        Args:
            model_name: 模型名称
            model_type: 模型类型
            max_tokens: 最大token数
            api_key: API密钥
            base_url: API基础URL
            supports_chat: 是否支持对话模式
            supports_completion: 是否支持补全模式
            supports_streaming: 是否支持流式输出
            supports_function_calling: 是否支持函数调用
            supports_vision: 是否支持视觉/图像
            default_temperature: 默认温度参数
            default_top_p: 默认top_p
            max_retries: 最大重试次数
            timeout: 请求超时时间(秒)
            extra_config: provider特定配置（dict），内部会转换成具体的 dataclass
        """
        self.model_name = model_name
        self.model_type = model_type
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.base_url = base_url

        self.supports_chat = supports_chat
        self.supports_completion = supports_completion
        self.supports_streaming = supports_streaming
        self.supports_function_calling = supports_function_calling
        self.supports_vision = supports_vision

        self.default_temperature = default_temperature
        self.default_top_p = default_top_p
        self.max_retries = max_retries
        self.timeout = timeout

        self.extra_config: ExtraConfigT = self._convert_extra_config(extra_config)

        self._validate_config()

        logger.info(f"Initialized LLM model: {self.model_type}/{self.model_name}, max_tokens={self.max_tokens}")

    def _validate_config(self) -> None:
        """验证配置"""
        if not self.base_url:
            logger.error(f"Configuration validation failed: {self.model_type} requires base_url")
            raise ValueError(f"{self.model_type} requires base_url")

        if self.extra_config.requires_auth and not self.api_key:
            logger.error(f"Configuration validation failed: {self.model_type} requires api_key")
            raise ValueError(f"{self.model_type} requires api_key")

        logger.debug(f"Configuration validated for {self.model_type}/{self.model_name}")

    def _get_extra_config_cls(self) -> type[BaseExtraConfig]:
        """
        从泛型参数自动提取 extra_config 类型

        子类通过 `class OpenAILLMModel(BaseLLMModel[OpenAIExtraConfig])` 声明泛型参数
        此方法会自动从 `__orig_bases__` 中提取泛型类型

        Returns:
            extra_config 的 pydantic model 类型

        Raises:
            ValueError: 如果无法提取泛型类型
        """
        if hasattr(self, "__orig_bases__"):
            for base in self.__orig_bases__:  # type: ignore
                if hasattr(base, "__args__") and base.__args__:
                    extra_config_type = base.__args__[0]
                    if isinstance(extra_config_type, type) and issubclass(extra_config_type, BaseExtraConfig):
                        return extra_config_type

        logger.warning(
            "无法从泛型参数提取 extra_config 类型，使用默认类型 BaseExtraConfig。"
            "请确保子类正确声明泛型参数，如：class OpenAILLMModel(BaseLLMModel[OpenAIExtraConfig])",
        )
        return BaseExtraConfig

    def _convert_extra_config(self, extra_config_dict: dict[str, Any]) -> ExtraConfigT:
        """
        将 dict 转换成具体的 pydantic model 类型

        自动从泛型参数提取类型并转换

        Args:
            extra_config_dict: extra_config 的字典形式

        Returns:
            类型化的 extra_config 实例
        """
        extra_config_cls = self._get_extra_config_cls()
        return extra_config_cls.from_dict(extra_config_dict)  # type: ignore

    # ========== 核心抽象方法（必须由子类实现） ==========

    @abstractmethod
    async def chat(self, request: LLMRequest) -> LLMResponse:
        """
        发起对话请求（非流式）

        Args:
            request: LLM 请求

        Returns:
            LLM 响应

        Raises:
            NotImplementedError: 子类未实现
        """
        raise NotImplementedError

    @abstractmethod
    async def chat_stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        """
        发起对话请求（流式）

        如果 provider 不原生支持流式，子类可以在内部转换为流式

        Args:
            request: LLM 请求

        Yields:
            流式响应块
        """
        raise NotImplementedError

    # ========== 可选实现的方法 ==========

    async def completion(self, request: CompletionRequest) -> CompletionResponse:
        """
        传统补全模式

        默认实现：将补全请求转换为对话请求
        如果不支持 completion，子类可以抛出 NotImplementedError

        Args:
            request: 补全请求

        Returns:
            补全响应

        Raises:
            NotImplementedError: 如果模型不支持补全模式
        """
        logger.debug(
            f"Completion request - prompt: {truncate_content(request.prompt)}, model: {request.model or self.model_name}, max_tokens: {request.max_tokens or self.max_tokens}",
        )

        if not self.supports_completion:
            logger.error(f"Completion mode not supported for {self.model_type}")
            raise NotImplementedError(f"{self.model_type} does not support completion mode")

        # 将补全请求转换为对话请求（单条 user 消息）
        chat_request = LLMRequest(
            messages=[ChatMessage(role="user", content=request.prompt)],
            model=request.model or self.model_name,
            temperature=request.temperature or self.default_temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p or self.default_top_p,
            stop=request.stop,
        )

        chat_response = await self.chat(chat_request)

        # 将对话响应转换为补全响应
        response = CompletionResponse(
            text=chat_response.content,
            usage=chat_response.usage,
            finish_reason=chat_response.finish_reason,
            model=chat_response.model,
        )

        logger.debug(
            f"Completion response - text: {truncate_content(response.text)}, "
            f"tokens: {response.usage.total_tokens}, finish_reason: {response.finish_reason}",
        )

        return response

    # ========== 通用工具方法 ==========

    def get_httpx_client(self) -> httpx.AsyncClient:
        """获取全局 httpx client"""
        from config.main import local_configs

        return local_configs.extensions.httpx.instance

    # ========== 请求构建辅助方法（用于 httpx 实现） ==========

    def build_endpoint_url(self) -> str:
        """
        构建 API 端点 URL

        默认实现: {base_url}{endpoint}
        """
        base_url = self.base_url or ""
        endpoint = self.extra_config.endpoint

        base_url = base_url.rstrip("/")
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint

        query_params = self.extra_config.query_params
        if query_params:
            query_string = "&".join(f"{k}={v}" for k, v in query_params.items())
            endpoint = f"{endpoint}?{query_string}"

        return f"{base_url}{endpoint}"

    def build_auth_headers(self) -> dict[str, str]:
        """
        构建认证头

        默认 Bearer Token 实现（通过 extra_config 配置）
        """
        if not self.extra_config.requires_auth:
            return {}

        if not self.api_key:
            raise ValueError("API key is required")

        auth_header = self.extra_config.auth_header
        auth_type = self.extra_config.auth_type
        value = f"{auth_type} {self.api_key}" if auth_type else self.api_key
        return {auth_header: value}

    def build_request_headers(self) -> dict[str, str]:
        """构建完整的请求头"""
        headers = {"Content-Type": "application/json"}
        headers.update(self.build_auth_headers())
        headers.update(self.extra_config.headers)
        return headers

    # ========== 重试逻辑（配置驱动） ==========

    def should_retry(self, status_code: int, attempt: int) -> bool:
        """判断是否应该重试（从 extra_config 读取）"""
        if attempt >= self.max_retries:
            return False
        should_retry = status_code in self.extra_config.retry_on_status_codes
        if should_retry:
            logger.debug(f"Should retry: status_code={status_code}, attempt={attempt}/{self.max_retries}")
        return should_retry

    def get_retry_delay(self, attempt: int) -> float:
        """获取重试延迟（从 extra_config 读取）"""
        strategy = self.extra_config.retry_strategy
        if strategy == "exponential":
            delay = 2**attempt
        elif strategy == "linear":
            delay = attempt * 2
        else:  # constant
            delay = 1.0

        logger.debug(f"Retry delay calculated: strategy={strategy}, attempt={attempt}, delay={delay}s")
        return delay

    # ========== 响应解析辅助方法（用于 httpx 实现） ==========

    def _extract_by_path(self, data: dict[str, Any], path: str) -> Any:
        """通过路径提取数据（点分隔路径）"""
        keys = path.split(".")
        result = data
        for key in keys:
            if isinstance(result, dict):
                result = result.get(key)
            else:
                return None
            if result is None:
                return None
        return result

    def _extract_content(self, response_data: dict[str, Any]) -> str:
        """从响应中提取内容"""
        content_path = self.extra_config.response_content_path
        content = self._extract_by_path(response_data, content_path)
        return content or ""

    def _extract_usage(self, response_data: dict[str, Any]) -> TokenUsage:
        """从响应中提取使用统计"""
        usage_path = self.extra_config.response_usage_path
        usage_data = self._extract_by_path(response_data, usage_path)

        if usage_data:
            return TokenUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )
        return TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    def _extract_finish_reason(self, response_data: dict[str, Any]) -> str:
        """从响应中提取结束原因"""
        finish_reason_path = self.extra_config.response_finish_reason_path
        return self._extract_by_path(response_data, finish_reason_path) or "stop"

    def _extract_model(self, response_data: dict[str, Any]) -> str | None:
        """从响应中提取模型名称"""
        model_path = self.extra_config.response_model_path
        return self._extract_by_path(response_data, model_path)

    def _extract_tool_calls(self, response_data: dict[str, Any]) -> list[ToolCall] | None:
        """从响应中提取工具调用"""
        tool_calls_path = self.extra_config.response_tool_calls_path
        if not tool_calls_path:
            return None

        tool_calls_data = self._extract_by_path(response_data, tool_calls_path)
        if not tool_calls_data:
            return None

        tool_calls = []
        for tc_data in tool_calls_data:
            tool_call = ToolCall(
                id=tc_data.get("id", ""),
                type=tc_data.get("type", "function"),
                function=tc_data.get("function", {}),
            )
            tool_calls.append(tool_call)

        return tool_calls

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name={self.model_name}, "
            f"model_type={self.model_type}, "
            f"max_tokens={self.max_tokens})"
        )
