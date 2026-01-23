"""
LLM 模块

提供统一的 LLM 模型抽象层，支持多种 provider
"""

from ext.llm.base import BaseLLMModel
from ext.llm.factory import LLMModelFactory
from ext.llm.types import (
    ChatMessage,
    LLMRequest,
    LLMResponse,
    StreamChunk,
    CompletionRequest,
    CompletionResponse,
    FunctionDefinition,
    ToolDefinition,
    ToolCall,
    TokenUsage,
    BaseExtraConfig,
    OpenAIExtraConfig,
    AzureOpenAIExtraConfig,
    DeepSeekExtraConfig,
    AnthropicExtraConfig,
)

__all__ = [
    # Base class
    "BaseLLMModel",
    # Factory
    "LLMModelFactory",
    # Request/Response models
    "ChatMessage",
    "LLMRequest",
    "LLMResponse",
    "StreamChunk",
    "CompletionRequest",
    "CompletionResponse",
    "Tool-related",
    "FunctionDefinition",
    "ToolDefinition",
    "ToolCall",
    # Utility models
    "TokenUsage",
    # Extra config types
    "BaseExtraConfig",
    "OpenAIExtraConfig",
    "AzureOpenAIExtraConfig",
    "DeepSeekExtraConfig",
    "AnthropicExtraConfig",
]
