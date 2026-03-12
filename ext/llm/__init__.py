"""
LLM 模块

提供统一的 LLM 模型抽象层，支持多种 provider
"""

from ext.llm.base import BaseLLMModel
from ext.llm.types import (
    ToolCall,
    LLMRequest,
    TokenUsage,
    ChatMessage,
    LLMResponse,
    StreamChunk,
    ToolDefinition,
    BaseExtraConfig,
    CompletionRequest,
    OpenAIExtraConfig,
    CompletionResponse,
    FunctionDefinition,
    DeepSeekExtraConfig,
    AnthropicExtraConfig,
    AzureOpenAIExtraConfig,
)
from ext.llm.factory import LLMModelFactory

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
