"""
LLM 模型抽象层

提供统一的 LLM 接口，支持动态切换不同的 LLM 服务提供商。
基于 pydantic_ai.models 实现。
"""

from ext.llm.base import LLMModel, ModelCapabilities
from ext.llm.factory import LLMModelFactory
from ext.llm.exceptions import (
    LLMError,
    LLMConfigError,
    LLMModelNotFoundError,
    LLMAPIError,
    LLMTimeoutError,
    LLMRateLimitError,
    LLMTokenLimitError,
    LLMCapabilityError,
    LLMStreamingError,
)

__all__ = [
    # 基类
    "LLMModel",
    "ModelCapabilities",
    # 工厂
    "LLMModelFactory",
    # 异常
    "LLMError",
    "LLMConfigError",
    "LLMModelNotFoundError",
    "LLMAPIError",
    "LLMTimeoutError",
    "LLMRateLimitError",
    "LLMTokenLimitError",
    "LLMCapabilityError",
    "LLMStreamingError",
]

__version__ = "0.1.0"
