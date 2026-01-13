"""
Embedding 模型抽象层

提供统一的 embedding 接口，支持动态切换不同的 embedding 服务提供商。
"""

from ext.embedding.base import EmbeddingModel, EmbeddingResult
from ext.embedding.factory import EmbeddingModelFactory
from ext.embedding.exceptions import (
    EmbeddingError,
    EmbeddingConfigError,
    EmbeddingModelNotFoundError,
    EmbeddingAPIError,
)

__all__ = [
    # 基类
    "EmbeddingModel",
    "EmbeddingResult",
    # 工厂
    "EmbeddingModelFactory",
    # 异常
    "EmbeddingError",
    "EmbeddingConfigError",
    "EmbeddingModelNotFoundError",
    "EmbeddingAPIError",
]

__version__ = "0.1.0"
