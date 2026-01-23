"""
Embedding 模块入口

提供 embedding 模型的统一接口和工厂类
"""

from ext.embedding.base import BaseEmbeddingModel
from ext.embedding.factory import EmbeddingModelFactory

__all__ = [
    "BaseEmbeddingModel",
    "EmbeddingModelFactory",
]
