"""
Embedding 模型实现模块

包含各种 embedding 服务提供商的具体实现。
"""

from ext.embedding.providers.openai import OpenAIEmbedding

__all__ = [
    "OpenAIEmbedding",
]
