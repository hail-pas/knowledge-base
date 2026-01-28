"""
OpenAI Embedding Provider

完全使用基类的默认实现，无需实现任何方法
"""

from ext.embedding.base import BaseEmbeddingModel
from ext.embedding.providers.types import OpenAIExtraConfig


class OpenAIEmbeddingModel(BaseEmbeddingModel[OpenAIExtraConfig]):
    """
    OpenAI Embedding Provider

    泛型参数: OpenAIExtraConfig（定义了 encoding_format 等字段）

    完全使用基类的默认实现，无需覆盖任何方法

    extra_config 类型会自动从泛型参数中提取
    """

    # 无需任何实现！
