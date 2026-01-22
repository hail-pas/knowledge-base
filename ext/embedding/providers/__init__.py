"""
Embedding Providers 自动发现和注册

在模块导入时自动注册已知的 embedding providers
"""

from loguru import logger

from ext.embedding.factory import EmbeddingModelFactory
from ext.ext_tortoise.enums import EmbeddingModelTypeEnum
from ext.embedding.providers.types import OpenAIExtraConfig

# 显式注册已知providers
try:
    from ext.embedding.providers.openai import OpenAIEmbeddingModel

    EmbeddingModelFactory.register(EmbeddingModelTypeEnum.openai, OpenAIEmbeddingModel)
    logger.info("Registered OpenAI Embedding provider")
except Exception as e:
    logger.warning(f"Failed to register OpenAI provider: {e}")

# 可选：自动扫描并注册其他providers（未来扩展）
# provider_dir = Path(__file__).parent
# for file in provider_dir.glob("*.py"):
#     if file.name.startswith("_"):
#         continue
#     module_name = file.stem
#     try:
#         module = importlib.import_module(f"ext.embedding.providers.{module_name}")
#         for attr_name in dir(module):
#             attr = getattr(module, attr_name)
#             if (isinstance(attr, type) and
#                 issubclass(attr, BaseEmbeddingModel) and
#                 attr != BaseEmbeddingModel):
#                 # 尝试注册（如果有对应的枚举值）
#                 pass
#     except Exception as e:
#         logger.warning(f"Failed to load provider {module_name}: {e}")

__all__ = ["OpenAIEmbeddingModel"]
