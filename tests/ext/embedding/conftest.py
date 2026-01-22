"""
Embedding 模块的 conftest.py

定义测试所需的 fixtures
"""

import os
import pytest
from ext.ext_tortoise.enums import EmbeddingModelTypeEnum
from ext.ext_tortoise.models.knowledge_base import EmbeddingModelConfig


# 获取环境变量配置
OPENAI_EMBEDDING_BASE_URL = os.getenv("OPENAI_EMBEDDING_BASE_URL")
OPENAI_EMBEDDING_MODEL_NAME = os.getenv("OPENAI_EMBEDDING_MODEL_NAME")
OPENAI_EMBEDDING_API_KEY = os.getenv("OPENAI_EMBEDDING_API_KEY")
OPENAI_EMBEDDING_DIMENSION = os.getenv("OPENAI_EMBEDDING_DIMENSION", "1536")
OPENAI_EMBEDDING_MAX_BATCH_SIZE = os.getenv("OPENAI_EMBEDDING_MAX_BATCH_SIZE", "100")
OPENAI_EMBEDDING_MAX_TOKEN_PER_TEXT = os.getenv("OPENAI_EMBEDDING_MAX_TOKEN_PER_TEXT", "8192")


# 跳过测试的条件
skip_if_no_api_key = pytest.mark.skipif(
    not OPENAI_EMBEDDING_API_KEY, reason="OPENAI_EMBEDDING_API_KEY not set in environment"
)


@pytest.fixture
def openai_embedding_config_dict():
    """返回 OpenAI Embedding 配置字典（不创建对象）"""
    return {
        "name": "test-openai-embedding",
        "type": EmbeddingModelTypeEnum.openai,
        "model_name": OPENAI_EMBEDDING_MODEL_NAME or "text-embedding-3-small",
        "dimension": int(OPENAI_EMBEDDING_DIMENSION),
        "api_key": OPENAI_EMBEDDING_API_KEY,
        "base_url": OPENAI_EMBEDDING_BASE_URL or "https://api.openai.com",
        "max_chunk_length": int(OPENAI_EMBEDDING_MAX_TOKEN_PER_TEXT),
        "batch_size": int(OPENAI_EMBEDDING_MAX_BATCH_SIZE),
        "max_retries": 3,
        "timeout": 60,
        "rate_limit": 60,
        "extra_config": {"encoding_format": "float"},
        "is_enabled": True,
        "is_default": False,
        "description": "测试用OpenAI Embedding配置",
    }


@pytest.fixture
async def openai_embedding_config(openai_embedding_config_dict):
    """创建并保存 OpenAI Embedding 配置到数据库"""
    # 清理可能存在的旧配置
    await EmbeddingModelConfig.filter(name=openai_embedding_config_dict["name"]).delete()

    # 创建新配置
    config = await EmbeddingModelConfig.create(**openai_embedding_config_dict)
    return config


@pytest.fixture
async def openai_embedding_config_with_extra(openai_embedding_config_dict):
    """创建带有额外配置的 OpenAI Embedding 配置"""
    extra_dict = openai_embedding_config_dict.copy()
    extra_dict["name"] = "test-openai-embedding-extra"
    extra_dict["extra_config"] = {
        "encoding_format": "float",
        "user": "test-user-123",
    }

    # 清理可能存在的旧配置
    await EmbeddingModelConfig.filter(name=extra_dict["name"]).delete()

    # 创建新配置
    config = await EmbeddingModelConfig.create(**extra_dict)
    return config


@pytest.fixture
def sample_texts():
    """示例文本"""
    return [
        "Hello, world!",
        "This is a test.",
        "Embedding models are useful for semantic search.",
        "Python is a popular programming language.",
        "Machine learning requires large datasets.",
    ]


@pytest.fixture
def long_text():
    """超长文本（用于测试长度警告）"""
    return "This is a very long text. " * 1000  # 超过 8192 字符
