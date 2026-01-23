"""
LLM 模块的 conftest.py

定义测试所需的 fixtures
"""

import os
import pytest
from ext.ext_tortoise.enums import LLMModelTypeEnum
from ext.ext_tortoise.models.knowledge_base import LLMModelConfig


# 获取环境变量配置
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")


# 跳过测试的条件
skip_if_no_api_key = pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY not set in environment")


@pytest.fixture
def openai_llm_config_dict():
    """返回 OpenAI LLM 配置字典（不创建对象）"""
    return {
        "name": "test-openai-llm",
        "type": LLMModelTypeEnum.openai,
        "model_name": OPENAI_MODEL_NAME or "gpt-3.5-turbo",
        "api_key": OPENAI_API_KEY,
        "base_url": OPENAI_BASE_URL or "https://api.openai.com",
        "max_tokens": 4096,
        "supports_chat": True,
        "supports_completion": False,
        "supports_streaming": True,
        "supports_function_calling": True,
        "supports_vision": False,
        "default_temperature": 0.7,
        "default_top_p": 1.0,
        "max_retries": 3,
        "timeout": 60,
        "extra_config": {},
        "is_enabled": True,
        "is_default": False,
        "description": "测试用OpenAI LLM配置",
    }


@pytest.fixture
async def openai_llm_config(openai_llm_config_dict):
    """创建并保存 OpenAI LLM 配置到数据库"""
    await LLMModelConfig.filter(name=openai_llm_config_dict["name"]).delete()

    config = await LLMModelConfig.create(**openai_llm_config_dict)
    return config


@pytest.fixture
async def openai_llm_config_with_extra(openai_llm_config_dict):
    """创建带有额外配置的 OpenAI LLM 配置"""
    extra_dict = openai_llm_config_dict.copy()
    extra_dict["name"] = "test-openai-llm-extra"
    extra_dict["extra_config"] = {
        "endpoint": "/v1/chat/completions",
        "headers": {"X-Custom-Header": "test-value"},
    }

    await LLMModelConfig.filter(name=extra_dict["name"]).delete()

    config = await LLMModelConfig.create(**extra_dict)
    return config


@pytest.fixture
def sample_chat_messages():
    """示例聊天消息"""
    from ext.llm.types import ChatMessage

    return [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="Hello, how are you?"),
    ]


@pytest.fixture
def sample_function_definition():
    """示例函数定义"""
    from ext.llm.types import FunctionDefinition, ToolDefinition

    return ToolDefinition(
        type="function",
        function=FunctionDefinition(
            name="get_weather",
            description="Get the current weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
        ),
    )
