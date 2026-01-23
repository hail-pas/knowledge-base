"""
Chain 测试 conftest

定义测试所需的 fixtures
"""

import pytest
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
# sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ext.llm import LLMModelFactory
# from ext.llm.chain import tool
from ext.ext_tortoise.enums import LLMModelTypeEnum
from ext.ext_tortoise.models.knowledge_base import LLMModelConfig
import os


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
        "name": "test-chain-openai-llm",
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
        "description": "测试 Chain 模块的 OpenAI LLM 配置",
    }


@pytest.fixture
async def openai_llm_config(openai_llm_config_dict):
    """创建并保存 OpenAI LLM 配置到数据库"""
    await LLMModelConfig.filter(name=openai_llm_config_dict["name"]).delete()
    config = await LLMModelConfig.create(**openai_llm_config_dict)
    return config


@pytest.fixture
async def openai_llm(openai_llm_config):
    """创建 OpenAI LLM 实例"""
    from ext.llm.chain import LLM

    model = await LLMModelFactory.create(openai_llm_config, use_cache=False)
    return LLM(model)


@pytest.fixture
def sample_weather_tool():
    """示例天气工具"""
    from ext.llm.chain import tool

    @tool
    def get_weather(location: str) -> str:
        """获取指定位置的天气信息

        Args:
            location: 城市名称

        Returns:
            天气描述
        """
        weather_data = {
            "北京": "晴，25°C",
            "上海": "多云，28°C",
            "广州": "阴，30°C",
        }
        return weather_data.get(location, f"{location}的天气数据不可用")

    return get_weather


@pytest.fixture
def sample_search_tool():
    """示例搜索工具"""
    from ext.llm.chain import tool

    @tool
    def search_web(query: str) -> str:
        """搜索网络

        Args:
            query: 搜索关键词

        Returns:
            搜索结果
        """
        return f"关于'{query}'的搜索结果..."

    return search_web
