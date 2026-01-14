"""
DeepSeek Provider 测试

使用真实 API 进行可用性验证测试。
配置从环境变量读取。
"""

import os
import pytest
from pydantic import BaseModel
from pydantic_ai import Agent, Tool

from ext.llm import LLMModelFactory
from ext.ext_tortoise.enums import LLMModelTypeEnum
from ext.ext_tortoise.models.knowledge_base import LLMModelConfig


# 获取环境变量配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
DEEPSEEK_MODEL_NAME = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")


# 跳过测试的条件
skip_if_no_api_key = pytest.mark.skipif(
    not DEEPSEEK_API_KEY,
    reason="DEEPSEEK_API_KEY not set in environment"
)


class TestSummary(BaseModel):
    """测试用结构化输出模型"""
    summary: str
    key_points: list[str]


@Tool
async def get_date() -> str:
    """获取当前日期"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")


def create_deepseek_config() -> LLMModelConfig:
    """创建 DeepSeek 配置"""
    config_dict = {"api_key": DEEPSEEK_API_KEY}
    if DEEPSEEK_BASE_URL:
        config_dict["base_url"] = DEEPSEEK_BASE_URL

    config = LLMModelConfig(
        name="test-deepseek",
        type=LLMModelTypeEnum.deepseek,
        model_name=DEEPSEEK_MODEL_NAME,
        config=config_dict,
        capabilities={
            "function_calling": False,
            "json_output": False,
            "streaming": True,
        },
        max_tokens=1000,
        is_enabled=True,
    )
    config._saved_in_db = True
    config.id = 3
    return config


@skip_if_no_api_key
class TestDeepSeekProvider:
    """DeepSeek Provider 测试"""

    @pytest.mark.asyncio
    async def test_create_model(self):
        """测试创建 DeepSeek 模型"""
        config = create_deepseek_config()
        model = await LLMModelFactory.create(config, use_cache=False)

        assert model is not None
        assert model.model_name == DEEPSEEK_MODEL_NAME

    @pytest.mark.asyncio
    async def test_basic_chat(self):
        """测试基本对话"""
        config = create_deepseek_config()
        model = await LLMModelFactory.create(config, use_cache=False)
        pydantic_model = model.get_model_for_agent()

        agent = Agent(pydantic_model)
        result = await agent.run("用一句话介绍 C++")

        assert result.output
        assert len(result.output) > 0

    @pytest.mark.asyncio
    async def test_structured_output(self):
        """测试结构化输出"""
        config = create_deepseek_config()
        model = await LLMModelFactory.create(config, use_cache=False)
        pydantic_model = model.get_model_for_agent()

        agent = Agent(pydantic_model, output_type=TestSummary)
        result = await agent.run("总结：Rust 是一种系统编程语言，注重安全")

        summary = result.output
        assert isinstance(summary, TestSummary)
        assert summary.summary
        assert isinstance(summary.key_points, list)

    @pytest.mark.asyncio
    async def test_tool_calling(self):
        """测试工具调用"""
        config = create_deepseek_config()
        model = await LLMModelFactory.create(config, use_cache=False)

        if not model.requires_capability("function_calling"):
            pytest.skip("Model does not support function calling")

        pydantic_model = model.get_model_for_agent()
        agent = Agent(pydantic_model, tools=[get_date])

        result = await agent.run("今天是几月几号？")

        assert result.output
        # Check if any messages contain tool calls by looking at the message types
        tool_call_found = False
        for message in result.all_messages():
            if hasattr(message, 'parts'):
                for part in message.parts:
                    if hasattr(part, 'tool_name'):
                        tool_call_found = True
                        break
        assert tool_call_found, "No tool calls were made"

    @pytest.mark.asyncio
    async def test_streaming(self):
        """测试流式输出"""
        config = create_deepseek_config()
        model = await LLMModelFactory.create(config, use_cache=False)

        if not model.requires_capability("streaming"):
            pytest.skip("Model does not support streaming")

        pydantic_model = model.get_model_for_agent()
        agent = Agent(pydantic_model)

        chunks = []
        async with agent.run_stream("数到 3") as result:
            async for chunk in result.stream_text():
                chunks.append(chunk)

        assert len(chunks) > 0
