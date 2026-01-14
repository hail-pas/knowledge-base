"""
OpenAI Provider 测试

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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")


# 跳过测试的条件
skip_if_no_api_key = pytest.mark.skipif(
    not OPENAI_API_KEY,
    reason="OPENAI_API_KEY not set in environment"
)


class TestSummary(BaseModel):
    """测试用结构化输出模型"""
    summary: str
    key_points: list[str]


@Tool
async def get_current_time() -> str:
    """获取当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_openai_config() -> LLMModelConfig:
    """创建 OpenAI 配置"""
    config_dict = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        config_dict["base_url"] = OPENAI_BASE_URL

    config = LLMModelConfig(
        name="test-openai",
        type=LLMModelTypeEnum.openai,
        model_name=OPENAI_MODEL_NAME,
        config=config_dict,
        capabilities={
            "function_calling": True,
            "json_output": True,
            "streaming": True,
        },
        max_tokens=1000,
        is_enabled=True,
    )
    config._saved_in_db = True
    config.id = 1
    return config


@skip_if_no_api_key
class TestOpenAIProvider:
    """OpenAI Provider 测试"""

    @pytest.mark.asyncio
    async def test_create_model(self):
        """测试创建 OpenAI 模型"""
        config = create_openai_config()
        model = await LLMModelFactory.create(config, use_cache=False)

        assert model is not None
        assert model.model_name == OPENAI_MODEL_NAME

    @pytest.mark.asyncio
    async def test_basic_chat(self):
        """测试基本对话"""
        config = create_openai_config()
        model = await LLMModelFactory.create(config, use_cache=False)
        pydantic_model = model.get_model_for_agent()

        agent = Agent(pydantic_model)
        result = await agent.run("用一句话介绍 Python")

        assert result.output
        assert len(result.output) > 0

    @pytest.mark.asyncio
    async def test_structured_output(self):
        """测试结构化输出"""
        config = create_openai_config()
        model = await LLMModelFactory.create(config, use_cache=False)
        pydantic_model = model.get_model_for_agent()

        agent = Agent(pydantic_model, output_type=TestSummary)
        result = await agent.run("总结：Python 是一种高级编程语言")

        summary = result.output
        assert isinstance(summary, TestSummary)
        assert summary.summary
        assert isinstance(summary.key_points, list)

    @pytest.mark.asyncio
    async def test_tool_calling(self):
        """测试工具调用"""
        config = create_openai_config()
        model = await LLMModelFactory.create(config, use_cache=False)

        if not model.requires_capability("function_calling"):
            pytest.skip("Model does not support function calling")

        pydantic_model = model.get_model_for_agent()
        agent = Agent(pydantic_model, tools=[get_current_time])

        result = await agent.run("现在几点了？")

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
        config = create_openai_config()
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
