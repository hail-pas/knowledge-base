"""
Azure OpenAI Provider 测试

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
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_BASE_URL = os.getenv("AZURE_OPENAI_BASE_URL")
AZURE_OPENAI_MODEL_NAME = os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4o")


# 跳过测试的条件
skip_if_no_api_key = pytest.mark.skipif(
    not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_BASE_URL,
    reason="AZURE_OPENAI_API_KEY or AZURE_OPENAI_BASE_URL not set in environment"
)


class TestSummary(BaseModel):
    """测试用结构化输出模型"""
    summary: str
    key_points: list[str]


@Tool
async def get_temperature(city: str) -> str:
    """获取指定城市的温度（模拟）"""
    return f"{city} 当前温度 22°C"


def create_azure_openai_config() -> LLMModelConfig:
    """创建 Azure OpenAI 配置"""
    config = LLMModelConfig(
        name="test-azure-openai",
        type=LLMModelTypeEnum.azure_openai,
        model_name=AZURE_OPENAI_MODEL_NAME,
        config={
            "api_key": AZURE_OPENAI_API_KEY,
            "endpoint": AZURE_OPENAI_BASE_URL,
        },
        capabilities={
            "function_calling": True,
            "json_output": True,
            "streaming": True,
        },
        max_tokens=1000,
        is_enabled=True,
    )
    config._saved_in_db = True
    config.id = 2
    return config


@skip_if_no_api_key
class TestAzureOpenAIProvider:
    """Azure OpenAI Provider 测试"""

    @pytest.mark.asyncio
    async def test_create_model(self):
        """测试创建 Azure OpenAI 模型"""
        config = create_azure_openai_config()
        model = await LLMModelFactory.create(config, use_cache=False)

        assert model is not None
        assert model.model_name == AZURE_OPENAI_MODEL_NAME

    @pytest.mark.asyncio
    async def test_basic_chat(self):
        """测试基本对话"""
        config = create_azure_openai_config()
        model = await LLMModelFactory.create(config, use_cache=False)
        pydantic_model = model.get_model_for_agent()

        agent = Agent(pydantic_model)
        result = await agent.run("用一句话介绍 Java")

        assert result.output
        assert len(result.output) > 0

    @pytest.mark.asyncio
    async def test_structured_output(self):
        """测试结构化输出"""
        config = create_azure_openai_config()
        model = await LLMModelFactory.create(config, use_cache=False)
        pydantic_model = model.get_model_for_agent()

        agent = Agent(pydantic_model, output_type=TestSummary)
        result = await agent.run("总结：Go 语言是一种静态类型语言")

        summary = result.output
        assert isinstance(summary, TestSummary)
        assert summary.summary
        assert isinstance(summary.key_points, list)

    @pytest.mark.asyncio
    async def test_tool_calling(self):
        """测试工具调用"""
        config = create_azure_openai_config()
        model = await LLMModelFactory.create(config, use_cache=False)

        if not model.requires_capability("function_calling"):
            pytest.skip("Model does not support function calling")

        pydantic_model = model.get_model_for_agent()
        agent = Agent(pydantic_model, tools=[get_temperature])

        result = await agent.run("上海现在多少度？")

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
        config = create_azure_openai_config()
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
