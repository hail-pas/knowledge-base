"""
OpenAI LLM Provider 集成测试

使用真实 API 进行测试，配置从环境变量读取
"""

import pytest

# 导入 providers 模块以触发自动注册
import ext.llm.providers

from ext.llm import LLMModelFactory
from ext.llm.types import ChatMessage, ToolDefinition, FunctionDefinition, ToolCall, LLMRequest
from tests.ext.llm.conftest import (
    openai_llm_config,
    openai_llm_config_with_extra,
    sample_chat_messages,
    sample_function_definition,
    skip_if_no_api_key,
)


@skip_if_no_api_key
class TestOpenAILLMIntegration:
    """OpenAI LLM 集成测试（需要真实 API）"""

    @pytest.mark.asyncio
    async def test_create_model_from_config(self, openai_llm_config):
        """测试从配置创建模型"""
        model = await LLMModelFactory.create(openai_llm_config, use_cache=False)

        assert model is not None
        assert model.model_name == openai_llm_config.model_name

    @pytest.mark.asyncio
    async def test_chat_basic(self, openai_llm_config, sample_chat_messages):
        """测试基础对话"""
        model = await LLMModelFactory.create(openai_llm_config, use_cache=False)

        request = LLMRequest(messages=sample_chat_messages)
        response = await model.chat(request)

        assert response.content is not None
        assert len(response.content) > 0
        assert response.role == "assistant"
        assert response.usage.total_tokens > 0

    @pytest.mark.asyncio
    async def test_chat_stream_basic(self, openai_llm_config, sample_chat_messages):
        """测试流式对话"""
        model = await LLMModelFactory.create(openai_llm_config, use_cache=False)

        request = LLMRequest(messages=sample_chat_messages)

        chunks = []
        async for chunk in model.chat_stream(request): # type: ignore
            chunks.append(chunk)

        assert len(chunks) > 0
        assert any("content" in chunk.delta for chunk in chunks)
        assert any(chunk.finish_reason for chunk in chunks)

    @pytest.mark.asyncio
    async def test_chat_with_temperature(self, openai_llm_config):
        """测试温度参数"""
        model = await LLMModelFactory.create(openai_llm_config, use_cache=False)

        messages = [ChatMessage(role="user", content="Say hello")]

        request_high = LLMRequest(messages=messages, temperature=1.0)
        response_high = await model.chat(request_high)

        request_low = LLMRequest(messages=messages, temperature=0.0)
        response_low = await model.chat(request_low)

        assert response_high.content is not None
        assert response_low.content is not None

    @pytest.mark.asyncio
    async def test_chat_with_max_tokens(self, openai_llm_config):
        """测试最大token数"""
        model = await LLMModelFactory.create(openai_llm_config, use_cache=False)

        messages = [ChatMessage(role="user", content="Tell me a short story")]

        request = LLMRequest(messages=messages, max_tokens=50)
        response = await model.chat(request)

        assert response.content is not None
        assert response.usage.completion_tokens <= 50

    @pytest.mark.asyncio
    async def test_chat_with_stop(self, openai_llm_config):
        """测试停止序列"""
        model = await LLMModelFactory.create(openai_llm_config, use_cache=False)

        messages = [ChatMessage(role="user", content="Count to 10")]

        request = LLMRequest(messages=messages, stop=["5"])
        response = await model.chat(request)

        assert response.content is not None
        assert "5" not in response.content or response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_chat_with_function_call(self, openai_llm_config, sample_function_definition):
        """测试函数调用"""
        model = await LLMModelFactory.create(openai_llm_config, use_cache=False)

        messages = [
            ChatMessage(role="user", content="What's the weather in San Francisco?"),
        ]

        request = LLMRequest(
            messages=messages,
            tools=[sample_function_definition],
            tool_choice="auto",
        )
        response = await model.chat(request)

        assert response.content is not None
        assert response.tool_calls is not None
        assert len(response.tool_calls) > 0
        assert response.tool_calls[0].type == "function"
        assert response.tool_calls[0].function["name"] == "get_weather"
        assert "arguments" in response.tool_calls[0].function

    @pytest.mark.asyncio
    async def test_chat_with_tool_call_required(self, openai_llm_config, sample_function_definition):
        """测试强制工具调用"""
        model = await LLMModelFactory.create(openai_llm_config, use_cache=False)

        messages = [
            ChatMessage(role="user", content="Get the weather"),
        ]

        request = LLMRequest(
            messages=messages,
            tools=[sample_function_definition],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )
        response = await model.chat(request)

        assert response.tool_calls is not None
        assert len(response.tool_calls) > 0
        assert response.tool_calls[0].function["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_chat_with_multiple_tools(self, openai_llm_config):
        """测试多工具调用"""
        model = await LLMModelFactory.create(openai_llm_config, use_cache=False)

        tools = [
            ToolDefinition(
                type="function",
                function=FunctionDefinition(
                    name="get_weather",
                    description="Get weather",
                    parameters={"type": "object", "properties": {}},
                ),
            ),
            ToolDefinition(
                type="function",
                function=FunctionDefinition(
                    name="get_time",
                    description="Get current time",
                    parameters={"type": "object", "properties": {}},
                ),
            ),
        ]

        messages = [ChatMessage(role="user", content="What's the weather and time?")]

        request = LLMRequest(messages=messages, tools=tools, tool_choice="auto")
        response = await model.chat(request)

        assert response.content is not None
        if response.tool_calls:
            assert all(tc.type == "function" for tc in response.tool_calls)

    @pytest.mark.asyncio
    async def test_chat_with_tool_response(self, openai_llm_config, sample_function_definition):
        """测试工具响应对话"""
        model = await LLMModelFactory.create(openai_llm_config, use_cache=False)

        messages = [
            ChatMessage(role="user", content="What's the weather in Beijing?"),
        ]

        request = LLMRequest(
            messages=messages,
            tools=[sample_function_definition],
            tool_choice="auto",
        )
        response = await model.chat(request)

        assert response.tool_calls is not None
        tool_call = response.tool_calls[0]
        tool_call_id = tool_call.id

        tool_response_messages = [
            ChatMessage(
                role="tool",
                content='{"temperature": 20, "condition": "sunny"}',
                tool_call_id=tool_call_id,
            ),
        ]

        request2 = LLMRequest(
            messages=messages + tool_response_messages,
            tools=[sample_function_definition],
        )
        response2 = await model.chat(request2)

        assert response2.content is not None
        assert len(response2.content) > 0

    @pytest.mark.asyncio
    async def test_chat_stream_with_function_call(self, openai_llm_config, sample_function_definition):
        """测试流式对话中的函数调用"""
        model = await LLMModelFactory.create(openai_llm_config, use_cache=False)

        messages = [
            ChatMessage(role="user", content="What's the weather in Shanghai?"),
        ]

        request = LLMRequest(
            messages=messages,
            tools=[sample_function_definition],
            tool_choice="auto",
        )

        chunks = []
        async for chunk in model.chat_stream(request): # type: ignore
            chunks.append(chunk)

        assert len(chunks) > 0
        assert any("tool_calls" in chunk.delta for chunk in chunks)

    @pytest.mark.asyncio
    async def test_model_caching(self, openai_llm_config):
        """测试模型缓存"""
        model1 = await LLMModelFactory.create(openai_llm_config, use_cache=True)
        model2 = await LLMModelFactory.create(openai_llm_config, use_cache=True)

        assert model1 is model2

        LLMModelFactory.clear_cache(openai_llm_config.id)

        model3 = await LLMModelFactory.create(openai_llm_config, use_cache=True)
        assert model1 is not model3

    @pytest.mark.asyncio
    async def test_chat_multiple_requests(self, openai_llm_config):
        """测试多次请求"""
        model = await LLMModelFactory.create(openai_llm_config, use_cache=False)

        for i in range(3):
            messages = [ChatMessage(role="user", content=f"Say {i}")]
            request = LLMRequest(messages=messages)
            response = await model.chat(request)

            assert response.content is not None
            assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_chat_with_system_message(self, openai_llm_config):
        """测试系统消息"""
        model = await LLMModelFactory.create(openai_llm_config, use_cache=False)

        messages = [
            ChatMessage(role="system", content="You are a pirate. Always speak like a pirate."),
            ChatMessage(role="user", content="Hello!"),
        ]

        request = LLMRequest(messages=messages)
        response = await model.chat(request)

        assert response.content is not None
        assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_chat_with_multi_turn(self, openai_llm_config):
        """测试多轮对话"""
        model = await LLMModelFactory.create(openai_llm_config, use_cache=False)

        messages = [
            ChatMessage(role="user", content="My name is Alice."),
        ]

        request1 = LLMRequest(messages=messages)
        response1 = await model.chat(request1)

        messages.append(ChatMessage(role="assistant", content=response1.content))
        messages.append(ChatMessage(role="user", content="What's my name?"))

        request2 = LLMRequest(messages=messages)
        response2 = await model.chat(request2)

        assert "Alice" in response2.content
