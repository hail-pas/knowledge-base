"""
测试 OpenAI LLM Provider 的单元测试（最小可用验证）

不依赖真实 API 的单元测试
"""

# 导入 providers 模块以触发自动注册
import ext.llm.providers

from ext.llm import LLMModelFactory
from ext.llm.base import BaseLLMModel
from ext.llm.providers.openai import OpenAILLMModel
from ext.llm.types import (
    OpenAIExtraConfig,
    LLMRequest,
    ChatMessage,
    ToolDefinition,
    FunctionDefinition,
)
from ext.ext_tortoise.enums import LLMModelTypeEnum
from ext.ext_tortoise.models.knowledge_base import LLMModelConfig


class TestLLMFactory:
    """测试 LLM 工厂"""

    def test_register_provider(self):
        """测试注册 provider"""
        assert LLMModelFactory.has_provider(LLMModelTypeEnum.openai)

    def test_get_registered_model_types(self):
        """测试获取已注册的模型类型"""
        types = LLMModelFactory.get_registered_model_types()
        assert len(types) > 0
        assert LLMModelTypeEnum.openai in types

    def test_get_cache_info(self):
        """测试获取缓存信息"""
        info = LLMModelFactory.get_cache_info()
        assert "cached_count" in info
        assert "cached_ids" in info
        assert "registered_models" in info


class TestOpenAILLMModel:
    """测试 OpenAI LLM 模型"""

    def test_generic_type_inference(self, openai_llm_config):
        """测试泛型类型自动推断"""
        if hasattr(OpenAILLMModel, "__orig_bases__"):
            bases = OpenAILLMModel.__orig_bases__  # type: ignore
            assert len(bases) > 0
            assert any(hasattr(base, "__args__") for base in bases)

    def test_create_model(self, openai_llm_config):
        """测试创建模型实例"""
        model = OpenAILLMModel(
            model_name=openai_llm_config.model_name,
            model_type=openai_llm_config.type.value,
            max_tokens=openai_llm_config.max_tokens,
            api_key=openai_llm_config.api_key,
            base_url=openai_llm_config.base_url,
            supports_chat=openai_llm_config.supports_chat,
            supports_completion=openai_llm_config.supports_completion,
            supports_streaming=openai_llm_config.supports_streaming,
            supports_function_calling=openai_llm_config.supports_function_calling,
            supports_vision=openai_llm_config.supports_vision,
            default_temperature=openai_llm_config.default_temperature,
            default_top_p=openai_llm_config.default_top_p,
            max_retries=openai_llm_config.max_retries,
            timeout=openai_llm_config.timeout,
            extra_config=openai_llm_config.extra_config,
        )

        assert model is not None
        assert model.model_name == openai_llm_config.model_name
        assert model.base_url == openai_llm_config.base_url
        assert model.supports_chat is True
        assert model.supports_streaming is True
        assert model.supports_function_calling is True

    def test_extra_config_type(self, openai_llm_config):
        """测试 extra_config 类型转换"""
        model = OpenAILLMModel(
            model_name=openai_llm_config.model_name,
            model_type=openai_llm_config.type.value,
            max_tokens=openai_llm_config.max_tokens,
            api_key=openai_llm_config.api_key,
            base_url=openai_llm_config.base_url,
            supports_chat=openai_llm_config.supports_chat,
            supports_completion=openai_llm_config.supports_completion,
            supports_streaming=openai_llm_config.supports_streaming,
            supports_function_calling=openai_llm_config.supports_function_calling,
            supports_vision=openai_llm_config.supports_vision,
            default_temperature=openai_llm_config.default_temperature,
            default_top_p=openai_llm_config.default_top_p,
            max_retries=openai_llm_config.max_retries,
            timeout=openai_llm_config.timeout,
            extra_config=openai_llm_config.extra_config,
        )

        assert isinstance(model.extra_config, OpenAIExtraConfig)

    def test_convert_messages_text(self, openai_llm_config):
        """测试消息转换（纯文本）"""
        model = OpenAILLMModel(
            model_name=openai_llm_config.model_name,
            model_type=openai_llm_config.type.value,
            max_tokens=openai_llm_config.max_tokens,
            api_key=openai_llm_config.api_key,
            base_url=openai_llm_config.base_url,
            supports_chat=openai_llm_config.supports_chat,
            supports_completion=openai_llm_config.supports_completion,
            supports_streaming=openai_llm_config.supports_streaming,
            supports_function_calling=openai_llm_config.supports_function_calling,
            supports_vision=openai_llm_config.supports_vision,
            default_temperature=openai_llm_config.default_temperature,
            default_top_p=openai_llm_config.default_top_p,
            max_retries=openai_llm_config.max_retries,
            timeout=openai_llm_config.timeout,
            extra_config=openai_llm_config.extra_config,
        )

        messages = [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Hello!"),
        ]

        converted = model._convert_messages(messages)

        assert len(converted) == 2
        assert converted[0]["role"] == "system"
        assert converted[0]["content"] == "You are a helpful assistant."
        assert converted[1]["role"] == "user"
        assert converted[1]["content"] == "Hello!"

    def test_convert_messages_multimodal(self, openai_llm_config):
        """测试消息转换（多模态）"""
        model = OpenAILLMModel(
            model_name=openai_llm_config.model_name,
            model_type=openai_llm_config.type.value,
            max_tokens=openai_llm_config.max_tokens,
            api_key=openai_llm_config.api_key,
            base_url=openai_llm_config.base_url,
            supports_chat=openai_llm_config.supports_chat,
            supports_completion=openai_llm_config.supports_completion,
            supports_streaming=openai_llm_config.supports_streaming,
            supports_function_calling=openai_llm_config.supports_function_calling,
            supports_vision=openai_llm_config.supports_vision,
            default_temperature=openai_llm_config.default_temperature,
            default_top_p=openai_llm_config.default_top_p,
            max_retries=openai_llm_config.max_retries,
            timeout=openai_llm_config.timeout,
            extra_config=openai_llm_config.extra_config,
        )

        messages = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}},
                ],
            ),
        ]

        converted = model._convert_messages(messages)

        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        assert isinstance(converted[0]["content"], list)
        assert len(converted[0]["content"]) == 2

    def test_convert_messages_with_name_and_tool_call_id(self, openai_llm_config):
        """测试消息转换（包含name和tool_call_id）"""
        model = OpenAILLMModel(
            model_name=openai_llm_config.model_name,
            model_type=openai_llm_config.type.value,
            max_tokens=openai_llm_config.max_tokens,
            api_key=openai_llm_config.api_key,
            base_url=openai_llm_config.base_url,
            supports_chat=openai_llm_config.supports_chat,
            supports_completion=openai_llm_config.supports_completion,
            supports_streaming=openai_llm_config.supports_streaming,
            supports_function_calling=openai_llm_config.supports_function_calling,
            supports_vision=openai_llm_config.supports_vision,
            default_temperature=openai_llm_config.default_temperature,
            default_top_p=openai_llm_config.default_top_p,
            max_retries=openai_llm_config.max_retries,
            timeout=openai_llm_config.timeout,
            extra_config=openai_llm_config.extra_config,
        )

        messages = [
            ChatMessage(role="user", content="Hello", name="user123"),
            ChatMessage(role="tool", content='{"result": "data"}', tool_call_id="call_123"),
        ]

        converted = model._convert_messages(messages)

        assert len(converted) == 2
        assert converted[0]["name"] == "user123"
        assert converted[1]["tool_call_id"] == "call_123"

    def test_convert_tools(self, openai_llm_config):
        """测试工具转换"""
        model = OpenAILLMModel(
            model_name=openai_llm_config.model_name,
            model_type=openai_llm_config.type.value,
            max_tokens=openai_llm_config.max_tokens,
            api_key=openai_llm_config.api_key,
            base_url=openai_llm_config.base_url,
            supports_chat=openai_llm_config.supports_chat,
            supports_completion=openai_llm_config.supports_completion,
            supports_streaming=openai_llm_config.supports_streaming,
            supports_function_calling=openai_llm_config.supports_function_calling,
            supports_vision=openai_llm_config.supports_vision,
            default_temperature=openai_llm_config.default_temperature,
            default_top_p=openai_llm_config.default_top_p,
            max_retries=openai_llm_config.max_retries,
            timeout=openai_llm_config.timeout,
            extra_config=openai_llm_config.extra_config,
        )

        tools = [
            ToolDefinition(
                type="function",
                function=FunctionDefinition(
                    name="get_weather",
                    description="Get weather",
                    parameters={"type": "object", "properties": {}},
                ),
            ),
        ]

        converted = model._convert_tools(tools)

        assert converted is not None
        assert len(converted) == 1
        assert converted[0]["type"] == "function"
        assert converted[0]["function"]["name"] == "get_weather"

    def test_convert_tools_none(self, openai_llm_config):
        """测试工具转换（None）"""
        model = OpenAILLMModel(
            model_name=openai_llm_config.model_name,
            model_type=openai_llm_config.type.value,
            max_tokens=openai_llm_config.max_tokens,
            api_key=openai_llm_config.api_key,
            base_url=openai_llm_config.base_url,
            supports_chat=openai_llm_config.supports_chat,
            supports_completion=openai_llm_config.supports_completion,
            supports_streaming=openai_llm_config.supports_streaming,
            supports_function_calling=openai_llm_config.supports_function_calling,
            supports_vision=openai_llm_config.supports_vision,
            default_temperature=openai_llm_config.default_temperature,
            default_top_p=openai_llm_config.default_top_p,
            max_retries=openai_llm_config.max_retries,
            timeout=openai_llm_config.timeout,
            extra_config=openai_llm_config.extra_config,
        )

        result = model._convert_tools(None)
        assert result is None

    def test_convert_tool_choice(self, openai_llm_config):
        """测试工具选择转换"""
        model = OpenAILLMModel(
            model_name=openai_llm_config.model_name,
            model_type=openai_llm_config.type.value,
            max_tokens=openai_llm_config.max_tokens,
            api_key=openai_llm_config.api_key,
            base_url=openai_llm_config.base_url,
            supports_chat=openai_llm_config.supports_chat,
            supports_completion=openai_llm_config.supports_completion,
            supports_streaming=openai_llm_config.supports_streaming,
            supports_function_calling=openai_llm_config.supports_function_calling,
            supports_vision=openai_llm_config.supports_vision,
            default_temperature=openai_llm_config.default_temperature,
            default_top_p=openai_llm_config.default_top_p,
            max_retries=openai_llm_config.max_retries,
            timeout=openai_llm_config.timeout,
            extra_config=openai_llm_config.extra_config,
        )

        assert model._convert_tool_choice("auto") == "auto"
        assert model._convert_tool_choice(None) is None
        assert model._convert_tool_choice({"type": "function", "function": {"name": "get_weather"}}) == {
            "type": "function",
            "function": {"name": "get_weather"},
        }

    def test_should_retry(self, openai_llm_config):
        """测试重试逻辑"""
        model = OpenAILLMModel(
            model_name=openai_llm_config.model_name,
            model_type=openai_llm_config.type.value,
            max_tokens=openai_llm_config.max_tokens,
            api_key=openai_llm_config.api_key,
            base_url=openai_llm_config.base_url,
            supports_chat=openai_llm_config.supports_chat,
            supports_completion=openai_llm_config.supports_completion,
            supports_streaming=openai_llm_config.supports_streaming,
            supports_function_calling=openai_llm_config.supports_function_calling,
            supports_vision=openai_llm_config.supports_vision,
            default_temperature=openai_llm_config.default_temperature,
            default_top_p=openai_llm_config.default_top_p,
            max_retries=openai_llm_config.max_retries,
            timeout=openai_llm_config.timeout,
            extra_config=openai_llm_config.extra_config,
        )

        assert model.should_retry(429, 0) is True
        assert model.should_retry(429, 3) is False
        assert model.should_retry(500, 0) is True
        assert model.should_retry(400, 0) is False

    def test_get_retry_delay(self, openai_llm_config):
        """测试重试延迟"""
        model = OpenAILLMModel(
            model_name=openai_llm_config.model_name,
            model_type=openai_llm_config.type.value,
            max_tokens=openai_llm_config.max_tokens,
            api_key=openai_llm_config.api_key,
            base_url=openai_llm_config.base_url,
            supports_chat=openai_llm_config.supports_chat,
            supports_completion=openai_llm_config.supports_completion,
            supports_streaming=openai_llm_config.supports_streaming,
            supports_function_calling=openai_llm_config.supports_function_calling,
            supports_vision=openai_llm_config.supports_vision,
            default_temperature=openai_llm_config.default_temperature,
            default_top_p=openai_llm_config.default_top_p,
            max_retries=openai_llm_config.max_retries,
            timeout=openai_llm_config.timeout,
            extra_config=openai_llm_config.extra_config,
        )

        assert model.get_retry_delay(0) == 1
        assert model.get_retry_delay(1) == 2
        assert model.get_retry_delay(2) == 4
