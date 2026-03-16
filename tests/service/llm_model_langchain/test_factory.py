import sys
from types import ModuleType
from typing import Any

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from ext.ext_tortoise.enums import LLMModelTypeEnum
from ext.ext_tortoise.models.knowledge_base import LLMModelConfig
from service.llm_model_langchain.model import CapabilityAwareChatModel
from service.llm_model_langchain.factory import LangChainLLMModelFactory


class FakeChatModel(BaseChatModel):
    model: str | None = None
    model_name: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    azure_deployment: str | None = None
    azure_endpoint: str | None = None
    api_version: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    timeout: float | None = None
    organization: str | None = None
    default_headers: dict[str, str] | None = None
    extra_body: dict[str, Any] | None = None
    model_kwargs: dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)

    def _build_result(self) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="2"))])

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._build_result()

    async def _agenerate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._build_result()

    def bind_tools(self, tools: Any, *, tool_choice: str | None = None, **kwargs: Any) -> str:
        return "bound-tools"

    def with_structured_output(self, schema: dict[str, Any] | type, *, include_raw: bool = False, **kwargs: Any) -> str:
        return "structured-output"


def _install_fake_module(name: str, **attrs) -> None:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


def _remove_fake_module(name: str) -> None:
    sys.modules.pop(name, None)


async def test_create_openai_langchain_model(monkeypatch):
    _install_fake_module("langchain_openai", ChatOpenAI=FakeChatModel, AzureChatOpenAI=FakeChatModel)

    config = await LLMModelConfig.create(
        name="langchain-openai",
        type=LLMModelTypeEnum.openai,
        model_name="gpt-4o-mini",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        max_tokens=2048,
        temperature=0.2,
        top_p=0.9,
        timeout=45,
        extra_config={"organization": "test-org", "extra_body": {"reasoning": {"effort": "low"}}},
        is_enabled=True,
    )

    try:
        model = await LangChainLLMModelFactory.create(config, use_cache=False)
        assert isinstance(model, CapabilityAwareChatModel)
        assert isinstance(model.wrapped, FakeChatModel)
        assert model.wrapped.model == "gpt-4o-mini"
        assert model.wrapped.api_key == "sk-test"
        assert model.wrapped.base_url == "https://api.openai.com/v1"
        assert model.wrapped.temperature == 0.2
        assert model.wrapped.top_p == 0.9
        assert model.wrapped.max_tokens == 2048
        assert model.wrapped.timeout == 45.0
        assert model.wrapped.organization == "test-org"
        assert model.wrapped.default_headers == {"OpenAI-Organization": "test-org"}
        assert model.wrapped.extra_body == {"reasoning": {"effort": "low"}}
        assert model.get_capabilities().supports_tools is False
    finally:
        _remove_fake_module("langchain_openai")


async def test_create_azure_langchain_model():
    _install_fake_module("langchain_openai", ChatOpenAI=FakeChatModel, AzureChatOpenAI=FakeChatModel)

    config = await LLMModelConfig.create(
        name="langchain-azure",
        type=LLMModelTypeEnum.azure_openai,
        model_name="gpt-4o-mini",
        api_key="azure-key",
        base_url="https://example-resource.openai.azure.com",
        extra_config={"deployment_name": "chat-deployment", "api_version": "2024-10-21"},
        is_enabled=True,
    )

    try:
        model = await LangChainLLMModelFactory.create(config, use_cache=False)
        assert isinstance(model, CapabilityAwareChatModel)
        assert isinstance(model.wrapped, FakeChatModel)
        assert model.wrapped.model == "gpt-4o-mini"
        assert model.wrapped.azure_deployment == "chat-deployment"
        assert model.wrapped.api_version == "2024-10-21"
    finally:
        _remove_fake_module("langchain_openai")


async def test_create_anthropic_langchain_model():
    _install_fake_module("langchain_anthropic", ChatAnthropic=FakeChatModel)

    config = await LLMModelConfig.create(
        name="langchain-anthropic",
        type=LLMModelTypeEnum.anthropic,
        model_name="claude-3-5-haiku-latest",
        api_key="anthropic-key",
        base_url="https://api.anthropic.com",
        parallel_tool_calls=True,
        is_enabled=True,
    )

    try:
        model = await LangChainLLMModelFactory.create(config, use_cache=False)
        assert isinstance(model, CapabilityAwareChatModel)
        assert isinstance(model.wrapped, FakeChatModel)
        assert model.wrapped.model_name == "claude-3-5-haiku-latest"
        assert model.wrapped.parallel_tool_calls is True
    finally:
        _remove_fake_module("langchain_anthropic")


async def test_missing_langchain_dependency_raises_clear_error(monkeypatch: pytest.MonkeyPatch):

    config = await LLMModelConfig.create(
        name="langchain-missing",
        type=LLMModelTypeEnum.openai,
        model_name="gpt-4o-mini",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        is_enabled=True,
    )

    def fake_load_class(module_name: str, class_name: str) -> type:
        raise ModuleNotFoundError(f"Missing dependency `{module_name}` for `{class_name}`")

    monkeypatch.setattr(
        LangChainLLMModelFactory,
        "_load_class",
        classmethod(lambda cls, module_name, class_name: fake_load_class(module_name, class_name)),
    )

    with pytest.raises(ModuleNotFoundError, match="langchain_openai"):
        await LangChainLLMModelFactory.create(config, use_cache=False)


async def test_reuses_declared_settings_and_capabilities():
    config = await LLMModelConfig.create(
        name="langchain-capabilities",
        type=LLMModelTypeEnum.openai,
        model_name="gpt-4o-mini",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        max_tokens=8192,
        temperature=0.3,
        top_p=0.8,
        parallel_tool_calls=True,
        supports_tools=True,
        supports_image_output=False,
        supports_json_schema_output=True,
        supports_json_object_output=False,
        default_structured_output_mode="native",
        native_output_requires_schema_in_instructions=True,
        is_enabled=True,
    )

    assert LangChainLLMModelFactory.get_model_kwargs(config) == {
        "temperature": 0.3,
        "max_tokens": 8192,
        "timeout": 60.0,
        "top_p": 0.8,
        "parallel_tool_calls": True,
    }
    capabilities = LangChainLLMModelFactory.get_declared_capabilities(config)

    assert capabilities.supports_tools is True
    assert capabilities.supports_image_output is False
    assert capabilities.supports_json_schema_output is True
    assert capabilities.supports_json_object_output is False
    assert capabilities.default_structured_output_mode == "native"
    assert capabilities.native_output_requires_schema_in_instructions is True


async def test_create_langchain_models_with_real_provider_classes():
    openai_config = await LLMModelConfig.create(
        name="langchain-openai-real",
        type=LLMModelTypeEnum.openai,
        model_name="gpt-4o-mini",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        temperature=0.2,
        timeout=30,
        extra_config={"headers": {"x-test": "1"}},
        is_enabled=True,
    )
    azure_config = await LLMModelConfig.create(
        name="langchain-azure-real",
        type=LLMModelTypeEnum.azure_openai,
        model_name="gpt-4o-mini",
        api_key="azure-key",
        base_url="https://example-resource.openai.azure.com",
        extra_config={"deployment_name": "chat-deployment", "api_version": "2024-10-21"},
        is_enabled=True,
    )
    anthropic_config = await LLMModelConfig.create(
        name="langchain-anthropic-real",
        type=LLMModelTypeEnum.anthropic,
        model_name="claude-3-5-haiku-latest",
        api_key="anthropic-key",
        base_url="https://api.anthropic.com",
        temperature=0.1,
        timeout=30,
        is_enabled=True,
    )

    openai_model = await LangChainLLMModelFactory.create(openai_config, use_cache=False)
    azure_model = await LangChainLLMModelFactory.create(azure_config, use_cache=False)
    anthropic_model = await LangChainLLMModelFactory.create(anthropic_config, use_cache=False)

    assert isinstance(openai_model, CapabilityAwareChatModel)
    assert isinstance(azure_model, CapabilityAwareChatModel)
    assert isinstance(anthropic_model, CapabilityAwareChatModel)
    assert isinstance(openai_model.wrapped, ChatOpenAI)
    assert isinstance(azure_model.wrapped, AzureChatOpenAI)
    assert isinstance(anthropic_model.wrapped, ChatAnthropic)


async def test_capability_aware_chat_model_proxies_calls_and_capabilities():
    _install_fake_module("langchain_openai", ChatOpenAI=FakeChatModel, AzureChatOpenAI=FakeChatModel)

    config = await LLMModelConfig.create(
        name="langchain-capability-aware",
        type=LLMModelTypeEnum.openai,
        model_name="gpt-4o-mini",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        supports_tools=True,
        supports_json_schema_output=True,
        default_structured_output_mode="native",
        is_enabled=True,
    )

    try:
        model = await LangChainLLMModelFactory.create(config, use_cache=False)
        response = model.invoke([HumanMessage(content="What is 1 + 1?")])

        assert response.content == "2"
        assert model.bind_tools([]) == "bound-tools"
        assert model.with_structured_output(dict) == "structured-output"

        capabilities = model.get_capabilities()
        assert capabilities.supports_tools is True
        assert capabilities.supports_json_schema_output is True
        assert capabilities.default_structured_output_mode == "native"
    finally:
        _remove_fake_module("langchain_openai")
