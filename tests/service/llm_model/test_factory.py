from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel

from ext.ext_tortoise.enums import LLMModelTypeEnum
from ext.ext_tortoise.models.knowledge_base import LLMModelConfig
from service.llm_model.factory import LLMModelFactory


async def test_create_openai_model_from_db_config():
    config = await LLMModelConfig.create(
        name="factory-openai",
        type=LLMModelTypeEnum.openai,
        model_name="gpt-4o-mini",
        api_key="sk-openai",
        base_url="https://api.openai.com/v1",
        max_tokens=2048,
        temperature=0.2,
        top_p=0.9,
        presence_penalty=0.1,
        frequency_penalty=0.2,
        seed=7,
        timeout=45,
        extra_config={"organization": "test-org"},
        is_enabled=True,
    )

    model = await LLMModelFactory.create(config)

    assert isinstance(model, Model)
    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == "gpt-4o-mini"
    assert model.system == "openai"
    assert model.base_url == "https://api.openai.com/v1/"
    assert model.settings == {
        "max_tokens": 2048,
        "temperature": 0.2,
        "top_p": 0.9,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
        "seed": 7,
        "timeout": 45.0,
        "extra_headers": {"OpenAI-Organization": "test-org"},
    }


async def test_create_azure_model_uses_deployment_name():
    config = await LLMModelConfig.create(
        name="factory-azure",
        type=LLMModelTypeEnum.azure_openai,
        model_name="gpt-4o-mini",
        api_key="azure-key",
        base_url="https://example-resource.openai.azure.com",
        extra_config={
            "deployment_name": "chat-deployment",
            "api_version": "2024-10-21",
            "headers": {"x-test": "1"},
        },
        is_enabled=True,
    )

    model = await LLMModelFactory.create(config, use_cache=False)

    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == "chat-deployment"
    assert model.system == "azure"
    assert model.base_url == "https://example-resource.openai.azure.com/openai/"
    assert model.settings == {
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 1.0,
        "timeout": 60.0,
        "extra_headers": {"x-test": "1"},
    }


async def test_create_anthropic_model_from_db_config():
    config = await LLMModelConfig.create(
        name="factory-anthropic",
        type=LLMModelTypeEnum.anthropic,
        model_name="claude-3-5-haiku-latest",
        api_key="anthropic-key",
        base_url="https://api.anthropic.com",
        parallel_tool_calls=True,
        is_enabled=True,
    )

    model = await LLMModelFactory.create(config, use_cache=False)

    assert isinstance(model, AnthropicModel)
    assert model.model_name == "claude-3-5-haiku-latest"
    assert model.system == "anthropic"
    assert model.base_url == "https://api.anthropic.com"
    assert model.settings == {
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 1.0,
        "timeout": 60.0,
        "parallel_tool_calls": True,
    }


async def test_factory_uses_cache_for_same_saved_config():
    config = await LLMModelConfig.create(
        name="factory-cache",
        type=LLMModelTypeEnum.deepseek,
        model_name="deepseek-chat",
        api_key="deepseek-key",
        base_url="https://api.deepseek.com/v1",
        is_enabled=True,
    )

    first = await LLMModelFactory.create(config)
    second = await LLMModelFactory.create_by_id(config.id)

    assert isinstance(first, OpenAIChatModel)
    assert first is second

    LLMModelFactory.clear_cache(config.id)


async def test_factory_reuses_pydantic_ai_settings_and_profile():
    config = await LLMModelConfig.create(
        name="factory-capabilities",
        type=LLMModelTypeEnum.openai,
        model_name="gpt-4o-mini",
        api_key="sk-openai",
        base_url="https://api.openai.com/v1",
        max_tokens=8192,
        temperature=0.3,
        top_p=0.8,
        supports_tools=True,
        supports_image_output=False,
        supports_json_schema_output=True,
        supports_json_object_output=False,
        default_structured_output_mode="native",
        native_output_requires_schema_in_instructions=True,
        parallel_tool_calls=True,
        is_enabled=True,
    )

    settings = LLMModelFactory.get_model_settings(config)
    profile = LLMModelFactory.get_model_profile(config)

    assert settings == {
        "max_tokens": 8192,
        "temperature": 0.3,
        "top_p": 0.8,
        "timeout": 60.0,
        "parallel_tool_calls": True,
    }
    assert profile.supports_tools is True
    assert profile.supports_image_output is False
    assert profile.supports_json_schema_output is True
    assert profile.supports_json_object_output is False
    assert profile.default_structured_output_mode == "native"
    assert profile.native_output_requires_schema_in_instructions is True
