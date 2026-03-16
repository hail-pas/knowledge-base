import os

import pytest
from pydantic_ai import Agent

from ext.ext_tortoise.enums import LLMModelTypeEnum
from ext.ext_tortoise.models.knowledge_base import LLMModelConfig
from service.llm_model.factory import LLMModelFactory


def _pick_live_llm_config() -> dict:
    candidates = [
        {
            "name": "test-live-openai-chat",
            "type": LLMModelTypeEnum.openai,
            "model_name": os.getenv("OPENAI_MODEL_NAME"),
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1",
            "extra_config": {},
        },
        {
            "name": "test-live-deepseek-chat",
            "type": LLMModelTypeEnum.deepseek,
            "model_name": os.getenv("DEEPSEEK_MODEL_NAME"),
            "api_key": os.getenv("DEEPSEEK_API_KEY"),
            "base_url": os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1",
            "extra_config": {},
        },
        {
            "name": "test-live-azure-chat",
            "type": LLMModelTypeEnum.azure_openai,
            "model_name": os.getenv("AZURE_OPENAI_MODEL_NAME"),
            "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            "base_url": os.getenv("AZURE_OPENAI_BASE_URL"),
            "extra_config": {
                "deployment_name": os.getenv("AZURE_OPENAI_MODEL_NAME") or "",
                "api_version": os.getenv("AZURE_OPENAI_API_VERSION") or "2024-02-15-preview",
            },
        },
    ]

    for candidate in candidates:
        if candidate["model_name"] and candidate["api_key"] and candidate["base_url"]:
            return candidate

    pytest.skip("No live LLM config found in environment. Run with `source tests/.env`.")


@pytest.mark.asyncio
async def test_live_model_can_chat():
    config_dict = _pick_live_llm_config()

    await LLMModelConfig.filter(name=config_dict["name"]).delete()
    config = await LLMModelConfig.create(
        **config_dict,
        max_tokens=128,
        temperature=0,
        top_p=1,
        timeout=60,
        is_enabled=True,
        is_default=False,
        description="Live chat integration test",
    )

    try:
        model = await LLMModelFactory.create(config, use_cache=False)
        agent = Agent(model=model, system_prompt="Reply briefly and follow the user's format exactly.")

        result = await agent.run(
            "What is 1 + 1? Reply with only the digit.",
            model_settings={"temperature": 0.0, "max_tokens": 16},
        )

        output = result.output.strip()
        assert output
        assert output in {"2", "2."}
    finally:
        LLMModelFactory.clear_cache(config.id)
        await LLMModelConfig.filter(id=config.id).delete()
