"""Test helpers that persist provider credentials in the same encrypted form as API writes."""

from ext.ext_tortoise.models.knowledge_base import (
    EmbeddingModelConfig,
    IndexingBackendConfig,
    LLMModelConfig,
)
from util.secret import encrypt_secret, transform_extra_config_secrets


async def create_llm_model_config(**data) -> LLMModelConfig:
    data["api_key"] = encrypt_secret(data.get("api_key"), purpose="llm_model.api_key")
    data["extra_config"] = transform_extra_config_secrets(
        data.get("extra_config"),
        purpose="llm_model.extra_config",
    )
    return await LLMModelConfig.create(**data)


async def create_embedding_model_config(**data) -> EmbeddingModelConfig:
    data["api_key"] = encrypt_secret(data.get("api_key"), purpose="embedding_model.api_key")
    data["extra_config"] = transform_extra_config_secrets(
        data.get("extra_config"),
        purpose="embedding_model.extra_config",
    )
    return await EmbeddingModelConfig.create(**data)


async def create_indexing_backend_config(**data) -> IndexingBackendConfig:
    data["password"] = encrypt_secret(data.get("password"), purpose="indexing_backend.password")
    data["extra_config"] = transform_extra_config_secrets(
        data.get("extra_config"),
        purpose="indexing_backend.extra_config",
    )
    return await IndexingBackendConfig.create(**data)
