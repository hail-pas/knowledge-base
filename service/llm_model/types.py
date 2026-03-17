from typing import Any

from pydantic import Field, BaseModel, ConfigDict


class BaseExtraConfig(BaseModel):
    """Provider-specific config consumed by native `pydantic_ai` models."""

    model_config = ConfigDict(extra="forbid")

    headers: dict[str, str] = Field(default_factory=dict, description="Extra headers passed via native model settings")
    extra_body: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra request body passed via native model settings",
    )

    @classmethod
    def from_dict(cls, data: dict) -> "BaseExtraConfig":
        valid_data = {k: v for k, v in data.items() if k in cls.model_fields}
        return cls.model_validate(valid_data)


class OpenAIExtraConfig(BaseExtraConfig):
    """No provider-specific overrides for native OpenAI chat."""

    organization: str | None = Field(default=None, description="OpenAI organization header shortcut")
    project: str | None = Field(default=None, description="OpenAI project header shortcut")


class DeepSeekExtraConfig(BaseExtraConfig):
    """No provider-specific overrides for native DeepSeek via OpenAI-compatible provider."""


class AnthropicExtraConfig(BaseExtraConfig):
    """No provider-specific overrides for native Anthropic chat."""


class AzureOpenAIExtraConfig(BaseExtraConfig):
    """Azure OpenAI provider settings required by the native Azure provider."""

    deployment_name: str = Field(default="", description="Azure deployment name")
    api_version: str = Field(default="2024-02-15-preview", description="Azure OpenAI API version")
