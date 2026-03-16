import pytest
from pydantic import ValidationError

from service.llm_model.types import AzureOpenAIExtraConfig, BaseExtraConfig, OpenAIExtraConfig


def test_base_extra_config_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        BaseExtraConfig(unknown_field="x")


def test_azure_extra_config_accepts_provider_specific_fields():
    config = AzureOpenAIExtraConfig(deployment_name="chat-deployment", api_version="2024-10-21")

    assert config.model_dump() == {
        "headers": {},
        "extra_body": {},
        "deployment_name": "chat-deployment",
        "api_version": "2024-10-21",
    }


def test_openai_extra_config_accepts_legacy_organization_field():
    config = OpenAIExtraConfig(organization="test-org")

    assert config.organization == "test-org"
