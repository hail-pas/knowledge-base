"""
Provider 特定的 extra_config 类型

为不同的 embedding provider 定义类型化的配置
"""

from typing import Optional
from pydantic import Field
from ext.embedding.types import BaseExtraConfig


class OpenAIExtraConfig(BaseExtraConfig):
    """
    OpenAI 特定配置

    扩展自 BaseExtraConfig，添加 OpenAI 特有的字段
    """

    encoding_format: str | None = Field(default=None, description="编码格式：float 或 base64")


class AzureOpenAIExtraConfig(BaseExtraConfig):
    """
    Azure OpenAI 特定配置
    """

    deployment_name: str = Field(default="", description="部署名称")
    api_version: str = Field(default="2024-02-15-preview", description="API版本")


class CohereExtraConfig(BaseExtraConfig):
    """
    Cohere 特定配置
    """

    input_type: str | None = Field(default=None, description="输入类型：search_document 或 search_query")
    truncate: str | None = Field(default=None, description="截断方式：NONE、END 或 START")
