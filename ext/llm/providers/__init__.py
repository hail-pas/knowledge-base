"""
LLM Provider 注册

自动注册所有 LLM providers
"""

from ext.llm.providers.openai import OpenAILLMModel
from ext.llm.providers.azure_openai import AzureOpenAILLMModel
from ext.llm.providers.deepseek import DeepSeekLLMModel
from ext.llm.providers.anthropic import AnthropicLLMModel

from ext.llm.factory import LLMModelFactory
from ext.ext_tortoise.enums import LLMModelTypeEnum

LLMModelFactory.register(LLMModelTypeEnum.openai, OpenAILLMModel)
LLMModelFactory.register(LLMModelTypeEnum.azure_openai, AzureOpenAILLMModel)
LLMModelFactory.register(LLMModelTypeEnum.deepseek, DeepSeekLLMModel)
LLMModelFactory.register(LLMModelTypeEnum.anthropic, AnthropicLLMModel)
