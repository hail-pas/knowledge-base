"""Config API 测试共享配置和 Fixtures"""

import pytest
import time
import random
import string


def _generate_unique_id():
    """生成唯一标识符"""
    return f"{int(time.time() * 1000)}{''.join(random.choices(string.ascii_lowercase, k=4))}"


@pytest.fixture
def embedding_config_data():
    """EmbeddingModelConfig 测试数据"""
    unique_id = _generate_unique_id()
    return {
        "name": f"test_embedding_{unique_id}",
        "type": "openai",
        "model_name_or_path": f"text-embedding-3-small-{unique_id}",
        "dimension": 768,
        "max_batch_size": 512,
        "max_token_per_text": 512,
        "max_token_per_request": 262144,
        "config": {
            "api_key": "test_key",
            "base_url": "https://api.openai.com/v1",
        },
        "is_enabled": True,
        "is_default": False,
        "description": "Test embedding config",
    }


@pytest.fixture
def indexing_backend_config_data():
    """IndexingBackendConfig 测试数据"""
    unique_id = _generate_unique_id()
    return {
        "name": f"test_indexing_{unique_id}",
        "type": "elasticsearch",
        "host": "localhost",
        "port": 9200,
        "username": "elastic",
        "password": "changeme",
        "secure": False,
        "config": {
            "index_prefix": "kb_",
        },
        "is_enabled": True,
        "is_default": False,
        "description": "Test indexing backend config",
    }


@pytest.fixture
def llm_model_config_data():
    """LLMModelConfig 测试数据"""
    unique_id = _generate_unique_id()
    return {
        "name": f"test_llm_{unique_id}",
        "type": "openai",
        "model_name": f"gpt-4o-mini-{unique_id}",
        "config": {
            "api_key": "test_key",
            "base_url": "https://api.openai.com/v1",
            "temperature": 0.7,
        },
        "capabilities": {
            "function_calling": True,
            "json_output": True,
        },
        "max_tokens": 4096,
        "max_retries": 3,
        "timeout": 60,
        "rate_limit": 60,
        "is_enabled": True,
        "is_default": False,
        "description": "Test LLM config",
    }
