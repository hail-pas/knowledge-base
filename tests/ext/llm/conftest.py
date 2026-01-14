"""
LLM 模块测试配置文件

提供测试用的基础 fixtures。
"""

import pytest
from ext.llm import LLMModelFactory


@pytest.fixture(scope="function", autouse=True)
def cleanup_llm_cache():
    """
    每个测试函数执行后清理 LLM 缓存

    防止测试之间的缓存干扰
    """
    yield

    # 测试结束后清理缓存
    LLMModelFactory.clear_cache()


@pytest.fixture(scope="session", autouse=True)
def cleanup_all_llm_resources():
    """
    所有测试结束后清理所有 LLM 资源
    """
    yield

    import asyncio

    # 异步关闭所有模型实例
    async def cleanup():
        await LLMModelFactory.close_all()

    # 在测试会话结束时运行
    try:
        asyncio.run(cleanup())
    except Exception:
        # 忽略清理过程中的错误
        pass


def pytest_configure(config):
    """
    配置 pytest，注册自定义标记
    """
    config.addinivalue_line(
        "markers", "openai: 标记需要 OpenAI API 的测试"
    )
    config.addinivalue_line(
        "markers", "azure_openai: 标记需要 Azure OpenAI API 的测试"
    )
    config.addinivalue_line(
        "markers", "deepseek: 标记需要 DeepSeek API 的测试"
    )
