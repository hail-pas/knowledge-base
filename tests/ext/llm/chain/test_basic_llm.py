"""
测试基础 LLM 调用

对应示例 1
"""

import pytest

from ext.llm.chain import LLM
from .conftest import skip_if_no_api_key


@skip_if_no_api_key
class TestBasicLLM:
    """测试基础 LLM 调用"""

    @pytest.mark.asyncio
    async def test_simple_invoke(self, openai_llm):
        """测试简单调用"""
        result = await openai_llm.ainvoke("你好")
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"✓ LLM 回答: {result[:100]}")

    @pytest.mark.asyncio
    async def test_streaming(self, openai_llm):
        """测试流式输出"""
        chunks = []
        async for chunk in openai_llm.astream("请讲一个简短的故事"):
            chunks.append(chunk)
            assert isinstance(chunk, str)

        full_text = "".join(chunks)
        assert len(full_text) > 0
        print(f"✓ 流式输出收到 {len(chunks)} 个块，总计 {len(full_text)} 字符")

    @pytest.mark.asyncio
    async def test_batch_invoke(self, openai_llm):
        """测试批量调用"""
        inputs = ["你好", "介绍一下自己", "Python 是什么"]
        results = await openai_llm.abatch(inputs)

        assert len(results) == len(inputs)
        for result in results:
            assert isinstance(result, str)
            assert len(result) > 0

        print(f"✓ 批量调用 {len(inputs)} 个请求全部成功")
