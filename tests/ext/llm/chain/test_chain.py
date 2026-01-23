"""
测试 Chain 组合

对应示例 8 和 9
"""

import pytest

from ext.llm.chain import (
    LLM,
    PromptTemplate,
    StrOutputParser,
    RunnablePassthrough,
    RunnableMap,
    RunnableSequence,
)
from .conftest import skip_if_no_api_key


class TestChainCombination:
    """测试 Chain 组合"""

    def test_pipe_operator(self):
        """测试 pipe 操作符"""
        prompt = PromptTemplate.from_template("你是{role}")
        parser = StrOutputParser()

        chain = prompt | parser
        assert isinstance(chain, RunnableSequence)
        assert len(chain.steps) == 2
        print("✓ pipe 操作符组合成功")

    @pytest.mark.asyncio
    async def test_chain_with_output_parser(self, openai_llm):
        """测试带输出解析器的 Chain"""
        prompt = PromptTemplate.from_template("将以下内容翻译成中文：{text}")
        parser = StrOutputParser()

        chain = prompt | openai_llm | parser
        result = await chain.ainvoke({"text": "Hello World"})

        assert isinstance(result, str)
        assert len(result) > 0
        print(f"✓ Chain 执行结果: {result}")

    @pytest.mark.asyncio
    async def test_runnable_map(self):
        """测试 RunnableMap"""
        prompt1 = PromptTemplate.from_template("第一个：{input}")
        prompt2 = PromptTemplate.from_template("第二个：{input}")

        chain = RunnableMap({"first": prompt1, "second": prompt2})
        result = await chain.ainvoke({"input": "测试"})

        assert isinstance(result, dict)
        assert "first" in result
        assert "second" in result
        print(f"✓ RunnableMap 执行成功: {result}")


@skip_if_no_api_key
class TestComplexChain:
    """测试复杂 Chain"""

    @pytest.mark.asyncio
    async def test_complex_chain(self, openai_llm):
        """测试复杂 Chain"""
        prompt = PromptTemplate.from_template("将以下文本翻译成中文：{text}")
        parser = StrOutputParser()

        chain = RunnableMap({"text": RunnablePassthrough()}) | prompt | openai_llm | parser

        result = await chain.ainvoke({"input": "Hello World"})
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"✓ 复杂 Chain 执行成功: {result}")
