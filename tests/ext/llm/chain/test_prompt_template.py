"""
测试 Prompt Template

对应示例 2
"""

import pytest

from ext.llm.chain import PromptTemplate, LLM
from .conftest import skip_if_no_api_key


class TestPromptTemplate:
    """测试 Prompt Template"""

    def test_from_template(self):
        """测试从字符串创建模板"""
        template = PromptTemplate.from_template("你是一个{role}。{task}")
        assert template.template == "你是一个{role}。{task}"
        assert "role" in template.input_variables
        assert "task" in template.input_variables
        print("✓ 创建模板成功")

    def test_format(self):
        """测试格式化模板"""
        template = PromptTemplate.from_template("你是一个{role}。{task}")
        formatted = template.format(role="翻译", task="翻译成中文")
        assert "翻译" in formatted
        print(f"✓ 格式化成功: {formatted}")

    def test_extract_variables(self):
        """测试提取变量"""
        template = PromptTemplate.from_template("你是{role}，处理{input1}和{input2}")
        assert set(template.input_variables) == {"role", "input1", "input2"}
        print("✓ 变量提取成功")

    def test_format_missing_variable(self):
        """测试缺少变量时抛出异常"""
        template = PromptTemplate.from_template("你是{role}，处理{input}")
        with pytest.raises(ValueError, match="Missing input variables"):
            template.format(role="翻译")
        print("✓ 缺少变量时正确抛出异常")


@skip_if_no_api_key
class TestPromptTemplateWithLLM:
    """测试 Prompt Template 与 LLM 组合"""

    @pytest.mark.asyncio
    async def test_prompt_with_llm(self, openai_llm):
        """测试 Prompt 与 LLM 组合"""
        prompt = PromptTemplate.from_template("将以下文本翻译成中文：{text}")
        chain = prompt | openai_llm

        result = await chain.ainvoke({"text": "Hello World"})
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"✓ Prompt + LLM 组合成功: {result[:100]}")
