"""
测试输出解析器

对应示例 11
"""

import pytest

from ext.llm.chain import StrOutputParser, JsonOutputParser


class TestStrOutputParser:
    """测试字符串输出解析器"""

    @pytest.mark.asyncio
    async def test_parse_text(self):
        """测试解析普通文本"""
        parser = StrOutputParser()
        result = await parser.ainvoke("Hello World")
        assert result == "Hello World"
        print(f"✓ 解析结果: {result}")

    @pytest.mark.asyncio
    async def test_parse_with_whitespace(self):
        """测试解析带空格的文本"""
        parser = StrOutputParser()
        result = await parser.ainvoke("  Hello World  ")
        assert result == "Hello World"
        print(f"✓ 去除空格: '{result}'")


class TestJsonOutputParser:
    """测试 JSON 输出解析器"""

    @pytest.mark.asyncio
    async def test_parse_json(self):
        """测试解析 JSON"""
        parser = JsonOutputParser()
        result = await parser.ainvoke('{"name": "Alice", "age": 25}')

        assert isinstance(result, dict)
        assert result["name"] == "Alice"
        assert result["age"] == 25
        print(f"✓ JSON 解析结果: {result}")

    @pytest.mark.asyncio
    async def test_parse_json_with_prefix(self):
        """测试解析带前缀的 JSON"""
        parser = JsonOutputParser()
        result = await parser.ainvoke('这是一个 JSON：{"name": "Bob", "age": 30}')

        assert isinstance(result, dict)
        assert result["name"] == "Bob"
        print(f"✓ 带前缀的 JSON 解析: {result}")

    @pytest.mark.asyncio
    async def test_parse_json_with_suffix(self):
        """测试解析带后缀的 JSON"""
        parser = JsonOutputParser()
        result = await parser.ainvoke('{"name": "Charlie", "age": 35} 这是结果')

        assert isinstance(result, dict)
        assert result["name"] == "Charlie"
        print(f"✓ 带后缀的 JSON 解析: {result}")

    @pytest.mark.asyncio
    async def test_parse_invalid_json(self):
        """测试解析无效 JSON"""
        parser = JsonOutputParser()

        with pytest.raises(ValueError, match="Failed to parse JSON"):
            await parser.ainvoke("这不是 JSON")
        print("✓ 无效 JSON 正确抛出异常")

    @pytest.mark.asyncio
    async def test_parse_json_array(self):
        """测试解析 JSON 数组"""
        parser = JsonOutputParser()
        result = await parser.ainvoke('[{"name": "Alice"}, {"name": "Bob"}]')

        # 结果可以是对象或数组（取决于 JSON 的结构）
        assert isinstance(result, (dict, list))
        print(f"✓ JSON 数组解析: {result}")
