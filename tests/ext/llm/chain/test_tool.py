"""
测试 Tool 定义

对应示例 3
"""

import pytest

from ext.llm.chain import tool


class TestToolDefinition:
    """测试 Tool 定义"""

    def test_tool_decorator(self):
        """测试 @tool 装饰器"""

        @tool
        def get_weather(location: str) -> str:
            """获取天气"""
            return f"{location}天气：晴"

        assert get_weather.name == "get_weather" # type: ignore
        assert "获取天气" in get_weather.description # type: ignore
        assert "location" in get_weather.parameters["properties"] # type: ignore
        assert "location" in get_weather.parameters["required"] # type: ignore
        print(f"✓ 工具定义: {get_weather.name}") # type: ignore
        print(f"✓ 工具描述: {get_weather.description}") # type: ignore

    def test_tool_with_custom_name(self):
        """测试自定义工具名称"""

        @tool(name="custom_name") # type: ignore
        def get_weather(location: str) -> str:
            """获取天气"""
            return f"{location}天气：晴"

        assert get_weather.name == "custom_name" # type: ignore
        print(f"✓ 自定义名称: {get_weather.name}") # type: ignore

    @pytest.mark.asyncio
    async def test_tool_invocation(self, sample_weather_tool):
        """测试工具调用"""
        result = await sample_weather_tool.ainvoke(location="北京")
        assert "晴" in result
        assert "25°C" in result
        print(f"✓ 工具调用结果: {result}")
