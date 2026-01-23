"""
测试多轮对话

对应示例 10
"""

import pytest

from ext.llm.chain import FunctionCallingAgent, InMemoryMemory
from .conftest import skip_if_no_api_key


@skip_if_no_api_key
class TestMultiTurnConversation:
    """测试多轮对话"""

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, openai_llm, sample_weather_tool):
        """测试多轮对话"""
        memory = InMemoryMemory(max_messages=50)
        agent = FunctionCallingAgent(
            llm=openai_llm,
            tools=[sample_weather_tool],
            memory=memory,
            system_prompt="你是一个友好的助手，能够记住用户的信息并帮助他们解决问题。",
            max_iterations=5,
        )

        questions = [
            "你好！",
            "我叫 Tom，住在北京",
            "北京今天的天气怎么样？",
            "记住我的名字了吗？",
            "给我推荐一些北京附近的景点",
        ]

        for i, question in enumerate(questions, 1):
            print(f"\n=== 第 {i} 轮对话 ===")
            print(f"用户: {question}")
            response = await agent.ainvoke(question)
            print(f"Agent: {response[:100]}...")
            assert isinstance(response, str)
            assert len(response) > 0

        # 验证记忆
        memory_vars = await memory.load_memory_variables()
        messages = memory_vars["messages"]
        assert len(messages) >= len(questions) * 2  # 每轮对话至少 2 条消息
        print(f"\n✓ Agent 记住了 {len(messages)} 条消息")

    @pytest.mark.asyncio
    async def test_agent_remembers_user_info(self, openai_llm, sample_weather_tool):
        """测试 Agent 记住用户信息"""
        memory = InMemoryMemory(max_messages=50)
        agent = FunctionCallingAgent(
            llm=openai_llm,
            tools=[sample_weather_tool],
            memory=memory,
            max_iterations=5,
        )

        # 第一轮
        response1 = await agent.ainvoke("我叫 Alice，来自上海")
        print(f"第一轮: {response1[:50]}...")

        # 第二轮
        response2 = await agent.ainvoke("我叫什么名字？我来自哪里？")
        print(f"第二轮: {response2[:50]}...")

        # 验证答案中包含用户信息
        assert "Alice" in response2 or "alice" in response2.lower()
        print("✓ Agent 正确记住了用户名字")
