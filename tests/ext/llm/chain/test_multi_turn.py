"""
测试多轮对话

对应示例 10
"""

import pytest

from ext.llm.chain import FunctionCallingAgent


@pytest.mark.skip_if_no_api_key
class TestMultiTurnConversation:
    """测试多轮对话"""

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, openai_llm, sample_weather_tool):
        """测试多轮对话"""
        agent = FunctionCallingAgent(
            llm=openai_llm,
            tools=[sample_weather_tool],
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

    @pytest.mark.asyncio
    async def test_agent_multi_round_no_state(self, openai_llm, sample_weather_tool):
        """测试 Agent 多轮调用（无状态）"""
        agent = FunctionCallingAgent(
            llm=openai_llm,
            tools=[sample_weather_tool],
            max_iterations=5,
        )

        response1 = await agent.ainvoke("请简短介绍一下你自己")
        print(f"第一轮: {response1[:50]}...")

        response2 = await agent.ainvoke("再用一句话介绍你可以做什么")
        print(f"第二轮: {response2[:50]}...")

        assert isinstance(response2, str)
        assert len(response2) > 0
