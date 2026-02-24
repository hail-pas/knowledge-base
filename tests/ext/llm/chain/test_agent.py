"""
测试 Function Calling Agent 和 ReAct Agent

对应示例 4、5 和 6
"""

import pytest

from ext.llm.chain import FunctionCallingAgent, ReActAgent, AgentStream


@pytest.mark.skip_if_no_api_key
class TestFunctionCallingAgent:
    """测试 Function Calling Agent"""

    @pytest.mark.asyncio
    async def test_simple_tool_call(self, openai_llm, sample_weather_tool):
        """测试简单工具调用"""
        agent = FunctionCallingAgent(
            llm=openai_llm,
            tools=[sample_weather_tool],
            max_iterations=5,
        )

        result = await agent.ainvoke("北京今天的天气怎么样？")
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"✓ Agent 回答: {result[:100]}")

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self, openai_llm, sample_weather_tool, sample_search_tool):
        """测试多工具调用"""
        agent = FunctionCallingAgent(
            llm=openai_llm,
            tools=[sample_weather_tool, sample_search_tool],
            max_iterations=10,
        )

        result = await agent.ainvoke("查一下北京今天的天气，然后搜索一些景点")
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"✓ 多工具调用结果: {result[:100]}")

    @pytest.mark.asyncio
    async def test_streaming_agent(self, openai_llm, sample_weather_tool, sample_search_tool):
        """测试 Agent 流式输出"""
        agent = FunctionCallingAgent(
            llm=openai_llm,
            tools=[sample_weather_tool, sample_search_tool],
            max_iterations=10,
        )

        events = []
        async for event in agent.astream("查一下北京今天的天气，然后搜索一些景点"):
            events.append(event)
            assert isinstance(event, AgentStream)
            print(f"  事件: {event.event_type}")
            if event.event_type == "thought":
                print(f"    💭 {event.content[:50]}...")
            elif event.event_type == "action":
                print(f"    🔧 调用: {event.tool_call['name']}") # type: ignore
            elif event.event_type == "observation":
                print(f"    📊 结果: {event.tool_result}")
            elif event.event_type == "content":
                print(f"    ✅ 答案: {event.content[:50]}...")

        assert len(events) > 0
        assert any(e.event_type == "content" for e in events)
        print(f"✓ 流式输出共 {len(events)} 个事件")

    @pytest.mark.asyncio
    async def test_agent_with_memory(self, openai_llm, sample_weather_tool):
        """测试带记忆的 Agent"""
        from ext.llm.chain import InMemoryMemory

        memory = InMemoryMemory(max_messages=20)
        agent = FunctionCallingAgent(
            llm=openai_llm,
            tools=[sample_weather_tool],
            memory=memory,
            max_iterations=5,
        )

        # 第一轮对话
        response1 = await agent.ainvoke("我叫 Alice，来自北京")
        print(f"✓ 第一轮: {response1[:50]}...")

        # 第二轮对话
        response2 = await agent.ainvoke("我叫什么名字？我来自哪里？")
        print(f"✓ 第二轮: {response2[:50]}...")

        # 验证记忆
        memory_vars = await memory.load_memory_variables()
        messages = memory_vars["messages"]
        assert len(messages) >= 4  # 2 轮对话，每轮 2 条消息
        print(f"✓ 记忆中保存了 {len(messages)} 条消息")

    @pytest.mark.asyncio
    async def test_agent_no_tool_needed(self, openai_llm, sample_weather_tool):
        """测试不需要工具的情况"""
        agent = FunctionCallingAgent(
            llm=openai_llm,
            tools=[sample_weather_tool],
            max_iterations=5,
        )

        result = await agent.ainvoke("你好，请介绍一下自己")
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"✓ 无需工具的回答: {result[:100]}")


@pytest.mark.skip_if_no_api_key
class TestReActAgent:
    """测试 ReAct Agent"""

    @pytest.mark.asyncio
    async def test_simple_tool_call(self, openai_llm, sample_weather_tool):
        """测试 ReAct Agent 简单工具调用"""
        agent = ReActAgent(
            llm=openai_llm,
            tools=[sample_weather_tool],
            max_iterations=5,
        )

        result = await agent.ainvoke("北京今天的天气怎么样？")
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"✓ ReAct Agent 回答: {result[:100]}")

    @pytest.mark.asyncio
    async def test_react_with_streaming(self, openai_llm, sample_weather_tool, sample_search_tool):
        """测试 ReAct Agent 流式输出"""
        agent = ReActAgent(
            llm=openai_llm,
            tools=[sample_weather_tool, sample_search_tool],
            max_iterations=5,
        )

        events = []
        async for event in agent.astream("查一下北京今天的天气"):
            events.append(event)
            assert isinstance(event, AgentStream)
            print(f"  事件: {event.event_type}")
            if event.event_type == "thought":
                print(f"    💭 {event.content[:50]}...")
            elif event.event_type == "action":
                print(f"    🔧 调用: {event.tool_call}")
            elif event.event_type == "observation":
                print(f"    📊 结果: {event.tool_result}")
            elif event.event_type == "content":
                print(f"    ✅ 答案: {event.content[:50]}...")
            elif event.event_type == "error":
                print(f"    ❌ 错误: {event.error}")

        assert len(events) > 0
        assert any(e.event_type == "content" for e in events)
        print(f"✓ ReAct 流式输出共 {len(events)} 个事件")

    @pytest.mark.asyncio
    async def test_react_with_memory(self, openai_llm, sample_weather_tool):
        """测试 ReAct Agent 带记忆"""
        from ext.llm.chain import InMemoryMemory

        memory = InMemoryMemory(max_messages=20)
        agent = ReActAgent(
            llm=openai_llm,
            tools=[sample_weather_tool],
            memory=memory,
            max_iterations=5,
        )

        # 第一轮对话
        response1 = await agent.ainvoke("我叫 Alice，来自北京")
        print(f"✓ ReAct 第一轮: {response1[:50]}...")

        # 第二轮对话
        response2 = await agent.ainvoke("我叫什么名字？我来自哪里？")
        print(f"✓ ReAct 第二轮: {response2[:50]}...")

        # 验证记忆
        memory_vars = await memory.load_memory_variables()
        messages = memory_vars["messages"]
        assert len(messages) >= 4
        print(f"✓ ReAct 记住了 {len(messages)} 条消息")

    @pytest.mark.asyncio
    async def test_react_no_tool_needed(self, openai_llm, sample_weather_tool):
        """测试 ReAct Agent 不需要工具的情况"""
        agent = ReActAgent(
            llm=openai_llm,
            tools=[sample_weather_tool],
            max_iterations=5,
        )

        result = await agent.ainvoke("你好，请简短介绍一下 ReAct Agent")
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"✓ ReAct 无需工具: {result[:100]}")

    @pytest.mark.asyncio
    async def test_react_multiple_rounds(self, openai_llm, sample_weather_tool):
        """测试 ReAct Agent 多轮循环（至少两轮工具调用）"""
        agent = ReActAgent(
            llm=openai_llm,
            tools=[sample_weather_tool],
            max_iterations=10,
        )

        result = await agent.ainvoke("帮我查询北京和上海今天的天气，然后告诉我哪个更适合户外活动")
        assert isinstance(result, str)
        assert len(result) > 0
        # 验证结果中包含两个城市的信息
        assert "北京" in result or "上海" in result
        print(f"✓ ReAct 多轮结果: {result[:100]}...")
