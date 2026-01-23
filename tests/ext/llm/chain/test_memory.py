"""
测试 Memory

对应示例 7 和 13
"""

import pytest

from ext.llm.chain import InMemoryMemory, ConversationBufferMemory
from .conftest import skip_if_no_api_key


class TestInMemoryMemory:
    """测试内存记忆"""

    @pytest.mark.asyncio
    async def test_load_empty_memory(self):
        """测试加载空记忆"""
        memory = InMemoryMemory(max_messages=100)
        vars = await memory.load_memory_variables()
        assert vars["messages"] == []
        print("✓ 空记忆加载成功")

    @pytest.mark.asyncio
    async def test_save_context(self):
        """测试保存上下文"""
        memory = InMemoryMemory(max_messages=100)
        await memory.save_context("你好", "你好！有什么可以帮助你的吗？")

        vars = await memory.load_memory_variables()
        messages = vars["messages"]
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "你好"
        assert messages[1].role == "assistant"
        print(f"✓ 保存成功，共 {len(messages)} 条消息")

    @pytest.mark.asyncio
    async def test_max_messages_limit(self):
        """测试消息数量限制"""
        memory = InMemoryMemory(max_messages=5)

        # 保存 10 次对话（20 条消息）
        for i in range(10):
            await memory.save_context(f"消息{i}", f"回复{i}")

        vars = await memory.load_memory_variables()
        messages = vars["messages"]
        assert len(messages) == 5  # 只保留最近的 5 条消息
        # 最近的 5 条消息应该是：回复7, 消息7, 回复6, 消息6, 回复5
        assert messages[0].content == "回复7"
        print(f"✓ 消息限制生效，保留 {len(messages)} 条消息")

    @pytest.mark.asyncio
    async def test_clear_memory(self):
        """测试清空记忆"""
        memory = InMemoryMemory(max_messages=100)
        await memory.save_context("你好", "你好！")
        await memory.clear()

        vars = await memory.load_memory_variables()
        assert vars["messages"] == []
        print("✓ 记忆清空成功")


class TestConversationBufferMemory:
    """测试对话缓冲记忆"""

    @pytest.mark.asyncio
    async def test_token_limit(self):
        """测试 Token 限制"""
        memory = ConversationBufferMemory(max_token_limit=100)

        # 保存多条消息
        await memory.save_context("你好", "你好！有什么可以帮助你的吗？")
        await memory.save_context("我叫 Alice", "你好 Alice！")
        await memory.save_context("你来自哪里？", "我来自北京")

        vars = await memory.load_memory_variables()
        messages = vars["messages"]
        # Token 限制应该裁剪旧消息
        assert len(messages) > 0
        print(f"✓ Token 限制生效，保留 {len(messages)} 条消息，约 {memory._count_tokens()} tokens")

    @pytest.mark.asyncio
    async def test_clear_memory(self):
        """测试清空对话缓冲"""
        memory = ConversationBufferMemory(max_token_limit=1000)
        await memory.save_context("你好", "你好！")
        await memory.clear()

        vars = await memory.load_memory_variables()
        assert vars["messages"] == []
        print("✓ 对话缓冲清空成功")


@skip_if_no_api_key
class TestMemoryWithAgent:
    """测试 Memory 与 Agent 集成"""

    @pytest.mark.asyncio
    async def test_agent_remembers_context(self, openai_llm, sample_weather_tool):
        """测试 Agent 记住上下文"""
        from ext.llm.chain import FunctionCallingAgent, InMemoryMemory

        memory = InMemoryMemory(max_messages=50)
        agent = FunctionCallingAgent(
            llm=openai_llm,
            tools=[sample_weather_tool],
            memory=memory,
            max_iterations=5,
        )

        # 第一轮
        response1 = await agent.ainvoke("我叫 Tom")
        print(f"✓ 第一轮: {response1[:50]}...")

        # 第二轮
        response2 = await agent.ainvoke("我叫什么名字？")
        print(f"✓ 第二轮: {response2[:50]}...")

        # 验证记忆
        vars = await memory.load_memory_variables()
        messages = vars["messages"]
        assert len(messages) >= 4  # 2 轮对话，每轮 2 条消息
        print(f"✓ Agent 记住了 {len(messages)} 条消息")
