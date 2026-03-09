"""
集成测试：验证 TraceManager 和 StepContext 功能

测试内容：
1. Trace 生命周期
2. Step 生命周期
3. Artifact 创建
4. 事件发送
5. 数据库持久化
"""

import asyncio
import pytest
from datetime import datetime
from uuid import uuid4

from service.chat.runtime import TraceManager
from service.chat.enums import StepTypeEnum, ArtifactTypeEnum


class MockWebSocket:
    """模拟 WebSocket 连接"""

    def __init__(self):
        self.events = []

    async def send_json(self, event):
        """模拟发送 JSON 事件"""
        self.events.append(event)


async def test_trace_lifecycle():
    """测试 Trace 生命周期"""

    print("测试1: Trace 生命周期")
    print("-" * 60)

    # 创建模拟 WebSocket
    websocket = MockWebSocket()

    # 创建 TraceManager
    async with TraceManager(
        user_id="user_123",
        session_id="session_456",
        chat_mode="rag",
        websocket=websocket,
    ) as trace:
        # 验证 Trace 已创建
        assert trace.trace_id.startswith("trace_")
        assert trace.user_id == "user_123"
        assert trace.status.value == "running"

        # 创建一个步骤
        async with trace.step_context(
            step_type=StepTypeEnum.retrieval,
            step_name="测试步骤",
            input={"query": "测试"},
        ) as step:
            # 创建产物
            artifact_id = step.create_artifact(
                artifact_type=ArtifactTypeEnum.text,
                artifact_data={"text": "测试产物"},
            )

            # 验证产物已创建
            assert artifact_id.startswith("artifact_")
            assert len(step.artifact_ids) == 1

            # 设置输出
            step.set_output({"status": "success"})

        # 验证步骤已添加
        assert len(trace.steps) == 1
        assert trace.steps[0]["status"] == "completed"
        assert trace.completed_steps == 1

    # 验证 Trace 已完成
    assert trace.status.value == "completed"

    # 验证事件已发送
    assert len(websocket.events) == 5  # trace_start, step_start, artifact_created, step_complete, trace_complete

    print("✅ Trace 生命周期测试通过！")
    print()


async def test_nested_steps():
    """测试嵌套步骤"""

    print("测试2: 嵌套步骤")
    print("-" * 60)

    websocket = MockWebSocket()

    async with TraceManager(
        user_id="user_123",
        session_id="session_456",
        websocket=websocket,
    ) as trace:
        # 父步骤
        async with trace.step_context(
            step_type=StepTypeEnum.retrieval,
            step_name="父步骤",
            input={},
        ) as parent_step:
            # 子步骤1
            async with trace.step_context(
                step_type=StepTypeEnum.retrieval,
                step_name="子步骤1",
                input={},
                parent_step_id=parent_step.step_id,
            ) as child_step1:
                child_step1.set_output({"result": "child1"})

            # 子步骤2
            async with trace.step_context(
                step_type=StepTypeEnum.retrieval,
                step_name="子步骤2",
                input={},
                parent_step_id=parent_step.step_id,
            ) as child_step2:
                child_step2.set_output({"result": "child2"})

            parent_step.set_output({"result": "parent"})

    # 验证嵌套关系
    assert len(trace.steps) == 3  # 1个父步骤 + 2个子步骤

    # 验证 parent_step_id
    assert trace.steps[1]["parent_step_id"] == trace.steps[0]["step_id"]  # 子步骤1的父步骤
    assert trace.steps[2]["parent_step_id"] == trace.steps[0]["step_id"]  # 子步骤2的父步骤

    print("✅ 嵌套步骤测试通过！")
    print()


async def test_error_handling():
    """测试错误处理"""

    print("测试3: 错误处理")
    print("-" * 60)

    websocket = MockWebSocket()

    async with TraceManager(
        user_id="user_123",
        session_id="session_456",
        websocket=websocket,
    ) as trace:
        try:
            async with trace.step_context(
                step_type=StepTypeEnum.retrieval,
                step_name="会失败的步骤",
                input={},
            ) as step:
                # 抛出异常
                raise ValueError("测试异常")
        except ValueError:
            pass  # 捕获异常

    # 验证步骤失败
    assert len(trace.steps) == 1
    assert trace.steps[0]["status"] == "failed"
    assert trace.failed_steps == 1
    assert "error_message" in trace.steps[0]

    # 验证错误事件已发送
    failed_events = [e for e in websocket.events if e["event_type"] == "on_step_failed"]
    assert len(failed_events) == 1

    print("✅ 错误处理测试通过！")
    print()


async def test_stream_update():
    """测试流式更新"""

    print("测试4: 流式更新")
    print("-" * 60)

    websocket = MockWebSocket()

    async with TraceManager(
        user_id="user_123",
        session_id="session_456",
        websocket=websocket,
    ) as trace:
        async with trace.step_context(
            step_type=StepTypeEnum.llm_call,
            step_name="LLM生成",
            input={},
        ) as step:
            # 模拟流式更新
            for i in range(5):
                await step.stream_update(
                    update_type="token_delta",
                    update_data={"token": f"字{i}", "index": i},
                )

            step.set_output({"text": "完整文本"})

    # 验证流式更新事件已发送
    update_events = [e for e in websocket.events if e["event_type"] == "on_step_update"]
    assert len(update_events) == 5

    print("✅ 流式更新测试通过！")
    print()


async def test_database_persistence():
    """测试数据库持久化"""

    print("测试5: 数据库持久化")
    print("-" * 60)

    websocket = MockWebSocket()

    async with TraceManager(
        user_id="user_123",
        session_id="session_456",
        message_id=f"msg_{uuid4()}",  # 设置 message_id
        websocket=websocket,
        persist_policy="immediate_async",  # 异步持久化
    ) as trace:
        async with trace.step_context(
            step_type=StepTypeEnum.retrieval,
            step_name="测试步骤",
            input={},
        ) as step:
            step.create_artifact(
                artifact_type=ArtifactTypeEnum.text,
                artifact_data={"text": "测试"},
            )
            step.set_output({"result": "success"})

    # 等待异步持久化完成
    await asyncio.sleep(1)

    # 验证数据已持久化
    from ext.ext_tortoise.models.knowledge_base import ChatMessageArtifact

    artifact_record = await ChatMessageArtifact.get_or_none(uid=trace.trace_id)

    assert artifact_record is not None
    assert len(artifact_record.steps) == 1
    assert len(artifact_record.artifacts) == 1
    assert artifact_record.total_steps == 1
    assert artifact_record.completed_steps == 1

    print("✅ 数据库持久化测试通过！")
    print()


async def run_all_tests():
    """运行所有测试"""

    print("=" * 60)
    print("🧪 开始运行集成测试")
    print("=" * 60)
    print()

    try:
        await test_trace_lifecycle()
        await test_nested_steps()
        await test_error_handling()
        await test_stream_update()
        await test_database_persistence()

        print("=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)

    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"❌ 测试失败: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    asyncio.run(run_all_tests())
