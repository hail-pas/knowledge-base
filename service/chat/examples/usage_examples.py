"""
使用示例：简单的 RAG 聊天

演示如何使用 TraceManager 和 StepContext 实现 RAG 聊天
"""

import asyncio
from fastapi import WebSocket

from service.chat.runtime import TraceManager
from service.chat.enums import StepTypeEnum, ArtifactTypeEnum


async def simple_retrieval_handler(
    user_id: str,
    session_id: str,
    query: str,
    websocket: WebSocket,
):
    """
    简单的 RAG 聊天处理器

    演示：
    1. 创建 Trace
    2. 创建检索步骤
    3. 创建产物
    4. 自动发送事件
    """

    # 使用 TraceManager
    async with TraceManager(
        user_id=user_id,
        session_id=session_id,
        chat_mode="rag",
        llm_model="gpt-4",
        websocket=websocket,
    ) as trace:
        # 创建检索步骤
        async with trace.step_context(
            step_type=StepTypeEnum.retrieval,
            step_name="知识库检索",
            input={"query": query, "top_k": 10},
        ) as step:
            # 模拟检索逻辑
            print(f"🔍 正在检索: {query}")

            # 模拟检索结果
            results = [
                {
                    "chunk_id": 123,
                    "content": "机器学习是人工智能的一个分支...",
                    "score": 0.95,
                    "metadata": {"source": "doc_1.pdf", "page": 10},
                },
                {
                    "chunk_id": 124,
                    "content": "深度学习是机器学习的一个子领域...",
                    "score": 0.92,
                    "metadata": {"source": "doc_2.pdf", "page": 5},
                },
            ]

            print(f"✅ 检索到 {len(results)} 个结果")

            # 创建产物
            step.create_artifact(
                artifact_type=ArtifactTypeEnum.retrieval_results,
                artifact_data={
                    "chunks": results,
                    "total_count": len(results),
                    "query": query,
                },
            )

            # 设置输出
            step.set_output(
                {
                    "results_count": len(results),
                    "query": query,
                }
            )

        # Trace 会自动：
        # 1. 发送 on_trace_start 事件
        # 2. 发送 on_step_start 事件
        # 3. 发送 on_artifact_created 事件
        # 4. 发送 on_step_complete 事件
        # 5. 发送 on_trace_complete 事件
        # 6. 持久化到数据库

    print("🎉 RAG 聊天完成！")


# 嵌套步骤示例
async def nested_retrieval_handler(
    user_id: str,
    session_id: str,
    query: str,
    websocket: WebSocket,
):
    """
    嵌套步骤示例：稠密检索 + 重排序

    演示：
    1. 父步骤：RAG检索
    2. 子步骤1：稠密向量检索
    3. 子步骤2：重排序
    """

    async with TraceManager(
        user_id=user_id,
        session_id=session_id,
        chat_mode="rag",
        websocket=websocket,
    ) as trace:
        # 父步骤：RAG检索
        async with trace.step_context(
            step_type=StepTypeEnum.retrieval,
            step_name="RAG检索",
            input={"query": query, "top_k": 10},
        ) as parent_step:
            # 子步骤1：稠密检索
            async with trace.step_context(
                step_type=StepTypeEnum.retrieval,
                step_name="稠密向量检索",
                input={"query": query, "method": "dense", "top_k": 50},
                parent_step_id=parent_step.step_id,  # 指定父步骤
            ) as dense_step:
                # 模拟稠密检索
                print("🔍 稠密向量检索中...")
                dense_results = [{"chunk_id": i, "content": f"内容{i}", "score": 0.9} for i in range(50)]

                dense_step.create_artifact(
                    artifact_type=ArtifactTypeEnum.retrieval_results,
                    artifact_data={"chunks": dense_results, "total_count": 50},
                )
                dense_step.set_output({"results_count": 50})

            # 子步骤2：重排序
            async with trace.step_context(
                step_type=StepTypeEnum.retrieval,
                step_name="结果重排序",
                input={"query": query, "candidates": dense_results},
                parent_step_id=parent_step.step_id,  # 指定父步骤
            ) as rerank_step:
                # 模拟重排序
                print("🔄 重排序中...")
                reranked_results = dense_results[:10]  # 取前10个

                rerank_step.create_artifact(
                    artifact_type=ArtifactTypeEnum.retrieval_results,
                    artifact_data={"chunks": reranked_results, "total_count": 10},
                )
                rerank_step.set_output({"results_count": 10})

            # 父步骤输出
            parent_step.set_output({"final_results_count": 10})

    print("🎉 嵌套 RAG 检索完成！")


# LLM 流式输出示例
async def llm_streaming_handler(
    user_id: str,
    session_id: str,
    messages: list,
    websocket: WebSocket,
):
    """
    LLM 流式输出示例

    演示：
    1. 使用 stream_update() 流式发送 token
    2. 实时向前端推送生成的内容
    """

    async with TraceManager(
        user_id=user_id,
        session_id=session_id,
        chat_mode="normal",
        websocket=websocket,
    ) as trace:
        async with trace.step_context(
            step_type=StepTypeEnum.llm_call,
            step_name="LLM生成",
            input={"model": "gpt-4", "messages": messages},
        ) as step:
            # 模拟 LLM 流式生成
            print("🤖 LLM 生成中...")

            full_text = ""
            tokens = ["机", "器", "学", "习", "是", "人", "工", "智", "能", "的", "一", "个", "分", "支", "。"]

            for i, token in enumerate(tokens):
                await asyncio.sleep(0.1)  # 模拟网络延迟
                full_text += token

                # 流式发送 token
                await step.stream_update(
                    update_type="token_delta",
                    update_data={"token": token, "index": i},
                )

                print(f"📤 发送 token: {token}")

            # 创建最终产物
            step.create_artifact(
                artifact_type=ArtifactTypeEnum.llm_output,
                artifact_data={
                    "text": full_text,
                    "finish_reason": "stop",
                    "model": "gpt-4",
                },
            )

            # 创建使用统计产物
            step.create_artifact(
                artifact_type=ArtifactTypeEnum.usage_stats,
                artifact_data={
                    "input_tokens": 100,
                    "output_tokens": len(tokens),
                    "total_tokens": 100 + len(tokens),
                    "model": "gpt-4",
                },
            )

            step.set_output({"text": full_text})

    print("🎉 LLM 流式生成完成！")


# 进度更新示例
async def progress_update_handler(
    user_id: str,
    session_id: str,
    query: str,
    websocket: WebSocket,
):
    """
    进度更新示例

    演示：
    1. 使用 update_progress() 更新步骤进度
    2. 前端可以显示进度条
    """

    async with TraceManager(
        user_id=user_id,
        session_id=session_id,
        chat_mode="rag",
        websocket=websocket,
    ) as trace:
        async with trace.step_context(
            step_type=StepTypeEnum.retrieval,
            step_name="知识库检索",
            input={"query": query},
        ) as step:
            # 模拟长时间运行的任务
            total_steps = 10

            for i in range(total_steps):
                await asyncio.sleep(0.5)  # 模拟处理

                # 更新进度
                progress = (i + 1) / total_steps * 100
                await step.update_progress(
                    progress_percentage=progress,
                    current_status=f"正在处理第 {i + 1}/{total_steps} 步",
                )

                print(f"📊 进度: {progress:.1f}%")

            # 完成后设置输出
            step.set_output({"status": "completed", "total_steps": total_steps})

    print("🎉 任务完成！")


# 运行所有示例
async def main():
    """运行所有示例（不需要 WebSocket）"""

    print("=" * 60)
    print("🚀 运行示例")
    print("=" * 60)
    print()

    # 示例1：简单检索
    print("示例1：简单检索")
    print("-" * 60)
    await simple_retrieval_handler(
        user_id="user_123",
        session_id="session_456",
        query="什么是机器学习？",
        websocket=None,  # 不使用 WebSocket
    )
    print()

    # 示例2：嵌套步骤
    print("示例2：嵌套步骤")
    print("-" * 60)
    await nested_retrieval_handler(
        user_id="user_123",
        session_id="session_456",
        query="什么是机器学习？",
        websocket=None,
    )
    print()

    # 示例3：LLM 流式输出
    print("示例3：LLM 流式输出")
    print("-" * 60)
    await llm_streaming_handler(
        user_id="user_123",
        session_id="session_456",
        messages=[{"role": "user", "content": "什么是机器学习？"}],
        websocket=None,
    )
    print()

    # 示例4：进度更新
    print("示例4：进度更新")
    print("-" * 60)
    await progress_update_handler(
        user_id="user_123",
        session_id="session_456",
        query="什么是机器学习？",
        websocket=None,
    )
    print()

    print("=" * 60)
    print("✅ 所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
