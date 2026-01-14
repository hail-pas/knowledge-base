# """
# Token 使用量获取示例

# 展示如何在 pydantic_ai agent 的 run 和流式模式下获取 token 消耗信息。
# """

# import os
# import pytest
# from pydantic_ai import Agent

# from ext.llm import LLMModelFactory
# from ext.ext_tortoise.enums import LLMModelTypeEnum
# from ext.ext_tortoise.models.knowledge_base import LLMModelConfig


# # 获取环境变量配置
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
# OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")


# # 跳过测试的条件
# skip_if_no_api_key = pytest.mark.skipif(
#     not OPENAI_API_KEY,
#     reason="OPENAI_API_KEY not set in environment"
# )


# def create_openai_config() -> LLMModelConfig:
#     """创建 OpenAI 配置"""
#     config_dict = {"api_key": OPENAI_API_KEY}
#     if OPENAI_BASE_URL:
#         config_dict["base_url"] = OPENAI_BASE_URL

#     config = LLMModelConfig(
#         name="test-token-usage",
#         type=LLMModelTypeEnum.openai,
#         model_name=OPENAI_MODEL_NAME,
#         config=config_dict,
#         capabilities={
#             "function_calling": True,
#             "json_output": True,
#             "streaming": True,
#         },
#         max_tokens=1000,
#         is_enabled=True,
#     )
#     config._saved_in_db = True
#     config.id = 99
#     return config


# @skip_if_no_api_key
# class TestTokenUsageExamples:
#     """Token 使用量获取示例"""

#     @pytest.mark.asyncio
#     async def test_run_token_usage(self):
#         """
#         示例 1: 使用 agent.run() 获取 token 消耗

#         在 run 模式下，返回的 AgentRunResult 对象有 usage() 方法，
#         返回 Usage 对象，包含以下字段：
#         - requests: 请求数
#         - request_tokens: 输入 token 数
#         - response_tokens: 输出 token 数
#         - total_tokens: 总 token 数 (request_tokens + response_tokens)
#         - details: 额外详细信息 (dict)
#         """
#         config = create_openai_config()
#         model = await LLMModelFactory.create(config, use_cache=False)
#         pydantic_model = model.get_model_for_agent()

#         agent = Agent(pydantic_model)

#         # 运行 agent 并获取结果
#         result = await agent.run("请用一句话介绍人工智能")

#         # 获取 token 使用量
#         usage = result.usage()

#         # 打印 token 使用信息
#         print("\n=== Token 使用量 (run 模式) ===")
#         print(f"请求数: {usage.requests}")
#         print(f"输入 tokens: {usage.request_tokens}")
#         print(f"输出 tokens: {usage.response_tokens}")
#         print(f"总 tokens: {usage.total_tokens}")
#         if usage.details:
#             print(f"详细信息: {usage.details}")

#         # 验证基本信息
#         assert usage.requests == 1
#         assert usage.request_tokens is not None
#         assert usage.response_tokens is not None
#         assert usage.total_tokens is not None
#         assert usage.total_tokens == usage.request_tokens + usage.response_tokens

#         # 返回的输出内容
#         print(f"\n模型输出: {result.output}")

#     @pytest.mark.asyncio
#     async def test_stream_token_usage(self):
#         """
#         示例 2: 使用 agent.run_stream() 获取 token 消耗

#         在流式模式下，AgentStream 对象有一个 usage() 方法。
#         注意：在流式输出期间，usage() 会返回当前已消耗的 token 数。
#         要获取完整的 token 使用量，需要等待流式输出完成后调用 usage()。
#         """
#         config = create_openai_config()
#         model = await LLMModelFactory.create(config, use_cache=False)
#         pydantic_model = model.get_model_for_agent()

#         agent = Agent(pydantic_model)

#         print("\n=== Token 使用量 (流式模式) ===")

#         # 使用流式输出
#         async with agent.run_stream("请列出 Python 的三个主要特点") as result:
#             # 在流式输出过程中，可以获取当前的 token 使用量
#             # 但此时可能还没有完整的数据
#             chunks = []
#             async for chunk in result.stream_text():
#                 chunks.append(chunk)
#                 # 可以在流式过程中打印部分 usage 信息（可选）
#                 # partial_usage = result.usage()
#                 # print(f"当前已消耗 tokens: {partial_usage.total_tokens}")

#         # 流式输出完成后，获取完整的 token 使用量
#         final_usage = result.usage()

#         print(f"请求数: {final_usage.requests}")
#         print(f"输入 tokens: {final_usage.request_tokens}")
#         print(f"输出 tokens: {final_usage.response_tokens}")
#         print(f"总 tokens: {final_usage.total_tokens}")
#         if final_usage.details:
#             print(f"详细信息: {final_usage.details}")

#         print(f"\n流式输出长度: {len(chunks)} 个 chunks")
#         print(f"完整输出: {''.join(chunks)}")

#         # 验证基本信息
#         assert final_usage.requests >= 1  # 流式模式下可能会有重试，请求数可能大于 1
#         assert final_usage.request_tokens is not None
#         assert final_usage.response_tokens is not None
#         assert final_usage.total_tokens is not None
#         assert final_usage.total_tokens == final_usage.request_tokens + final_usage.response_tokens

#     @pytest.mark.asyncio
#     async def test_usage_in_multiple_runs(self):
#         """
#         示例 3: 多次运行时的 token 累计

#         演示如何在多次调用中累计 token 使用量
#         """
#         config = create_openai_config()
#         model = await LLMModelFactory.create(config, use_cache=False)
#         pydantic_model = model.get_model_for_agent()

#         agent = Agent(pydantic_model)

#         print("\n=== 多次运行的 Token 累计 ===")

#         from pydantic_ai.usage import Usage

#         total_usage = Usage()

#         prompts = [
#             "什么是 Python?",
#             "什么是 JavaScript?",
#             "什么是 Rust?",
#         ]

#         for i, prompt in enumerate(prompts, 1):
#             result = await agent.run(prompt)
#             run_usage = result.usage()

#             print(f"\n运行 {i}:")
#             print(f"  输入 tokens: {run_usage.request_tokens}")
#             print(f"  输出 tokens: {run_usage.response_tokens}")
#             print(f"  总 tokens: {run_usage.total_tokens}")

#             # 累计使用量
#             total_usage.incr(run_usage)

#         print(f"\n累计统计:")
#         print(f"总请求数: {total_usage.requests}")
#         print(f"总输入 tokens: {total_usage.request_tokens}")
#         print(f"总输出 tokens: {total_usage.response_tokens}")
#         print(f"总 tokens: {total_usage.total_tokens}")

#         # 验证累计值
#         assert total_usage.requests == 3
#         assert total_usage.total_tokens == total_usage.request_tokens + total_usage.response_tokens

#     @pytest.mark.asyncio
#     async def test_usage_with_structured_output(self):
#         """
#         示例 4: 结构化输出时的 token 消耗

#         展示在使用结构化输出时如何获取 token 使用量
#         """
#         from pydantic import BaseModel

#         config = create_openai_config()
#         model = await LLMModelFactory.create(config, use_cache=False)
#         pydantic_model = model.get_model_for_agent()

#         class Summary(BaseModel):
#             """摘要结构"""
#             title: str
#             description: str
#             key_points: list[str]

#         agent = Agent(pydantic_model, output_type=Summary)

#         print("\n=== 结构化输出的 Token 使用量 ===")

#         result = await agent.run("总结：Python 是一种高级编程语言，强调代码可读性，支持多种编程范式")
#         usage = result.usage()

#         print(f"输入 tokens: {usage.request_tokens}")
#         print(f"输出 tokens: {usage.response_tokens}")
#         print(f"总 tokens: {usage.total_tokens}")

#         summary = result.output
#         print(f"\n结构化输出:")
#         print(f"  标题: {summary.title}")
#         print(f"  描述: {summary.description}")
#         print(f"  关键点: {summary.key_points}")

#         # 验证
#         assert isinstance(summary, Summary)
#         assert usage.total_tokens is not None
