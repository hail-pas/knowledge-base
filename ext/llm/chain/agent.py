"""
Agent 实现

提供多种 Agent 类型，包括 Function Calling 和 ReAct
支持完整的流式输出
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional
from collections.abc import AsyncIterator

from loguru import logger

from ext.llm.types import ChatMessage, LLMRequest, LLMResponse

from ext.llm.chain.base import Runnable
from ext.llm.chain.exceptions import MaxIterationsError, ToolExecutionError, ToolNotFoundError
from ext.llm.chain.llm import LLM
from ext.llm.chain.memory import BaseMemory
from ext.llm.chain.tool import Tool
from util.general import truncate_content


class AgentStream:
    """Agent 流式输出事件

    包含多种事件类型，用于流式输出 Agent 的执行过程
    """

    event_type: Literal["thought", "action", "observation", "content", "error", "max_iterations_reached", "retry"]
    content: Any
    tool_call: dict[str, Any] | None = None
    tool_result: dict[str, Any] | None = None
    error: Exception | None = None

    def __init__(
        self,
        event_type: Literal["thought", "action", "observation", "content", "error", "max_iterations_reached", "retry"],
        content: Any,
        tool_call: dict[str, Any] | None = None,
        tool_result: dict[str, Any] | None = None,
        error: Exception | None = None,
    ):
        """初始化 AgentStream

        Args:
            event_type: 事件类型
            content: 事件内容
            tool_call: 工具调用信息（action 事件）
            tool_result: 工具执行结果（observation 事件）
            error: 错误信息（error 事件）
        """
        self.event_type = event_type
        self.content = content
        self.tool_call = tool_call
        self.tool_result = tool_result
        self.error = error

    def __repr__(self) -> str:
        return f"AgentStream(event_type='{self.event_type}', content={self.content[:50] if isinstance(self.content, str) else self.content})"


class Agent(Runnable[str, str], ABC):
    """Agent 抽象基类

    提供智能工具调用和对话能力
    """

    def __init__(
        self,
        llm: LLM,
        tools: list[Tool],
        memory: BaseMemory | None = None,
        max_iterations: int = 10,
        system_prompt: str | None = None,
    ):
        """初始化 Agent

        Args:
            llm: LLM 实例
            tools: 工具列表
            memory: 记忆（可选）
            max_iterations: 最大迭代次数
            system_prompt: 系统提示词（可选）
        """
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt

    @abstractmethod
    async def ainvoke(self, input: str) -> str:
        """执行 Agent

        Args:
            input: 用户输入

        Returns:
            最终答案
        """

    @abstractmethod
    async def astream(self, input: str) -> AsyncIterator[AgentStream]: # type: ignore
        """流式执行 Agent

        Args:
            input: 用户输入

        Yields:
            AgentStream 事件
        """

    def _get_tool(self, tool_name: str) -> Tool:
        """根据名称获取工具

        Args:
            tool_name: 工具名称

        Returns:
            Tool 实例

        Raises:
            ToolNotFoundError: 工具未找到
        """
        for tool in self.tools:
            if tool.name == tool_name:
                return tool

        raise ToolNotFoundError(tool_name)

    async def _execute_tool(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """执行工具

        Args:
            tool_call: 工具调用信息

        Returns:
            工具执行结果
        """
        tool_name = tool_call["name"]
        arguments_str = tool_call["arguments"]
        arguments = json.loads(arguments_str)

        try:
            tool = self._get_tool(tool_name)
            result = await tool.ainvoke(**arguments)
            return {"result": result, "success": True}
        except Exception as e:
            raise ToolExecutionError(tool_name, e)


class FunctionCallingAgent(Agent):
    """Function Calling Agent

    使用 LLM 原生的 function calling 能力
    适用于 OpenAI、Anthropic 等支持 function calling 的模型
    """

    async def ainvoke(self, input: str) -> str:
        """执行 Function Calling Agent

        Args:
            input: 用户输入

        Returns:
            最终答案
        """
        logger.debug(
            f"FunctionCallingAgent ainvoke - input length: {len(input)}, "
            f"tools: {len(self.tools)}, max_iterations: {self.max_iterations}",
        )

        # 构建消息历史
        messages = await self._build_messages(input)

        # 获取工具定义
        tool_definitions = [tool.to_definition() for tool in self.tools]

        # 循环调用 LLM 和执行工具
        for iteration in range(self.max_iterations):
            logger.debug(f"FunctionCallingAgent iteration {iteration + 1}/{self.max_iterations}")

            # 调用 LLM
            response = await self._call_llm(messages, tool_definitions)

            # 检查是否有工具调用
            if not response.tool_calls:
                # 没有工具调用，返回最终答案
                result = response.content or ""

                logger.debug(f"FunctionCallingAgent completed - output length: {len(result)}")

                # 保存到记忆
                if self.memory:
                    await self.memory.save_context(input, result)

                return result

            # 执行工具调用
            for tool_call_data in response.tool_calls:
                tool_call = {
                    "id": tool_call_data.id,
                    "name": tool_call_data.function["name"],
                    "arguments": tool_call_data.function["arguments"],
                }

                logger.info(f"FunctionCallingAgent executing tool: {tool_call['name']}")

                tool_result = await self._execute_tool(tool_call)

                logger.debug(f"Tool '{tool_call['name']}' result: {truncate_content(str(tool_result))}")

                # 添加工具响应到消息历史
                messages.append(
                    ChatMessage(
                        role="assistant",
                        content=response.content or "",
                        tool_calls=[tool_call_data], # type: ignore
                    ),
                )
                messages.append(
                    ChatMessage(
                        role="tool",
                        content=json.dumps(tool_result),
                        tool_call_id=tool_call["id"],
                    ),
                )

        # 达到最大迭代次数
        logger.warning(f"FunctionCallingAgent reached max iterations: {self.max_iterations}")
        raise MaxIterationsError(self.max_iterations)

    async def astream(self, input: str) -> AsyncIterator[AgentStream]: # type: ignore
        """流式执行 Function Calling Agent

        Args:
            input: 用户输入

        Yields:
            AgentStream 事件
        """
        logger.debug(
            f"FunctionCallingAgent astream - input length: {len(input)}, "
            f"tools: {len(self.tools)}, max_iterations: {self.max_iterations}",
        )

        # 构建消息历史
        messages = await self._build_messages(input)

        # 获取工具定义
        tool_definitions = [tool.to_definition() for tool in self.tools]

        # 循环调用 LLM 和执行工具
        for iteration in range(self.max_iterations):
            logger.debug(f"FunctionCallingAgent stream iteration {iteration + 1}/{self.max_iterations}")

            # 调用 LLM（非流式）
            response = await self._call_llm(messages, tool_definitions)

            yield AgentStream(event_type="thought", content=response.content or "")

            # 检查是否有工具调用
            if not response.tool_calls:
                # 没有工具调用，返回最终答案
                result = response.content or ""

                logger.debug(f"FunctionCallingAgent stream completed - output length: {len(result)}")

                # 保存到记忆
                if self.memory:
                    await self.memory.save_context(input, result)

                yield AgentStream(event_type="content", content=result)
                return

            # 执行工具调用
            for tool_call_data in response.tool_calls:
                tool_call = {
                    "id": tool_call_data.id,
                    "name": tool_call_data.function["name"],
                    "arguments": tool_call_data.function["arguments"],
                }

                logger.info(f"FunctionCallingAgent stream executing tool: {tool_call['name']}")

                yield AgentStream(event_type="action", content="", tool_call=tool_call)

                try:
                    tool_result = await self._execute_tool(tool_call)

                    logger.debug(f"Tool '{tool_call['name']}' result: {truncate_content(str(tool_result))}")

                    yield AgentStream(
                        event_type="observation",
                        content="",
                        tool_result=tool_result,
                    )

                    # 添加工具响应到消息历史
                    messages.append(
                        ChatMessage(
                            role="assistant",
                            content=response.content or "",
                            tool_calls=[tool_call_data], # type: ignore
                        ),
                    )
                    messages.append(
                        ChatMessage(
                            role="tool",
                            content=json.dumps(tool_result),
                            tool_call_id=tool_call["id"],
                        ),
                    )
                except ToolExecutionError as e:
                    logger.error(f"Tool execution failed: {e}")
                    yield AgentStream(
                        event_type="error",
                        content=str(e),
                        error=e,
                    )
                    raise

        # 达到最大迭代次数
        logger.warning(f"FunctionCallingAgent stream reached max iterations: {self.max_iterations}")
        yield AgentStream(
            event_type="max_iterations_reached",
            content=f"Agent reached maximum iterations ({self.max_iterations})",
        )
        raise MaxIterationsError(self.max_iterations)

    async def _build_messages(self, input: str) -> list[ChatMessage]:
        """构建消息历史

        Args:
            input: 用户输入

        Returns:
            消息列表
        """
        messages = []

        # 添加系统提示词
        if self.system_prompt:
            messages.append(ChatMessage(role="system", content=self.system_prompt))

        # 加载记忆
        if self.memory:
            memory_vars = await self.memory.load_memory_variables()
            messages.extend(memory_vars.get("messages", []))

        # 添加用户输入
        messages.append(ChatMessage(role="user", content=input))

        return messages

    async def _call_llm(
        self,
        messages: list[ChatMessage],
        tools: list[Any],
    ) -> LLMResponse:
        """调用 LLM

        Args:
            messages: 消息列表
            tools: 工具定义列表

        Returns:
            LLM 响应
        """
        # 转换消息格式
        messages_dict = [
            {
                "role": msg.role,
                "content": msg.content,
                "tool_call_id": msg.tool_call_id,
            }
            if msg.tool_call_id
            else {
                "role": msg.role,
                "content": msg.content,
            }
            for msg in messages
        ]

        # 调用 LLM
        response = await self.llm.model.chat(
            LLMRequest(
                messages=messages_dict, # type: ignore
                tools=tools,
                tool_choice="auto",
            ),
        )

        return response


class ReActAgent(Agent):
    """ReAct Agent

    采用 Reasoning + Acting 模式
    通过文本推理循环执行工具
    适用于不支持 function calling 的模型
    """

    def __init__(
        self,
        llm: LLM,
        tools: list[Tool],
        memory: BaseMemory | None = None,
        max_iterations: int = 10,
        system_prompt: str | None = None,
    ):
        """初始化 ReAct Agent

        Args:
            llm: LLM 实例
            tools: 工具列表
            memory: 记忆（可选）
            max_iterations: 最大迭代次数
            system_prompt: 系统提示词（可选）
        """
        super().__init__(llm, tools, memory, max_iterations, system_prompt)

        # 默认 ReAct 提示词模板
        self.react_template = """你是一个智能助手，可以使用以下工具来回答问题。

可用工具：
{tool_descriptions}

请按照以下格式思考：

Thought: 你的思考过程
Action: 工具名称
Action Input: 工具参数（JSON 格式）
Observation: 工具返回结果
...（重复 Thought、Action、Action Input、Observation）
Thought: 我知道最终答案了
Final Answer: 最终答案

开始！

问题：{input}
{previous_steps}
"""

    async def ainvoke(self, input: str) -> str:
        """执行 ReAct Agent

        Args:
            input: 用户输入

        Returns:
            最终答案
        """
        logger.debug(
            f"ReActAgent ainvoke - input length: {len(input)}, "
            f"tools: {len(self.tools)}, max_iterations: {self.max_iterations}",
        )

        # 构建工具描述
        tool_descriptions = "\n".join(f"- {tool.name}: {tool.description}" for tool in self.tools)

        # 初始化上下文
        context = ""
        previous_steps = ""

        # 循环推理
        for iteration in range(self.max_iterations):
            logger.debug(f"ReActAgent iteration {iteration + 1}/{self.max_iterations}")

            # 构建提示词
            prompt = self.react_template.format(
                tool_descriptions=tool_descriptions,
                input=input,
                previous_steps=previous_steps,
            )

            # 调用 LLM
            response = await self.llm.ainvoke(prompt)

            # 解析响应
            parsed = self._parse_react_response(response)

            if parsed["final_answer"]:
                # 找到最终答案
                result = parsed["final_answer"]

                logger.debug(f"ReActAgent completed - output length: {len(result)}")

                # 保存到记忆
                if self.memory:
                    await self.memory.save_context(input, result)

                return result

            if parsed["action"]:
                # 执行工具
                tool_name = parsed["action"]
                tool_input = parsed["action_input"]

                logger.info(f"ReActAgent executing tool: {tool_name}")

                try:
                    tool = self._get_tool(tool_name)
                    tool_result = await tool.ainvoke(**tool_input)

                    logger.debug(f"Tool '{tool_name}' result: {truncate_content(str(tool_result))}")

                    observation = f"Observation: {json.dumps(tool_result, ensure_ascii=False)}"
                    previous_steps += f"\nThought: {parsed['thought']}\nAction: {tool_name}\nAction Input: {json.dumps(tool_input)}\n{observation}\n"
                except Exception as e:
                    logger.error(f"ReActAgent tool execution error: {e}")
                    observation = f"Observation: Error: {str(e)}"
                    previous_steps += f"\nThought: {parsed['thought']}\nAction: {tool_name}\nAction Input: {json.dumps(tool_input)}\n{observation}\n"

        # 达到最大迭代次数
        logger.warning(f"ReActAgent reached max iterations: {self.max_iterations}")
        raise MaxIterationsError(self.max_iterations)

    async def astream(self, input: str) -> AsyncIterator[AgentStream]: # type: ignore
        """流式执行 ReAct Agent

        Args:
            input: 用户输入

        Yields:
            AgentStream 事件
        """
        logger.debug(
            f"ReActAgent astream - input length: {len(input)}, "
            f"tools: {len(self.tools)}, max_iterations: {self.max_iterations}",
        )

        # 构建工具描述
        tool_descriptions = "\n".join(f"- {tool.name}: {tool.description}" for tool in self.tools)

        # 初始化上下文
        previous_steps = ""

        # 循环推理
        for iteration in range(self.max_iterations):
            logger.debug(f"ReActAgent stream iteration {iteration + 1}/{self.max_iterations}")

            # 构建提示词
            prompt = self.react_template.format(
                tool_descriptions=tool_descriptions,
                input=input,
                previous_steps=previous_steps,
            )

            # 调用 LLM
            response = await self.llm.ainvoke(prompt)

            yield AgentStream(event_type="thought", content=response)

            # 解析响应
            parsed = self._parse_react_response(response)

            if parsed["final_answer"]:
                # 找到最终答案
                result = parsed["final_answer"]

                logger.debug(f"ReActAgent stream completed - output length: {len(result)}")

                # 保存到记忆
                if self.memory:
                    await self.memory.save_context(input, result)

                yield AgentStream(event_type="content", content=result)
                return

            if parsed["action"]:
                # 执行工具
                tool_name = parsed["action"]
                tool_input = parsed["action_input"]

                logger.info(f"ReActAgent stream executing tool: {tool_name}")

                yield AgentStream(
                    event_type="action",
                    content="",
                    tool_call={
                        "name": tool_name,
                        "arguments": json.dumps(tool_input),
                    },
                )

                try:
                    tool = self._get_tool(tool_name)
                    tool_result = await tool.ainvoke(**tool_input)

                    logger.debug(f"Tool '{tool_name}' result: {truncate_content(str(tool_result))}")

                    observation = f"Observation: {json.dumps(tool_result, ensure_ascii=False)}"
                    previous_steps += f"\nThought: {parsed['thought']}\nAction: {tool_name}\nAction Input: {json.dumps(tool_input)}\n{observation}\n"

                    yield AgentStream(
                        event_type="observation",
                        content="",
                        tool_result=tool_result,
                    )
                except Exception as e:
                    logger.error(f"ReActAgent stream tool execution error: {e}")
                    observation = f"Observation: Error: {str(e)}"
                    previous_steps += f"\nThought: {parsed['thought']}\nAction: {tool_name}\nAction Input: {json.dumps(tool_input)}\n{observation}\n"

                    yield AgentStream(
                        event_type="error",
                        content=str(e),
                        error=e,
                    )

        # 达到最大迭代次数
        logger.warning(f"ReActAgent stream reached max iterations: {self.max_iterations}")
        yield AgentStream(
            event_type="max_iterations_reached",
            content=f"Agent reached maximum iterations ({self.max_iterations})",
        )
        raise MaxIterationsError(self.max_iterations)

    def _parse_react_response(self, response: str) -> dict[str, Any]:
        """解析 ReAct 响应

        Args:
            response: LLM 响应

        Returns:
            解析后的字典
        """
        import re

        result = {
            "thought": None,
            "action": None,
            "action_input": None,
            "final_answer": None,
        }

        # 提取 Thought
        thought_match = re.search(r"Thought:\s*(.*?)(?=\n(?:Action|Final Answer)|$)", response, re.DOTALL)
        if thought_match:
            result["thought"] = thought_match.group(1).strip() # type: ignore

        # 提取 Action
        action_match = re.search(r"Action:\s*(\w+)", response)
        if action_match:
            result["action"] = action_match.group(1) # type: ignore

        # 提取 Action Input
        action_input_match = re.search(r"Action Input:\s*(\{.*?\})", response, re.DOTALL)
        if action_input_match:
            try:
                result["action_input"] = json.loads(action_input_match.group(1))
            except json.JSONDecodeError:
                pass

        # 提取 Final Answer
        final_answer_match = re.search(r"Final Answer:\s*(.*?)(?:\n|$)", response, re.DOTALL)
        if final_answer_match:
            result["final_answer"] = final_answer_match.group(1).strip() # type: ignore

        return result


__all__ = [
    "Agent",
    "AgentStream",
    "FunctionCallingAgent",
    "ReActAgent",
]
