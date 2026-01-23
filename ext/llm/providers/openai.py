"""
OpenAI LLM Provider

使用官方 OpenAI SDK 实现
"""

from typing import AsyncIterator, Optional, Any
from loguru import logger

import openai
from openai import AsyncOpenAI

from ext.llm.base import BaseLLMModel
from ext.llm.types import (
    OpenAIExtraConfig,
    LLMRequest,
    LLMResponse,
    StreamChunk,
    ChatMessage,
    TokenUsage,
    ToolCall,
)
from util.general import truncate_content


class OpenAILLMModel(BaseLLMModel[OpenAIExtraConfig]):
    """
    OpenAI LLM Provider

    使用官方 SDK，支持所有 OpenAI 功能：
    - Chat Completions
    - Completions（传统模式）
    - Streaming
    - Function Calling
    - Vision (GPT-4 Vision)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.debug(
            f"Initializing OpenAI client - base_url: {self.base_url}, timeout: {self.timeout}, max_retries: {self.max_retries}"
        )
        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def _convert_messages(self, messages: list[ChatMessage]) -> list[dict]:
        """
        转换消息格式

        Args:
            messages: ChatMessage 列表

        Returns:
            OpenAI 格式的消息列表
        """
        converted = []
        for msg in messages:
            content = msg.content

            # 处理多模态内容
            if isinstance(content, list):
                # content 已经是多模态格式
                content_dict = {"type": "text", "text": content[0].get("text", "") if content else ""}
                if len(content) > 1 and content[1].get("type") == "image_url":
                    content_dict = content
                converted_msg = {"role": msg.role, "content": content}
            else:
                # 纯文本
                converted_msg = {"role": msg.role, "content": content}

            if msg.name:
                converted_msg["name"] = msg.name
            if msg.tool_call_id:
                converted_msg["tool_call_id"] = msg.tool_call_id

            converted.append(converted_msg)

        return converted

    def _convert_tools(self, tools: Optional[list]) -> Optional[list]:
        """
        转换工具格式

        Args:
            tools: 工具定义列表

        Returns:
            OpenAI 格式的工具列表
        """
        if not tools:
            return None

        converted = []
        for tool in tools:
            converted.append(
                {
                    "type": tool.type,
                    "function": {
                        "name": tool.function.name,
                        "description": tool.function.description,
                        "parameters": tool.function.parameters,
                    },
                }
            )

        return converted

    def _convert_tool_choice(self, tool_choice: Optional[Any]) -> Any:
        """
        转换工具选择格式

        Args:
            tool_choice: 工具选择策略

        Returns:
            OpenAI 格式的 tool_choice
        """
        if not tool_choice:
            return None
        if isinstance(tool_choice, str):
            return tool_choice
        if isinstance(tool_choice, dict):
            return tool_choice
        return None

    async def chat(self, request: LLMRequest) -> LLMResponse:
        """
        发起对话请求（非流式）

        Args:
            request: LLM 请求

        Returns:
            LLM 响应
        """
        logger.debug(
            f"OpenAI chat request - model: {request.model or self.model_name}, "
            f"messages: {len(request.messages)}, "
            f"temperature: {request.temperature or self.default_temperature}, "
            f"max_tokens: {request.max_tokens or self.max_tokens}, "
            f"tools: {len(request.tools) if request.tools else 0}"
        )

        try:
            kwargs = {
                "model": request.model or self.model_name,
                "messages": self._convert_messages(request.messages),
                "temperature": request.temperature or self.default_temperature,
                "max_tokens": request.max_tokens or self.max_tokens,
                "top_p": request.top_p or self.default_top_p,
                "stream": False,
            }

            if request.frequency_penalty is not None:
                kwargs["frequency_penalty"] = request.frequency_penalty
            if request.presence_penalty is not None:
                kwargs["presence_penalty"] = request.presence_penalty
            if request.stop is not None:
                kwargs["stop"] = request.stop
            if self._convert_tools(request.tools) is not None:
                kwargs["tools"] = self._convert_tools(request.tools)
            if self._convert_tool_choice(request.tool_choice) is not None:
                kwargs["tool_choice"] = self._convert_tool_choice(request.tool_choice)
            if request.response_format is not None:
                kwargs["response_format"] = request.response_format

            response = await self._client.chat.completions.create(**kwargs)

            parsed_response = self._parse_response(response)

            logger.debug(
                f"OpenAI chat response - content: {truncate_content(parsed_response.content)}, "
                f"tokens: {parsed_response.usage.total_tokens}, finish_reason: {parsed_response.finish_reason}"
            )

            return parsed_response

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in chat: {e}")
            raise RuntimeError(f"Unexpected error: {str(e)}")

    def _parse_response(self, response) -> LLMResponse:
        """
        解析 OpenAI 响应

        Args:
            response: OpenAI SDK 响应对象

        Returns:
            统一的 LLMResponse
        """
        if not response.choices:
            logger.error(f"Response has no choices: {response}")
            raise RuntimeError(f"API response has no choices. Response: {response}")

        choice = response.choices[0]
        message = choice.message

        # 提取内容
        content = message.content or ""

        # 提取工具调用
        tool_calls = None
        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                tool_call = ToolCall(
                    id=tc.id,
                    type=tc.type,
                    function={
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                )
                tool_calls.append(tool_call)

        # 提取使用统计
        usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
        )

        return LLMResponse(
            content=content,
            role=message.role,
            usage=usage,
            finish_reason=choice.finish_reason,
            tool_calls=tool_calls,
            model=response.model,
        )

    async def chat_stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:  # type: ignore
        """
        发起对话请求（流式）

        Args:
            request: LLM 请求

        Yields:
            流式响应块
        """
        logger.debug(
            f"OpenAI chat stream request - model: {request.model or self.model_name}, messages: {len(request.messages)}"
        )

        try:
            kwargs = {
                "model": request.model or self.model_name,
                "messages": self._convert_messages(request.messages),
                "temperature": request.temperature or self.default_temperature,
                "max_tokens": request.max_tokens or self.max_tokens,
                "top_p": request.top_p or self.default_top_p,
                "stream": True,
            }

            if request.frequency_penalty is not None:
                kwargs["frequency_penalty"] = request.frequency_penalty
            if request.presence_penalty is not None:
                kwargs["presence_penalty"] = request.presence_penalty
            if request.stop is not None:
                kwargs["stop"] = request.stop
            if self._convert_tools(request.tools) is not None:
                kwargs["tools"] = self._convert_tools(request.tools)
            if self._convert_tool_choice(request.tool_choice) is not None:
                kwargs["tool_choice"] = self._convert_tool_choice(request.tool_choice)
            if request.response_format is not None:
                kwargs["response_format"] = request.response_format

            stream = await self._client.chat.completions.create(**kwargs)

            chunk_count = 0
            async for chunk in stream:
                chunk_count += 1
                yield self._parse_stream_chunk(chunk)

            logger.debug(f"OpenAI stream completed - total chunks: {chunk_count}")

        except openai.APIError as e:
            logger.error(f"OpenAI API error in stream: {e}")
            raise RuntimeError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in chat_stream: {e}")
            raise RuntimeError(f"Unexpected error: {str(e)}")

    def _parse_stream_chunk(self, chunk) -> StreamChunk:
        """
        解析流式响应块

        Args:
            chunk: OpenAI SDK 流式块

        Returns:
            统一的 StreamChunk
        """
        delta = {}

        if chunk.choices:
            choice = chunk.choices[0]
            delta["index"] = choice.index

            if choice.delta.content:
                delta["content"] = choice.delta.content

            if choice.delta.role:
                delta["role"] = choice.delta.role

            if choice.delta.tool_calls:
                delta["tool_calls"] = []
                for tc in choice.delta.tool_calls:
                    tc_dict = {
                        "index": tc.index,
                        "id": tc.id,
                        "type": tc.type,
                    }
                    if tc.function:
                        tc_dict["function"] = {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    delta["tool_calls"].append(tc_dict)

        # 使用统计和结束原因只在最后一块
        usage = None
        finish_reason = None

        if chunk.usage:
            usage = TokenUsage(
                prompt_tokens=chunk.usage.prompt_tokens,
                completion_tokens=chunk.usage.completion_tokens,
                total_tokens=chunk.usage.total_tokens,
            )

        if chunk.choices and chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason

        return StreamChunk(
            delta=delta,
            usage=usage,
            finish_reason=finish_reason,
            index=chunk.choices[0].index if chunk.choices else 0,
        )
