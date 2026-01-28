"""
Anthropic LLM Provider

使用官方 Anthropic SDK 实现
"""

import orjson
from typing import Any
from collections.abc import AsyncIterator
from loguru import logger

import anthropic
from anthropic import AsyncAnthropic

from ext.llm.base import BaseLLMModel
from ext.llm.types import (
    AnthropicExtraConfig,
    LLMRequest,
    LLMResponse,
    StreamChunk,
    ChatMessage,
    TokenUsage,
    ToolCall,
)
from util.general import truncate_content


class AnthropicLLMModel(BaseLLMModel[AnthropicExtraConfig]):
    """
    Anthropic Claude LLM Provider

    使用官方 SDK，支持：
    - Messages API（对话）
    - Streaming
    - Tool Use（函数调用）
    - Vision
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.debug(
            f"Initializing Anthropic client - base_url: {self.base_url}, timeout: {self.timeout}, max_retries: {self.max_retries}",
        )
        self._client = AsyncAnthropic(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def _convert_messages(self, messages: list[ChatMessage]) -> tuple[list[dict], str | None]:
        """
        转换消息格式

        Anthropic 的消息格式与 OpenAI 不同：
        - system 消息独立在 system 参数中
        - content 支持结构化格式（text/image等）

        Args:
            messages: ChatMessage 列表

        Returns:
            (messages, system_prompt) 元组
        """
        converted_messages = []
        system_prompt = None

        for msg in messages:
            # 处理 system 消息
            if msg.role == "system":
                if isinstance(msg.content, str):
                    system_prompt = msg.content
                else:
                    # 多模态 system 消息，提取文本
                    system_prompt = msg.content[0].get("text", "") if isinstance(msg.content, list) else ""
                continue

            # 转换其他消息
            converted_msg = {"role": msg.role}

            # 处理多模态内容
            if isinstance(msg.content, list):
                # Anthropic 多模态格式
                content_blocks = []
                for item in msg.content:
                    if item.get("type") == "text":
                        content_blocks.append({"type": "text", "text": item.get("text", "")})
                    elif item.get("type") == "image_url":
                        content_blocks.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": item.get("image_url", {}).get("url", "").split(",")[1]
                                    if "," in item.get("image_url", {}).get("url", "")
                                    else "",
                                },
                            },
                        )
                converted_msg["content"] = content_blocks  # type: ignore
            else:
                # 纯文本
                converted_msg["content"] = msg.content

            converted_messages.append(converted_msg)

        return converted_messages, system_prompt

    def _convert_tools(self, tools) -> list[dict] | None:
        """
        转换工具格式

        Anthropic 的工具格式略有不同

        Args:
            tools: 工具定义列表

        Returns:
            Anthropic 格式的工具列表
        """
        if not tools:
            return None

        converted = []
        for tool in tools:
            converted.append(
                {
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "input_schema": tool.function.parameters or {"type": "object", "properties": {}},
                },
            )

        return converted

    def _convert_tool_choice(self, tool_choice: Any) -> Any:
        """
        转换工具选择格式

        Args:
            tool_choice: 工具选择策略

        Returns:
            Anthropic 格式的 tool_choice
        """
        if not tool_choice:
            return None
        if isinstance(tool_choice, str):
            # auto/any/none
            return tool_choice
        if isinstance(tool_choice, dict):
            # {"type": "tool", "name": "my_tool"}
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
            f"Anthropic chat request - model: {request.model or self.model_name}, "
            f"messages: {len(request.messages)}, "
            f"temperature: {request.temperature or self.default_temperature}, "
            f"max_tokens: {request.max_tokens or self.max_tokens}, "
            f"tools: {len(request.tools) if request.tools else 0}",
        )

        try:
            messages, system_prompt = self._convert_messages(request.messages)

            kwargs = {
                "model": request.model or self.model_name,
                "messages": messages,
                "max_tokens": request.max_tokens or self.max_tokens,
                "temperature": request.temperature or self.default_temperature,
                "top_p": request.top_p or self.default_top_p,
            }

            if system_prompt:
                kwargs["system"] = system_prompt

            if request.stop:
                kwargs["stop_sequences"] = request.stop if isinstance(request.stop, list) else [request.stop]

            if request.tools:
                kwargs["tools"] = self._convert_tools(request.tools)

            if request.tool_choice:
                kwargs["tool_choice"] = self._convert_tool_choice(request.tool_choice)

            response = await self._client.messages.create(**kwargs)

            parsed_response = self._parse_response(response)

            logger.debug(
                f"Anthropic chat response - content: {truncate_content(parsed_response.content)}, "
                f"tokens: {parsed_response.usage.total_tokens}, finish_reason: {parsed_response.finish_reason}",
            )

            return parsed_response

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            raise RuntimeError(f"Anthropic API error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in chat: {e}")
            raise RuntimeError(f"Unexpected error: {str(e)}")

    def _parse_response(self, response) -> LLMResponse:
        """
        解析 Anthropic 响应

        Args:
            response: Anthropic SDK 响应对象

        Returns:
            统一的 LLMResponse
        """
        # 提取内容
        content = ""
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                # Tool use block - 暂时不处理
                pass

        # 提取工具调用
        tool_calls = None
        for block in response.content:
            if block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_call = ToolCall(
                    id=block.id,
                    type="function",
                    function={"name": block.name, "arguments": orjson.dumps(block.input)},
                )
                tool_calls.append(tool_call)

        # 提取使用统计
        usage = TokenUsage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

        return LLMResponse(
            content=content,
            role="assistant",
            usage=usage,
            finish_reason=response.stop_reason or "stop",
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
            f"Anthropic chat stream request - model: {request.model or self.model_name}, "
            f"messages: {len(request.messages)}",
        )

        try:
            messages, system_prompt = self._convert_messages(request.messages)

            kwargs = {
                "model": request.model or self.model_name,
                "messages": messages,
                "max_tokens": request.max_tokens or self.max_tokens,
                "temperature": request.temperature or self.default_temperature,
                "top_p": request.top_p or self.default_top_p,
            }

            if system_prompt:
                kwargs["system"] = system_prompt

            if request.stop:
                kwargs["stop_sequences"] = request.stop if isinstance(request.stop, list) else [request.stop]

            if request.tools:
                kwargs["tools"] = self._convert_tools(request.tools)

            if request.tool_choice:
                kwargs["tool_choice"] = self._convert_tool_choice(request.tool_choice)

            stream = await self._client.messages.create(**kwargs, stream=True)

            current_delta = {}
            final_usage = None
            final_stop_reason = None
            chunk_count = 0

            async for event in stream:
                if event.type == "content_block_start":
                    # 新的内容块开始
                    if event.content_block.type == "text":
                        current_delta["type"] = "text"

                elif event.type == "content_block_delta":
                    # 内容增量
                    if event.delta.type == "text_delta":
                        current_delta["content"] = event.delta.text
                        chunk_count += 1
                        yield StreamChunk(delta=current_delta.copy())

                elif event.type == "message_stop":
                    # 消息结束
                    final_stop_reason = "stop"

                elif event.type == "message_delta":
                    # 消息级别的增量（使用统计）
                    if hasattr(event, "usage"):
                        final_usage = TokenUsage(
                            prompt_tokens=event.usage.input_tokens,  # type: ignore
                            completion_tokens=event.usage.output_tokens,
                            total_tokens=event.usage.input_tokens + event.usage.output_tokens,  # type: ignore
                        )

            # 发送最终的 usage 和 finish_reason
            if final_usage or final_stop_reason:
                yield StreamChunk(
                    delta={"role": "assistant"},
                    usage=final_usage,
                    finish_reason=final_stop_reason,
                )

            logger.debug(f"Anthropic stream completed - total chunks: {chunk_count}")

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error in stream: {e}")
            raise RuntimeError(f"Anthropic API error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in chat_stream: {e}")
            raise RuntimeError(f"Unexpected error: {str(e)}")
