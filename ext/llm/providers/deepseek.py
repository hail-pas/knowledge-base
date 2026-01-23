"""
DeepSeek LLM Provider

使用 httpx 直接调用 API（完全 OpenAI 兼容）
"""

from typing import AsyncIterator, Any
import json
import httpx
from loguru import logger

from ext.llm.base import BaseLLMModel
from ext.llm.types import (
    DeepSeekExtraConfig,
    LLMRequest,
    LLMResponse,
    StreamChunk,
    ChatMessage,
    TokenUsage,
    ToolCall,
)


class DeepSeekLLMModel(BaseLLMModel[DeepSeekExtraConfig]):
    """
    DeepSeek LLM Provider

    完全 OpenAI 兼容，使用 httpx 直接调用
    """

    def _convert_messages(self, messages: list[ChatMessage]) -> list[dict]:
        """
        转换消息格式

        Args:
            messages: ChatMessage 列表

        Returns:
            DeepSeek 格式的消息列表（与 OpenAI 相同）
        """
        converted = []
        for msg in messages:
            content = msg.content

            # 处理多模态内容
            if isinstance(content, list):
                converted_msg = {"role": msg.role, "content": content}
            else:
                converted_msg = {"role": msg.role, "content": content}

            if msg.name:
                converted_msg["name"] = msg.name
            if msg.tool_call_id:
                converted_msg["tool_call_id"] = msg.tool_call_id

            converted.append(converted_msg)

        return converted

    def _convert_tools(self, tools) -> Any:
        """
        转换工具格式（与 OpenAI 相同）
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

    def _convert_tool_choice(self, tool_choice: Any) -> Any:
        """
        转换工具选择格式（与 OpenAI 相同）
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
        client = self.get_httpx_client()
        url = self.build_endpoint_url()
        headers = self.build_request_headers()
        body = self._build_request_body(request)

        for attempt in range(self.max_retries + 1):
            try:
                response = await client.post(url, json=body, headers=headers, timeout=self.timeout)

                if response.status_code == 200:
                    response_data = response.json()
                    return self._parse_response(response_data)
                else:
                    error_data = response.json()
                    error_message = self._extract_error_message(error_data) or "Unknown error"

                    if self.should_retry(response.status_code, attempt):
                        delay = self.get_retry_delay(attempt)
                        logger.warning(
                            f"DeepSeek 请求失败（{response.status_code}: {error_message}），"
                            f"{delay}秒后重试（{attempt + 1}/{self.max_retries}）"
                        )
                        import asyncio

                        await asyncio.sleep(delay)
                        continue
                    else:
                        raise RuntimeError(f"DeepSeek API error: {error_message} (状态码: {response.status_code})")

            except httpx.TimeoutException as e:
                if attempt == self.max_retries:
                    raise RuntimeError(f"DeepSeek 请求超时: {str(e)}")
                delay = self.get_retry_delay(attempt)
                logger.warning(f"DeepSeek 请求超时: {str(e)}，{delay}秒后重试")
                import asyncio

                await asyncio.sleep(delay)

            except httpx.ConnectError as e:
                if attempt == self.max_retries:
                    raise RuntimeError(f"DeepSeek 连接错误: {str(e)}")
                delay = self.get_retry_delay(attempt)
                logger.warning(f"DeepSeek 连接错误: {str(e)}，{delay}秒后重试")
                import asyncio

                await asyncio.sleep(delay)

        raise RuntimeError(f"DeepSeek API 请求失败，已达最大重试次数")

    def _build_request_body(self, request: LLMRequest) -> dict:
        """
        构建请求体

        Args:
            request: LLM 请求

        Returns:
            DeepSeek API 请求体
        """
        body = {
            "model": request.model or self.model_name,
            "messages": self._convert_messages(request.messages),
            "temperature": request.temperature or self.default_temperature,
            "max_tokens": request.max_tokens or self.max_tokens,
            "top_p": request.top_p or self.default_top_p,
        }

        # 可选参数
        if request.frequency_penalty is not None:
            body["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            body["presence_penalty"] = request.presence_penalty
        if request.stop is not None:
            body["stop"] = request.stop

        # 工具调用
        if request.tools:
            body["tools"] = self._convert_tools(request.tools)
        if request.tool_choice:
            body["tool_choice"] = self._convert_tool_choice(request.tool_choice)

        # 响应格式
        if request.response_format:
            body["response_format"] = request.response_format

        return body

    def _parse_response(self, response_data: dict) -> LLMResponse:
        """
        解析响应

        Args:
            response_data: DeepSeek API 响应

        Returns:
            统一的 LLMResponse
        """
        choice = response_data["choices"][0]
        message = choice["message"]

        # 提取内容
        content = message.get("content", "")

        # 提取工具调用
        tool_calls = None
        if "tool_calls" in message:
            tool_calls = []
            for tc in message["tool_calls"]:
                tool_call = ToolCall(
                    id=tc["id"],
                    type=tc["type"],
                    function={
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    },
                )
                tool_calls.append(tool_call)

        # 提取使用统计
        usage_data = response_data.get("usage", {})
        usage = TokenUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return LLMResponse(
            content=content,
            role=message.get("role", "assistant"),
            usage=usage,
            finish_reason=choice.get("finish_reason", "stop"),
            tool_calls=tool_calls,
            model=response_data.get("model"),
        )

    def _extract_error_message(self, response_data: dict) -> str:
        """
        从响应中提取错误信息

        Args:
            response_data: 响应数据

        Returns:
            错误信息
        """
        if "error" in response_data:
            error = response_data["error"]
            if isinstance(error, dict):
                return error.get("message", "")
            return str(error)
        return ""

    async def chat_stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]: # type: ignore
        """
        发起对话请求（流式）

        Args:
            request: LLM 请求

        Yields:
            流式响应块
        """
        client = self.get_httpx_client()
        url = self.build_endpoint_url()
        headers = self.build_request_headers()
        body = self._build_request_body(request)
        body["stream"] = True

        try:
            async with client.stream("POST", url, json=body, headers=headers, timeout=self.timeout) as response:
                if response.status_code != 200:
                    error_data = response.json()
                    error_message = self._extract_error_message(error_data) or "Unknown error"
                    raise RuntimeError(f"DeepSeek API error: {error_message} (状态码: {response.status_code})")

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix

                        if data == "[DONE]":
                            break

                        try:
                            chunk_data = json.loads(data)
                            yield self._parse_stream_chunk(chunk_data)
                        except json.JSONDecodeError:
                            continue

        except httpx.TimeoutException as e:
            raise RuntimeError(f"DeepSeek 流式请求超时: {str(e)}")
        except httpx.ConnectError as e:
            raise RuntimeError(f"DeepSeek 流式请求连接错误: {str(e)}")
        except Exception as e:
            logger.error(f"DeepSeek 流式请求错误: {e}")
            raise RuntimeError(f"DeepSeek 流式请求错误: {str(e)}")

    def _parse_stream_chunk(self, chunk_data: dict) -> StreamChunk:
        """
        解析流式响应块

        Args:
            chunk_data: 流式响应块数据

        Returns:
            统一的 StreamChunk
        """
        delta = {}

        if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
            choice = chunk_data["choices"][0]
            delta["index"] = choice.get("index", 0)

            if "delta" in choice:
                chunk_delta = choice["delta"]

                if "content" in chunk_delta:
                    delta["content"] = chunk_delta["content"]
                if "role" in chunk_delta:
                    delta["role"] = chunk_delta["role"]
                if "tool_calls" in chunk_delta:
                    delta["tool_calls"] = chunk_delta["tool_calls"]

        # 使用统计和结束原因只在最后一块
        usage = None
        finish_reason = None

        if "usage" in chunk_data:
            usage_data = chunk_data["usage"]
            usage = TokenUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )

        if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
            finish_reason = chunk_data["choices"][0].get("finish_reason")

        return StreamChunk(
            delta=delta,
            usage=usage,
            finish_reason=finish_reason,
            index=delta.get("index", 0),
        )
