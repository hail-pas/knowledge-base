"""
Azure OpenAI LLM Provider

使用 OpenAI SDK 的 Azure 扩展实现
"""

from typing import AsyncIterator, Any
from loguru import logger

from openai import AsyncAzureOpenAI

from ext.llm.base import BaseLLMModel
from ext.llm.types import (
    AzureOpenAIExtraConfig,
    LLMRequest,
    LLMResponse,
    StreamChunk,
)
from util.general import truncate_content


class AzureOpenAILLMModel(BaseLLMModel[AzureOpenAIExtraConfig]):
    """
    Azure OpenAI LLM Provider

    继承 OpenAI 的实现，使用 Azure 专用的 SDK
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Azure OpenAI 需要额外的参数
        deployment_name = self.extra_config.deployment_name or self.model_name
        api_version = self.extra_config.api_version

        logger.debug(
            f"Initializing Azure OpenAI client - deployment: {deployment_name}, "
            f"api_version: {api_version}, endpoint: {self.base_url}"
        )

        self._client = AsyncAzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.base_url,  # type: ignore
            api_version=api_version,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        self._deployment_name = deployment_name

    def _convert_messages(self, messages: list) -> list:
        """
        转换消息格式（与 OpenAI 相同）

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
        logger.debug(
            f"Azure OpenAI chat request - deployment: {self._deployment_name}, "
            f"messages: {len(request.messages)}, "
            f"temperature: {request.temperature or self.default_temperature}"
        )

        try:
            response = await self._client.chat.completions.create(  # type: ignore
                model=self._deployment_name,
                messages=self._convert_messages(request.messages),
                temperature=request.temperature or self.default_temperature,
                max_tokens=request.max_tokens or self.max_tokens,
                top_p=request.top_p or self.default_top_p,
                top_k=request.top_k,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                stop=request.stop,
                tools=self._convert_tools(request.tools),
                tool_choice=self._convert_tool_choice(request.tool_choice),
                response_format=request.response_format,
                stream=False,
            )

            parsed_response = self._parse_response(response)

            logger.debug(
                f"Azure OpenAI chat response - content: {truncate_content(parsed_response.content)}, "
                f"tokens: {parsed_response.usage.total_tokens}, finish_reason: {parsed_response.finish_reason}"
            )

            return parsed_response

        except Exception as e:
            logger.error(f"Azure OpenAI API error: {e}")
            raise RuntimeError(f"Azure OpenAI API error: {str(e)}")

    def _parse_response(self, response) -> LLMResponse:
        """
        解析 Azure OpenAI 响应

        Args:
            response: OpenAI SDK 响应对象

        Returns:
            统一的 LLMResponse
        """
        from ext.llm.types import TokenUsage, ToolCall

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
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
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
            f"Azure OpenAI chat stream request - deployment: {self._deployment_name}, messages: {len(request.messages)}"
        )

        try:
            stream = await self._client.chat.completions.create(  # type: ignore
                model=self._deployment_name,
                messages=self._convert_messages(request.messages),
                temperature=request.temperature or self.default_temperature,
                max_tokens=request.max_tokens or self.max_tokens,
                top_p=request.top_p or self.default_top_p,
                top_k=request.top_k,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                stop=request.stop,
                tools=self._convert_tools(request.tools),
                tool_choice=self._convert_tool_choice(request.tool_choice),
                response_format=request.response_format,
                stream=True,
            )

            chunk_count = 0
            async for chunk in stream:
                chunk_count += 1
                yield self._parse_stream_chunk(chunk)

            logger.debug(f"Azure OpenAI stream completed - total chunks: {chunk_count}")

        except Exception as e:
            logger.error(f"Azure OpenAI API error in stream: {e}")
            raise RuntimeError(f"Azure OpenAI API error: {str(e)}")

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
            from ext.llm.types import TokenUsage

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
