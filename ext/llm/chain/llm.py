"""
LLM Runnable 封装

提供对 ext/llm 模型的简洁封装
"""

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from ext.llm.base import BaseLLMModel
from ext.llm.types import ChatMessage, LLMRequest, StreamChunk
from loguru import logger

from ext.llm.chain.base import Runnable


class LLM(Runnable[str, str]):
    """LLM Runnable 封装

    提供简洁的接口来调用 LLM 模型
    """

    def __init__(
        self,
        model: BaseLLMModel,
        default_temperature: Optional[float] = None,
        default_max_tokens: Optional[int] = None,
    ):
        """初始化 LLM

        Args:
            model: LLM 模型实例
            default_temperature: 默认温度参数
            default_max_tokens: 默认最大 token 数
        """
        self.model = model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

    @classmethod
    def from_name(
        cls,
        model_name: str,
        default_temperature: Optional[float] = None,
        default_max_tokens: Optional[int] = None,
    ):
        """从配置名称创建 LLM

        Args:
            model_name: 模型配置名称
            default_temperature: 默认温度参数
            default_max_tokens: 默认最大 token 数

        Returns:
            LLM 实例或延迟初始化的包装器
        """

        return _LLMFactoryWrapper(model_name, default_temperature, default_max_tokens)

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        default_temperature: Optional[float] = None,
        default_max_tokens: Optional[int] = None,
    ):
        """从配置字典创建 LLM

        Args:
            config: 模型配置字典
            default_temperature: 默认温度参数
            default_max_tokens: 默认最大 token 数

        Returns:
            LLM 实例
        """
        from ext.llm import LLMModelFactory
        from ext.ext_tortoise.models.knowledge_base import LLMModelConfig

        model_config = LLMModelConfig(**config)
        model = asyncio.run(LLMModelFactory.create(model_config, use_cache=False))
        return cls(model, default_temperature, default_max_tokens)

    async def ainvoke(
        self,
        input: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """调用 LLM（非流式）

        Args:
            input: 输入文本
            temperature: 温度参数（覆盖默认值）
            max_tokens: 最大 token 数（覆盖默认值）
            **kwargs: 其他 LLM 参数

        Returns:
            LLM 响应文本

        Raises:
            RuntimeError: LLM 调用失败
        """
        logger.debug(
            f"LLM ainvoke - input length: {len(input)}, "
            f"temperature: {temperature or self.default_temperature}, "
            f"max_tokens: {max_tokens or self.default_max_tokens}"
        )

        request = LLMRequest(
            messages=[ChatMessage(role="user", content=input)],
            temperature=temperature or self.default_temperature or self.model.default_temperature,
            max_tokens=max_tokens or self.default_max_tokens or self.model.max_tokens,
            **kwargs,
        )

        try:
            response = await self.model.chat(request)
            logger.debug(f"LLM ainvoke result - output length: {len(response.content or '')}")
            return response.content or ""
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            raise RuntimeError(f"LLM invocation failed: {e}")

    async def astream(
        self,
        input: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """流式调用 LLM

        Args:
            input: 输入文本
            temperature: 温度参数（覆盖默认值）
            max_tokens: 最大 token 数（覆盖默认值）
            **kwargs: 其他 LLM 参数

        Yields:
            流式输出的文本块

        Raises:
            RuntimeError: LLM 调用失败
        """
        logger.debug(
            f"LLM astream - input length: {len(input)}, temperature: {temperature or self.default_temperature}"
        )

        request = LLMRequest(
            messages=[ChatMessage(role="user", content=input)],
            temperature=temperature or self.default_temperature or self.model.default_temperature,
            max_tokens=max_tokens or self.default_max_tokens or self.model.max_tokens,
            stream=True,
            **kwargs,
        )

        try:
            chunk_count = 0
            async for chunk in self.model.chat_stream(request):  # type: ignore
                if "content" in chunk.delta:
                    chunk_count += 1
                    yield chunk.delta["content"]
            logger.debug(f"LLM astream completed - total chunks: {chunk_count}")
        except Exception as e:
            logger.error(f"LLM streaming failed: {e}")
            raise RuntimeError(f"LLM streaming failed: {e}")

    async def ainvoke_with_messages(
        self,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """使用消息列表调用 LLM

        Args:
            messages: 聊天消息列表
            temperature: 温度参数
            max_tokens: 最大 token 数
            **kwargs: 其他 LLM 参数

        Returns:
            LLM 响应文本
        """
        logger.debug(
            f"LLM ainvoke_with_messages - messages: {len(messages)}, "
            f"temperature: {temperature or self.default_temperature}"
        )

        request = LLMRequest(
            messages=messages,
            temperature=temperature or self.default_temperature or self.model.default_temperature,
            max_tokens=max_tokens or self.default_max_tokens or self.model.max_tokens,
            **kwargs,
        )

        try:
            response = await self.model.chat(request)
            logger.debug(f"LLM ainvoke_with_messages result - output length: {len(response.content or '')}")
            return response.content or ""
        except Exception as e:
            logger.error(f"LLM invocation with messages failed: {e}")
            raise RuntimeError(f"LLM invocation with messages failed: {e}")

    async def abatch(
        self,
        inputs: List[str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> List[str]:
        """批量调用 LLM

        Args:
            inputs: 输入文本列表
            temperature: 温度参数
            max_tokens: 最大 token 数
            **kwargs: 其他 LLM 参数

        Returns:
            LLM 响应文本列表
        """
        tasks = [self.ainvoke(inp, temperature, max_tokens, **kwargs) for inp in inputs]
        return await asyncio.gather(*tasks)


class _LLMFactoryWrapper:
    """延迟初始化的 LLM 工厂包装器

    用于处理异步环境中的创建问题
    """

    def __init__(
        self,
        model_name: str,
        default_temperature: Optional[float],
        default_max_tokens: Optional[int],
    ):
        self._model_name = model_name
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens
        self._llm_instance: Optional[LLM] = None

    async def _get_instance(self) -> LLM:
        """获取或创建 LLM 实例"""
        if self._llm_instance is None:
            from ext.llm import LLMModelFactory

            model = await LLMModelFactory.create_by_name(self._model_name)
            self._llm_instance = LLM(model)
        return self._llm_instance

    async def ainvoke(self, input: str, **kwargs: Any) -> str:
        """调用 LLM（非流式）"""
        llm = await self._get_instance()
        return await llm.ainvoke(input, **kwargs)

    async def astream(self, input: str, **kwargs: Any) -> AsyncIterator[str]:
        """流式调用 LLM"""
        llm = await self._get_instance()
        async for chunk in llm.astream(input, **kwargs):
            yield chunk

    async def abatch(self, inputs: List[str], **kwargs: Any) -> List[str]:  # type: ignore
        """批量调用 LLM"""
        llm = await self._get_instance()
        return await llm.abatch(inputs, **kwargs)

    async def ainvoke_with_messages(self, messages: List[ChatMessage], **kwargs: Any) -> str:
        """使用消息列表调用 LLM"""
        llm = await self._get_instance()
        return await llm.ainvoke_with_messages(messages, **kwargs)

    async def abatch(self, inputs: List[str], **kwargs: Any) -> List[str]:
        """批量调用 LLM"""
        llm = await self._get_instance()
        return await llm.abatch(inputs, **kwargs)

    @property
    def model(self):
        """获取底层 LLM 模型实例"""
        llm = self._llm_instance
        if llm is not None:
            return llm.model
        raise AttributeError("LLM not yet initialized. Use in async context.")


__all__ = [
    "LLM",
]
