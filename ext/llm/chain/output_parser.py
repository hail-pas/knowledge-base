"""
输出解析器实现

提供将 LLM 输出转换为特定格式的解析器
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, TypeVar
from loguru import logger

from ext.llm.chain.base import Runnable

OutputT = TypeVar("OutputT")


class BaseOutputParser(Generic[OutputT], Runnable[str, OutputT]):
    """输出解析器基类

    用于将 LLM 的字符串输出转换为特定格式
    """

    @abstractmethod
    async def parse(self, text: str) -> OutputT:
        """解析文本

        Args:
            text: LLM 输出文本

        Returns:
            解析后的数据
        """
        pass

    async def ainvoke(self, input: str) -> OutputT:
        """异步调用（解析）

        Args:
            input: 输入文本

        Returns:
            解析后的数据
        """
        return await self.parse(input)


class StrOutputParser(BaseOutputParser[str]):
    """字符串输出解析器

    直接返回输入文本，不做任何转换
    """

    async def parse(self, text: str) -> str:
        """直接返回文本

        Args:
            text: 输入文本

        Returns:
            输入文本（去除首尾空格）
        """
        logger.debug(f"StrOutputParser parse - input length: {len(text)}")
        result = text.strip()
        logger.debug(f"StrOutputParser result - output length: {len(result)}")
        return result


class JsonOutputParser(BaseOutputParser[Dict[str, Any]]):
    """JSON 输出解析器

    将 LLM 输出解析为 JSON 对象
    """

    def __init__(self, pydantic_object: Optional[Any] = None):
        """初始化 JSON 解析器

        Args:
            pydantic_object: 可选的 Pydantic 模型，用于验证和转换输出
        """
        self.pydantic_object = pydantic_object

    async def parse(self, text: str) -> Dict[str, Any]:
        """解析 JSON 文本

        Args:
            text: JSON 文本

        Returns:
            解析后的字典

        Raises:
            ValueError: JSON 解析失败
        """
        logger.debug(f"JsonOutputParser parse - input length: {len(text)}")

        # 尝试提取 JSON（处理可能的前后缀文本）
        cleaned_text = self._clean_json_text(text)

        try:
            parsed = json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            logger.error(f"JsonOutputParser JSON decode error: {e}")
            raise ValueError(f"Failed to parse JSON: {e}\nInput: {text}")

        # 如果有 Pydantic 模型，进行验证
        if self.pydantic_object is not None:
            try:
                parsed = self.pydantic_object(**parsed).model_dump()
                logger.debug(f"JsonOutputParser validated with Pydantic model")
            except Exception as e:
                logger.error(f"JsonOutputParser Pydantic validation error: {e}")
                raise ValueError(f"Failed to validate with Pydantic model: {e}")

        logger.debug(f"JsonOutputParser parsed: {parsed}")
        return parsed

    def _clean_json_text(self, text: str) -> str:
        """清理 JSON 文本，移除可能的前后缀

        Args:
            text: 原始文本

        Returns:
            清理后的 JSON 文本
        """
        text = text.strip()

        # 尝试找到第一个 [ 和最后一个 ]
        start_idx = text.find("[")
        end_idx = text.rfind("]")

        if start_idx != -1 and end_idx != -1:
            return text[start_idx : end_idx + 1]

        # 尝试找到第一个 { 和最后一个 }
        start_idx = text.find("{")
        end_idx = text.rfind("}")

        if start_idx != -1 and end_idx != -1:
            return text[start_idx : end_idx + 1]

        # 返回原文本
        return text

    def get_format_instructions(self) -> str:
        """获取格式说明

        Returns:
            格式说明字符串
        """
        if self.pydantic_object is not None:
            return f"Output must be a valid JSON object matching following schema: {self.pydantic_object.model_json_schema()}"
        return "Output must be a valid JSON object."


__all__ = [
    "BaseOutputParser",
    "StrOutputParser",
    "JsonOutputParser",
]
