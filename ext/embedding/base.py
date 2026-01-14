from abc import ABC, abstractmethod
from typing import List, Optional, Union
from pydantic import BaseModel
import numpy as np
from loguru import logger


class EmbeddingResult(BaseModel):
    """Embedding 结果数据结构"""

    embedding: List[float]
    """向量数据"""

    index: int
    """在原始输入中的索引"""

    text: str
    """原始文本"""

    model: str = ""
    """使用的模型标识"""

    class Config:
        arbitrary_types_allowed = True


class EmbeddingModel(ABC):
    """Embedding 模型抽象基类

    所有 embedding 模型实现必须继承此类并实现核心方法。
    提供统一的接口，支持动态切换不同的 embedding 服务提供商。
    """

    def __init__(
        self,
        model_name_or_path: str,
        dimension: int,
        max_batch_size: int = 32,
        max_token_per_request: int = 8191,
        max_token_per_text: Optional[int] = None,
        config: dict | None = None,
    ):
        """
        初始化 Embedding 模型

        Args:
            model_name_or_path: 模型标识符或路径
            dimension: 向量维度
            max_batch_size: 最大批处理大小，用于 embed_batch 的自动分批
            max_token_per_request: 单次请求最大 token 数
            max_token_per_text: 单个文本最大 token 数，超过时会打印警告日志
            config: 模型配置参数（如 API key、endpoint 等）
        """
        self.model_name_or_path = model_name_or_path
        self._dimension = dimension
        self.max_batch_size = max_batch_size
        self.max_token_per_request = max_token_per_request
        self.max_token_per_text = max_token_per_text if max_token_per_text is not None else max_token_per_request
        self.config = config or {}

        # 估算 token 数的简单实现（基于字符的粗略估计，4 chars ≈ 1 token）
        # 子类可以覆盖此方法以使用更精确的 tokenizer
        self._chars_per_token = 4

    @property
    def dimension(self) -> int:
        """获取向量维度"""
        return self._dimension

    @abstractmethod
    async def _embed_batch_impl(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """
        实际执行批量 embedding 的实现方法（由子类实现）

        Args:
            texts: 文本列表

        Returns:
            向量列表

        Raises:
            EmbeddingAPIError: API 调用失败
            EmbeddingTimeoutError: 请求超时
        """
        pass

    def _estimate_tokens(self, text: str) -> int:
        """
        估算文本的 token 数量（基于字符的粗略估计）

        Args:
            text: 输入文本

        Returns:
            估算的 token 数量
        """
        return (len(text) + self._chars_per_token - 1) // self._chars_per_token

    def _split_by_token_limit(
        self,
        texts: List[str],
        max_tokens: Optional[int] = None
    ) -> List[List[str]]:
        """
        根据 token 限制将文本列表分成多个批次

        Args:
            texts: 原始文本列表
            max_tokens: 单批次最大 token 数，默认使用 self.max_token_per_request

        Returns:
            分批后的文本列表的列表
        """
        if max_tokens is None:
            max_tokens = self.max_token_per_request

        batches = []
        current_batch = []
        current_tokens = 0

        for text in texts:
            text_tokens = self._estimate_tokens(text)

            # 如果单个文本就超过限制，需要单独处理或截断
            # 这里选择单独处理，让 API 自己决定是否拒绝
            if text_tokens > max_tokens:
                logger.warning(
                    f"单个文本的 token 长度 ({text_tokens}) 超过了单批次的 token 限制 ({max_tokens})。"
                    f"文本预览: {text[:100]}{'...' if len(text) > 100 else ''}"
                )
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
                batches.append([text])
                continue

            # 检查加入当前批次是否会超过限制
            if current_tokens + text_tokens > max_tokens:
                batches.append(current_batch)
                current_batch = [text]
                current_tokens = text_tokens
            else:
                current_batch.append(text)
                current_tokens += text_tokens

        # 添加最后一个批次
        if current_batch:
            batches.append(current_batch)

        return batches

    def _split_by_batch_size(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> List[List[str]]:
        """
        根据批处理大小将文本列表分成多个批次

        Args:
            texts: 原始文本列表
            batch_size: 每批大小，默认使用 self.max_batch_size

        Returns:
            分批后的文本列表的列表
        """
        if batch_size is None:
            batch_size = self.max_batch_size

        return [
            texts[i:i + batch_size]
            for i in range(0, len(texts), batch_size)
        ]

    async def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> List[EmbeddingResult]:
        """
        批量生成文本 embedding（支持自动分批处理）

        此方法会自动处理以下情况：
        1. 文本数量超过 batch_size 时自动分批
        2. 单批文本的 token 总数超过 max_tokens 时自动分批

        Args:
            texts: 文本列表
            batch_size: 每批大小，None 时使用 self.max_batch_size
            max_tokens: 单批次最大 token 数，None 时使用 self.max_token_per_request

        Returns:
            EmbeddingResult 列表，顺序与输入文本一致

        Raises:
            EmbeddingConfigError: 配置错误
            EmbeddingAPIError: API 调用失败
            EmbeddingTimeoutError: 请求超时
            EmbeddingBatchError: 批处理部分失败
        """
        if not texts:
            return []

        if batch_size is None:
            batch_size = self.max_batch_size

        if max_tokens is None:
            max_tokens = self.max_token_per_request

        # 先按 batch_size 分批
        size_based_batches = self._split_by_batch_size(texts, batch_size)

        # 再在每个批次内按 token 限制进一步分批
        final_batches = []
        for batch in size_based_batches:
            token_based_batches = self._split_by_token_limit(batch, max_tokens)
            final_batches.extend(token_based_batches)

        # 执行批量 embedding
        all_embeddings: list[EmbeddingResult] = []
        index = 0

        for batch in final_batches:
            # 检查单个文本的 token 长度是否超过配置的限制
            for text in batch:
                text_tokens = self._estimate_tokens(text)
                if text_tokens > self.max_token_per_text:
                    logger.warning(
                        f"单个文本的 token 长度 ({text_tokens}) 超过了配置的最大长度 ({self.max_token_per_text})。"
                        f"模型: {self.model_name_or_path}, "
                        f"文本预览: {text[:100]}{'...' if len(text) > 100 else ''}"
                    )

            try:
                embeddings = await self._embed_batch_impl(batch)

                # 构造结果对象
                for i, (text, embedding) in enumerate(zip(batch, embeddings)):
                    result = EmbeddingResult(
                        embedding=embedding,
                        index=index,
                        text=text,
                        model=self.model_name_or_path,
                    )
                    all_embeddings.append(result)
                    index += 1

            except Exception as e:
                # 转换为自定义异常
                from ext.embedding.exceptions import (
                    EmbeddingAPIError,
                    EmbeddingTimeoutError,
                    EmbeddingBatchError
                )

                error_msg = f"Batch embedding failed: {str(e)}"

                # 判断是否是超时
                if "timeout" in str(e).lower():
                    raise EmbeddingTimeoutError(error_msg) from e

                # 判断是否是 API 错误
                raise EmbeddingAPIError(error_msg) from e

        return all_embeddings

    async def embed(
        self,
        text: str,
    ) -> EmbeddingResult:
        """
        生成单个文本的 embedding

        Args:
            text: 输入文本

        Returns:
            EmbeddingResult 对象

        Raises:
            EmbeddingAPIError: API 调用失败
            EmbeddingTimeoutError: 请求超时
        """
        results = await self.embed_batch([text])
        return results[0]

    async def get_embeddings(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """
        便捷方法：直接获取向量列表（不返回元数据）

        Args:
            texts: 文本列表

        Returns:
            向量列表（二维列表）
        """
        results = await self.embed_batch(texts)
        return [result.embedding for result in results]

    def get_dimension(self) -> int:
        """获取向量维度"""
        return self.dimension

    def validate_config(self, required_keys: List[str] | None = None) -> None:
        """
        验证配置是否包含必需的参数

        Args:
            required_keys: 必需的配置键列表

        Raises:
            EmbeddingConfigError: 配置缺少必需参数
        """
        if required_keys is None:
            return

        missing_keys = [key for key in required_keys if key not in self.config]
        if missing_keys:
            from ext.embedding.exceptions import EmbeddingConfigError
            raise EmbeddingConfigError(
                f"Missing required config keys: {', '.join(missing_keys)}"
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name={self.model_name_or_path}, "
            f"dimension={self.dimension}, "
            f"max_batch_size={self.max_batch_size})"
        )
