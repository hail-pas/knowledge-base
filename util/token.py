"""
Token计数工具类

提供多种token计数方式：
1. 按字符计数
2. 按token计数（使用tiktoken）
3. 按中文词汇计数（使用jieba）
"""

from typing import Any, cast

from loguru import logger

try:
    import jieba  # pyright: ignore[reportMissingImports]
except ImportError:
    jieba = None
    logger.warning("jieba not installed, Chinese word segmentation disabled")

jieba = jieba if jieba is not None else cast(Any, None)

try:
    import tiktoken
except ImportError:
    tiktoken = None
    logger.warning("tiktoken not installed, token counting disabled")


class TokenCounter:
    """Token计数工具类"""

    @staticmethod
    def count_by_chars(text: str) -> int:
        """
        按字符计数

        Args:
            text: 输入文本

        Returns:
            字符数
        """
        return len(text)

    @staticmethod
    def count_by_tokens(text: str, encoding: str = "cl100k_base") -> int:
        """
        按token计数（使用tiktoken）

        Args:
            text: 输入文本
            encoding: token编码名称，默认 cl100k_base (GPT-4)

        Returns:
            token数量
        """
        if tiktoken is None:
            logger.warning("tiktoken not installed, fallback to char count. Install with: pip install tiktoken")
            return len(text)
        try:
            enc = tiktoken.get_encoding(encoding)
            return len(enc.encode(text))
        except Exception as e:
            logger.warning(f"tiktoken encoding failed: {e}, fallback to char count")
            return len(text)

    @staticmethod
    def count_by_jieba(text: str) -> int:
        """
        按中文词汇计数（使用jieba）

        Args:
            text: 输入文本

        Returns:
            词汇数量
        """
        if jieba is None:
            logger.warning("jieba not installed, fallback to char count. Install with: pip install jieba")
            return len(text)
        try:
            return len(list(jieba.cut(text)))
        except Exception as e:
            logger.warning(f"jieba cut failed: {e}, fallback to char count")
            return len(text)

    @staticmethod
    def count(text: str, mode: str = "chars", encoding: str = "cl100k_base") -> int:
        """
        通用计数方法

        Args:
            text: 输入文本
            mode: 计数模式，支持 "chars", "tokens", "jieba"
            encoding: token编码名称（仅当mode="tokens"时有效）

        Returns:
            计数结果

        Raises:
            ValueError: 不支持的计数模式
        """
        if mode == "chars":
            return TokenCounter.count_by_chars(text)
        if mode == "tokens":
            return TokenCounter.count_by_tokens(text, encoding)
        if mode == "jieba":
            return TokenCounter.count_by_jieba(text)
        raise ValueError(f"Unsupported count mode: {mode}")
