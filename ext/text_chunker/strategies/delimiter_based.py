"""
按分隔符切块策略

按照自定义分隔符进行切块
"""

import re
from loguru import logger

from ext.document_parser.core.parse_result import ParseResult
from ext.text_chunker.config.strategy_config import DelimiterChunkConfig
from ext.text_chunker.strategies.base import BaseChunkStrategy


class DelimiterChunkStrategy(BaseChunkStrategy[DelimiterChunkConfig]):
    """按分隔符切块策略

    按照配置的分隔符列表进行切块，
    如果单个切块超过max_chunk_size，可回退到按长度切块
    """

    async def chunk(self, parse_result: ParseResult) -> list:
        """
        执行按分隔符切块

        Args:
            parse_result: 文档解析结果

        Returns:
            切块结果列表
        """
        from ext.text_chunker.core.coordinate_mapper import CoordinateMapper

        self._set_mapper(CoordinateMapper(parse_result))

        text = parse_result.content
        delimiters = self.config.delimiters
        regex_prefix = self.config.regex_prefix
        keep_delimiter = self.config.keep_delimiter
        max_chunk_size = self.config.max_chunk_size
        overlap = self.config.overlap
        fallback_to_length = self.config.fallback_to_length

        # 按优先级尝试分隔符
        chunks = self._split_by_delimiters(
            text, delimiters, regex_prefix, keep_delimiter, max_chunk_size, overlap, fallback_to_length
        )

        # 构建ChunkResult列表
        results = []
        global_pos = 0
        chunk_index = 0

        for chunk_text in chunks:
            if not chunk_text:
                continue

            start = global_pos
            end = global_pos + len(chunk_text)
            global_pos = end

            overlap_start = None
            overlap_end = None
            if overlap > 0 and chunk_index > 0:
                overlap_start = start - overlap
                overlap_end = start

            chunk = self._build_chunk(
                content=chunk_text,
                global_start=start,
                global_end=end,
                overlap_start=overlap_start,
                overlap_end=overlap_end,
                metadata={"chunk_index": chunk_index, "strategy": "delimiter"},
            )
            results.append(chunk)
            chunk_index += 1

        logger.info(f"Chunked text into {len(results)} chunks using delimiter strategy")
        return results

    def _split_by_delimiters(
        self,
        text: str,
        delimiters: list[str],
        regex_prefix: str,
        keep_delimiter: bool,
        max_chunk_size: int,
        overlap: int,
        fallback_to_length: bool,
    ) -> list[str]:
        """
        按分隔符切分文本

        Args:
            text: 输入文本
            delimiters: 分隔符列表（支持前缀标记区分正则表达式）
            regex_prefix: 正则表达式前缀标记
            keep_delimiter: 是否保留分隔符
            max_chunk_size: 最大切块大小
            overlap: 重叠大小
            fallback_to_length: 超长时是否回退到按长度切块

        Returns:
            切块列表
        """
        for delimiter in delimiters:
            is_regex = delimiter.startswith(regex_prefix)
            actual_delim = delimiter[len(regex_prefix) :] if is_regex else delimiter

            if is_regex:
                parts = self._split_by_regex(text, actual_delim, keep_delimiter)
            else:
                parts = self._split_by_string(text, actual_delim, keep_delimiter)

            if parts is None:
                continue

            if len(parts) <= 1:
                continue

            has_oversized = any(len(part) > max_chunk_size for part in parts)

            if has_oversized and fallback_to_length:
                logger.info(
                    f"Some parts exceed max_chunk_size ({max_chunk_size}), falling back to length-based splitting"
                )
                return self._split_by_length(text, max_chunk_size, overlap)

            chunks = self._merge_small_parts(parts, max_chunk_size, keep_delimiter)
            return chunks

        if fallback_to_length:
            logger.warning("No delimiter produced valid splits, falling back to length-based")
            return self._split_by_length(text, max_chunk_size, overlap)

        return [text]

    def _split_by_string(self, text: str, delimiter: str, keep_delimiter: bool) -> list[str] | None:
        """
        按普通字符串分隔符切分

        Args:
            text: 输入文本
            delimiter: 字符串分隔符
            keep_delimiter: 是否保留分隔符

        Returns:
            切块列表，如果分隔符为空字符串返回None
        """
        if not delimiter:
            return None

        escaped_delim = re.escape(delimiter)
        pattern = f"{escaped_delim}(?!$)" if keep_delimiter else f"(?<={escaped_delim})"
        return re.split(pattern, text)

    def _split_by_regex(self, text: str, pattern: str, keep_delimiter: bool) -> list[str] | None:
        """
        按正则表达式分隔符切分，支持精确保留分隔符

        Args:
            text: 输入文本
            pattern: 正则表达式模式
            keep_delimiter: 是否保留分隔符

        Returns:
            切块列表，如果正则表达式无效返回None
        """
        if not pattern:
            return None

        try:
            compiled = re.compile(pattern)
        except re.error as e:
            logger.warning(f"Invalid regex pattern '{pattern}': {e}")
            return None

        parts = []
        last_end = 0

        for match in compiled.finditer(text):
            parts.append(text[last_end : match.start()])

            if keep_delimiter:
                parts[-1] += match.group()

            last_end = match.end()

        if last_end < len(text):
            parts.append(text[last_end:])

        return [p for p in parts if p]

    def _split_by_length(self, text: str, max_chunk_size: int, overlap: int) -> list[str]:
        """
        按长度切分文本

        Args:
            text: 输入文本
            max_chunk_size: 最大切块大小
            overlap: 重叠大小

        Returns:
            切块列表
        """
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + max_chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - overlap if overlap > 0 else end

        return chunks

    def _merge_small_parts(self, parts: list[str], max_chunk_size: int, keep_delimiter: bool) -> list[str]:
        """
        合并过小的文本块

        Args:
            parts: 原始分割结果
            max_chunk_size: 最大切块大小
            keep_delimiter: 是否保留分隔符

        Returns:
            合并后的切块列表
        """
        if not parts:
            return []

        chunks = []
        current_chunk = parts[0]

        for part in parts[1:]:
            # 检查添加分隔符后的长度
            test_chunk = current_chunk + part if keep_delimiter else current_chunk + part

            if len(test_chunk) <= max_chunk_size:
                current_chunk = test_chunk
            else:
                # 超过大小，保存当前chunk，开始新的chunk
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = part

        # 添加最后一个chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks
