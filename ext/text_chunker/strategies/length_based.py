"""
按长度切块策略

支持按字符数或token数进行切块
"""

from loguru import logger

from ext.document_parser.core.parse_result import ParseResult
from ext.text_chunker.config.strategy_config import LengthChunkConfig
from ext.text_chunker.strategies.base import BaseChunkStrategy
from util.token import TokenCounter


class LengthChunkStrategy(BaseChunkStrategy[LengthChunkConfig]):
    """按长度切块策略

    根据配置的字符数或token数进行等长切块，
    支持chunk之间的overlap
    """

    async def chunk(self, parse_result: ParseResult) -> list:
        """
        执行按长度切块

        Args:
            parse_result: 文档解析结果

        Returns:
            切块结果列表
        """
        from ext.text_chunker.core.coordinate_mapper import CoordinateMapper

        self._set_mapper(CoordinateMapper(parse_result))

        text = parse_result.content
        chunk_size = self.config.chunk_size
        overlap = self.config.overlap
        mode = self.config.mode

        if mode == "tokens":
            # token模式下，先计算总token数
            total_tokens = TokenCounter.count_by_tokens(text, self.config.encoding)
            if total_tokens == 0:
                logger.warning("Empty text, returning empty chunks")
                return []
            # 使用字符长度作为近似，按比例调整chunk_size
            char_ratio = len(text) / total_tokens
            chunk_size = int(chunk_size * char_ratio)
            overlap = int(overlap * char_ratio)
            logger.debug(f"Token mode: {total_tokens} tokens, adjusted chunk_size={chunk_size}, overlap={overlap}")

        chunks = []
        start = 0
        chunk_index = 0
        previous_end = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            content = text[start:end]

            overlap_start = None
            overlap_end = None
            if overlap > 0 and chunk_index > 0 and start < previous_end:
                overlap_start = start
                overlap_end = min(previous_end, end)

            chunk = self._build_chunk(
                content=content,
                global_start=start,
                global_end=end,
                overlap_start=overlap_start,
                overlap_end=overlap_end,
                metadata={"chunk_index": chunk_index, "strategy": "length"},
            )
            chunks.append(chunk)
            previous_end = end

            # 如果已经到达文本末尾，退出循环
            if end >= len(text):
                break

            # 移动起始位置（考虑overlap）
            if overlap > 0 and end > overlap:
                start = end - overlap
            else:
                start = end
            chunk_index += 1

            # 避免无限循环（overlap >= chunk_size时）
            if overlap >= chunk_size and len(chunks) > 1:
                logger.warning(f"Overlap ({overlap}) >= chunk_size ({chunk_size}), breaking to avoid infinite loop")
                break

        logger.info(f"Chunked text into {len(chunks)} chunks using length strategy")
        return chunks
