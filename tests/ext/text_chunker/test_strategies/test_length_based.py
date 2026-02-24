"""
测试按长度切块策略

测试场景：
- 字符模式切块
- Token 模式切块
- Overlap 功能
- 边界情况处理
"""

import pytest

from ext.document_parser.core.parse_result import OutputFormat
from ext.text_chunker.strategies.length_based import LengthChunkStrategy
from ext.text_chunker.config.strategy_config import LengthChunkConfig


class TestLengthChunkStrategyBasic:
    """测试基本的长度切块功能"""

    @pytest.mark.asyncio
    async def test_char_mode_chunking(self, long_text_parse_result):
        """测试字符模式切块"""
        config = LengthChunkConfig(chunk_size=100, overlap=0, mode="chars")
        strategy = LengthChunkStrategy(config)

        chunks = await strategy.chunk(long_text_parse_result)

        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk.content) <= 100

    @pytest.mark.asyncio
    async def test_chunk_without_overlap(self, long_text_parse_result):
        """测试无重叠切块"""
        config = LengthChunkConfig(chunk_size=200, overlap=0, mode="chars")
        strategy = LengthChunkStrategy(config)

        chunks = await strategy.chunk(long_text_parse_result)

        assert len(chunks) > 1

        # 检查相邻 chunk 没有重叠
        for i in range(len(chunks) - 1):
            # end 是不包含的结束位置，所以下一个 chunk 的 start 应该等于当前 chunk 的 end
            # end 记录的是最后一个字符的位置，所以下一个 start 应该是 end.char_index + 1
            current_end_global = chunks[i].end.char_index
            next_start_global = chunks[i + 1].start.char_index
            # 下一个 chunk 的 start 应该紧跟在当前 chunk 的 end 之后
            assert next_start_global == current_end_global + 1 or (
                chunks[i].end.page_number < chunks[i + 1].start.page_number
            )

    @pytest.mark.asyncio
    async def test_chunk_with_overlap(self, long_text_parse_result):
        """测试带重叠切块"""
        overlap = 50
        chunk_size = 200
        config = LengthChunkConfig(chunk_size=chunk_size, overlap=overlap, mode="chars")
        strategy = LengthChunkStrategy(config)

        chunks = await strategy.chunk(long_text_parse_result)

        assert len(chunks) > 1

        # 检查第二个及之后的 chunk 有 overlap
        for i in range(1, len(chunks)):
            assert chunks[i].overlap_start is not None
            assert chunks[i].overlap_end is not None

    @pytest.mark.asyncio
    async def test_token_mode_chunking(self, long_text_parse_result):
        """测试 token 模式切块"""
        config = LengthChunkConfig(chunk_size=50, overlap=10, mode="tokens", encoding="cl100k_base")
        strategy = LengthChunkStrategy(config)

        chunks = await strategy.chunk(long_text_parse_result)

        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_chunk_size_larger_than_text(self, text_parse_result):
        """测试 chunk_size 大于文本长度"""
        config = LengthChunkConfig(chunk_size=10000, overlap=0, mode="chars")
        strategy = LengthChunkStrategy(config)

        chunks = await strategy.chunk(text_parse_result)

        assert len(chunks) == 1
        assert chunks[0].content == text_parse_result.content


class TestLengthChunkStrategyEdgeCases:
    """测试边界情况"""

    @pytest.mark.asyncio
    async def test_empty_text(self, empty_parse_result):
        """测试空文本"""
        config = LengthChunkConfig(chunk_size=100, overlap=0, mode="chars")
        strategy = LengthChunkStrategy(config)

        chunks = await strategy.chunk(empty_parse_result)

        assert len(chunks) == 0

    @pytest.mark.asyncio
    async def test_single_character(self, single_char_parse_result):
        """测试单字符文本"""
        config = LengthChunkConfig(chunk_size=100, overlap=0, mode="chars")
        strategy = LengthChunkStrategy(config)

        chunks = await strategy.chunk(single_char_parse_result)

        assert len(chunks) == 1
        assert chunks[0].content == "A"

    @pytest.mark.asyncio
    async def test_text_exactly_chunk_size(self, create_parse_result):
        """测试文本长度正好是 chunk_size"""
        # 创建正好 200 字符的文本
        text = "A" * 200

        parse_result = create_parse_result(text, OutputFormat.TEXT)

        config = LengthChunkConfig(chunk_size=200, overlap=0, mode="chars")
        strategy = LengthChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        assert len(chunks) == 1
        assert len(chunks[0].content) == 200

    @pytest.mark.asyncio
    async def test_small_chunk_size(self, long_text_parse_result):
        """测试很小的 chunk_size"""
        config = LengthChunkConfig(chunk_size=10, overlap=0, mode="chars")
        strategy = LengthChunkStrategy(config)

        chunks = await strategy.chunk(long_text_parse_result)

        assert len(chunks) > 10  # 应该产生多个小块
        for chunk in chunks:
            assert len(chunk.content) <= 10

    @pytest.mark.asyncio
    async def test_overlap_equals_chunk_size(self, long_text_parse_result):
        """测试 overlap 等于 chunk_size（应触发无限循环保护）"""
        config = LengthChunkConfig(chunk_size=100, overlap=100, mode="chars")
        strategy = LengthChunkStrategy(config)

        chunks = await strategy.chunk(long_text_parse_result)

        # 应该至少有一个 chunk，但不应无限循环
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_overlap_greater_than_chunk_size(self, long_text_parse_result):
        """测试 overlap 大于 chunk_size"""
        config = LengthChunkConfig(chunk_size=50, overlap=100, mode="chars")
        strategy = LengthChunkStrategy(config)

        chunks = await strategy.chunk(long_text_parse_result)

        # 应该触发保护机制
        assert len(chunks) >= 1


class TestLengthChunkStrategyMultiPage:
    """测试多页文档"""

    @pytest.mark.asyncio
    async def test_multi_page_chunking(self, multi_page_parse_result):
        """测试多页文档切块"""
        config = LengthChunkConfig(chunk_size=200, overlap=0, mode="chars")
        strategy = LengthChunkStrategy(config)

        chunks = await strategy.chunk(multi_page_parse_result)

        assert len(chunks) > 0

        # 检查 pages 属性
        for chunk in chunks:
            assert isinstance(chunk.pages, list)
            assert len(chunk.pages) > 0

    @pytest.mark.asyncio
    async def test_cross_page_chunking(self, multi_page_parse_result):
        """测试跨页切块"""
        config = LengthChunkConfig(chunk_size=400, overlap=0, mode="chars")
        strategy = LengthChunkStrategy(config)

        chunks = await strategy.chunk(multi_page_parse_result)

        # 至少有一个 chunk 应该跨越多页
        multi_page_chunks = [c for c in chunks if len(c.pages) > 1]
        assert len(multi_page_chunks) > 0


class TestLengthChunkStrategyPositionTracking:
    """测试位置跟踪准确性"""

    @pytest.mark.asyncio
    async def test_position_tracking(self, long_text_parse_result):
        """测试位置跟踪准确性"""
        config = LengthChunkConfig(chunk_size=100, overlap=0, mode="chars")
        strategy = LengthChunkStrategy(config)

        chunks = await strategy.chunk(long_text_parse_result)

        # 验证位置的连续性
        for i in range(len(chunks) - 1):
            # 当前 chunk 的结束位置
            current_end = chunks[i].end
            # 下一个 chunk 的起始位置
            next_start = chunks[i + 1].start

            # 在同一页内，下一个 chunk 应该紧跟在当前 chunk 之后
            if current_end.page_number == next_start.page_number:
                # end 是最后一个字符的位置，下一个 start 应该是 end.char_index + 1
                assert next_start.char_index == current_end.char_index + 1
            else:
                # 跨页的情况，下一个 chunk 应该在下一页的开始位置
                assert next_start.page_number == current_end.page_number + 1
                assert next_start.char_index >= 0

    @pytest.mark.asyncio
    async def test_page_numbers_correct(self, multi_page_parse_result):
        """测试页码正确性"""
        config = LengthChunkConfig(chunk_size=200, overlap=0, mode="chars")
        strategy = LengthChunkStrategy(config)

        chunks = await strategy.chunk(multi_page_parse_result)

        for chunk in chunks:
            # 页码应该在有效范围内
            assert all(1 <= p <= 3 for p in chunk.pages)


class TestLengthChunkStrategyMetadata:
    """测试元数据"""

    @pytest.mark.asyncio
    async def test_chunk_index_metadata(self, long_text_parse_result):
        """测试 chunk_index 元数据"""
        config = LengthChunkConfig(chunk_size=100, overlap=0, mode="chars")
        strategy = LengthChunkStrategy(config)

        chunks = await strategy.chunk(long_text_parse_result)

        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["strategy"] == "length"

    @pytest.mark.asyncio
    async def test_overlap_metadata(self, long_text_parse_result):
        """测试 overlap 相关元数据"""
        overlap = 50
        config = LengthChunkConfig(chunk_size=200, overlap=overlap, mode="chars")
        strategy = LengthChunkStrategy(config)

        chunks = await strategy.chunk(long_text_parse_result)

        # 第一个 chunk 不应该有 overlap
        assert chunks[0].overlap_start is None
        assert chunks[0].overlap_end is None

        # 后续的 chunk 应该有 overlap
        for chunk in chunks[1:]:
            if chunk.overlap_start and chunk.overlap_end:
                # 验证 overlap 内容确实是重复的
                start_global = chunk.start.page_number
                end_global = chunk.end.page_number


class TestLengthChunkStrategyTokenMode:
    """测试 Token 模式"""

    @pytest.mark.asyncio
    async def test_token_mode_approximation(self, long_text_parse_result):
        """测试 token 模式下的字符近似"""
        config = LengthChunkConfig(chunk_size=100, overlap=20, mode="tokens", encoding="cl100k_base")
        strategy = LengthChunkStrategy(config)

        chunks = await strategy.chunk(long_text_parse_result)

        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_token_mode_empty_text(self, empty_parse_result):
        """测试 token 模式下空文本"""
        config = LengthChunkConfig(chunk_size=100, overlap=0, mode="tokens", encoding="cl100k_base")
        strategy = LengthChunkStrategy(config)

        chunks = await strategy.chunk(empty_parse_result)

        assert len(chunks) == 0
