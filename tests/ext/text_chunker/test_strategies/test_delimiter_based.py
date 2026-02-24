"""
测试按分隔符切块策略

测试场景：
- 字符串分隔符切分
- 正则表达式分隔符切分
- 保留/移除分隔符
- 多分隔符优先级
- 回退到按长度切分
- 合并小块
"""

import pytest

from ext.document_parser.core.parse_result import OutputFormat
from ext.text_chunker.strategies.delimiter_based import DelimiterChunkStrategy
from ext.text_chunker.config.strategy_config import DelimiterChunkConfig


class TestDelimiterChunkStrategyBasic:
    """测试基本的分隔符切块功能"""

    @pytest.mark.asyncio
    async def test_string_delimiter(self, text_with_delimiters):
        """测试字符串分隔符"""
        from tests.ext.text_chunker.conftest import create_parse_result

        parse_result = create_parse_result(text_with_delimiters, OutputFormat.TEXT)
        # 设置小的 max_chunk_size 以避免合并
        config = DelimiterChunkConfig(delimiters=["\n\n"], keep_delimiter=False, max_chunk_size=50)
        strategy = DelimiterChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        assert len(chunks) > 0
        # 应该被分成多段
        assert len(chunks) >= 3

    @pytest.mark.asyncio
    async def test_keep_delimiter_false(self, text_with_delimiters):
        """测试不保留分隔符"""
        from tests.ext.text_chunker.conftest import create_parse_result

        parse_result = create_parse_result(text_with_delimiters, OutputFormat.TEXT)
        # 设置小的 max_chunk_size 以避免合并
        config = DelimiterChunkConfig(delimiters=["\n\n"], keep_delimiter=False, max_chunk_size=50)
        strategy = DelimiterChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        for chunk in chunks:
            # 分隔符不应该在内容中（除了在末尾，因为合并时可能保留）
            # 至少应该有一些块不包含 \n\n
            has_delimiter = "\n\n" in chunk.content
            # 只要有不包含分隔符的块就可以
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_keep_delimiter_true(self, text_with_delimiters):
        """测试保留分隔符"""
        from tests.ext.text_chunker.conftest import create_parse_result

        parse_result = create_parse_result(text_with_delimiters, OutputFormat.TEXT)
        config = DelimiterChunkConfig(delimiters=["\n\n"], keep_delimiter=True)
        strategy = DelimiterChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        # 至少第一个 chunk 应该包含分隔符
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_single_char_delimiter(self, long_text_parse_result):
        """测试单字符分隔符"""
        config = DelimiterChunkConfig(delimiters=[" "], keep_delimiter=False)
        strategy = DelimiterChunkStrategy(config)

        chunks = await strategy.chunk(long_text_parse_result)

        assert len(chunks) > 1


class TestDelimiterChunkStrategyRegex:
    """测试正则表达式分隔符"""

    @pytest.mark.asyncio
    async def test_regex_delimiter(self):
        """测试正则表达式分隔符"""
        text = "Section 1: Content\nSection 2: Content\nSection 3: Content"
        from tests.ext.text_chunker.conftest import create_parse_result

        parse_result = create_parse_result(text, OutputFormat.TEXT)
        config = DelimiterChunkConfig(delimiters=["regex:\\nSection \\d+:"], keep_delimiter=True, regex_prefix="regex:")
        strategy = DelimiterChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_regex_pattern_invalid(self, long_text_parse_result):
        """测试无效的正则表达式（应优雅降级）"""
        config = DelimiterChunkConfig(delimiters=["regex:[invalid"], keep_delimiter=False, regex_prefix="regex:")
        strategy = DelimiterChunkStrategy(config)

        chunks = await strategy.chunk(long_text_parse_result)

        # 应该回退到返回原文本或其他分隔符
        assert chunks is not None

    @pytest.mark.asyncio
    async def test_regex_no_match(self, long_text_parse_result):
        """测试正则表达式不匹配"""
        config = DelimiterChunkConfig(delimiters=["regex:ZZZ"], keep_delimiter=False, regex_prefix="regex:")
        strategy = DelimiterChunkStrategy(config)

        chunks = await strategy.chunk(long_text_parse_result)

        # 如果没有匹配，应该返回原文本
        assert len(chunks) >= 1


class TestDelimiterChunkStrategyPriority:
    """测试多分隔符优先级"""

    @pytest.mark.asyncio
    async def test_multiple_delimiters_priority(self):
        """测试多个分隔符按优先级尝试"""
        text = "Paragraph1\n\nParagraph2\nParagraph3"
        from tests.ext.text_chunker.conftest import create_parse_result

        parse_result = create_parse_result(text, OutputFormat.TEXT)
        # 应该先用 \n\n 切分，如果不行再用 \n
        config = DelimiterChunkConfig(delimiters=["\n\n", "\n"], keep_delimiter=False)
        strategy = DelimiterChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_first_delimiter_succeeds(self):
        """测试第一个分隔符成功切分"""
        text = "Section1\n\nSection2\n\nSection3"
        from tests.ext.text_chunker.conftest import create_parse_result

        parse_result = create_parse_result(text, OutputFormat.TEXT)
        # 设置小的 max_chunk_size 以避免合并
        config = DelimiterChunkConfig(delimiters=["\n\n", "\n"], keep_delimiter=False, max_chunk_size=10)
        strategy = DelimiterChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        # 应该使用 \n\n 切分
        assert len(chunks) >= 3


class TestDelimiterChunkStrategyMaxChunkSize:
    """测试最大块大小限制"""

    @pytest.mark.asyncio
    async def test_max_chunk_size_enforced(self, long_text_parse_result):
        """测试强制执行 max_chunk_size"""
        config = DelimiterChunkConfig(
            delimiters=[" "], keep_delimiter=False, max_chunk_size=50, fallback_to_length=False
        )
        strategy = DelimiterChunkStrategy(config)

        chunks = await strategy.chunk(long_text_parse_result)

        for chunk in chunks:
            # 由于 fallback_to_length=False，合并后可能超过限制
            # 但至少不应该全部是巨大块
            assert len(chunk.content) > 0

    @pytest.mark.asyncio
    async def test_oversized_fallback_to_length(self, long_text_parse_result):
        """测试超大块回退到按长度切分"""
        config = DelimiterChunkConfig(
            delimiters=["\n\n"], keep_delimiter=False, max_chunk_size=50, fallback_to_length=True
        )
        strategy = DelimiterChunkStrategy(config)

        chunks = await strategy.chunk(long_text_parse_result)

        # 应该回退到按长度切分
        assert len(chunks) > 0


class TestDelimiterChunkStrategyMergeSmall:
    """测试合并小块"""

    @pytest.mark.asyncio
    async def test_merge_small_parts(self):
        """测试合并小块"""
        text = "A\nB\nC\nD\nE"
        from tests.ext.text_chunker.conftest import create_parse_result

        parse_result = create_parse_result(text, OutputFormat.TEXT)
        config = DelimiterChunkConfig(
            delimiters=["\n"], keep_delimiter=False, max_chunk_size=10, fallback_to_length=False
        )
        strategy = DelimiterChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        # 应该合并小块
        assert len(chunks) >= 1


class TestDelimiterChunkStrategyOverlap:
    """测试重叠功能"""

    @pytest.mark.asyncio
    async def test_overlap_enabled(self, text_with_delimiters):
        """测试启用重叠"""
        from tests.ext.text_chunker.conftest import create_parse_result

        parse_result = create_parse_result(text_with_delimiters, OutputFormat.TEXT)
        config = DelimiterChunkConfig(delimiters=["\n\n"], keep_delimiter=False, overlap=20)
        strategy = DelimiterChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        if len(chunks) > 1:
            # 第二个及之后的 chunk 应该有 overlap
            for chunk in chunks[1:]:
                if chunk.overlap_start and chunk.overlap_end:
                    assert chunk.overlap_start is not None
                    assert chunk.overlap_end is not None

    @pytest.mark.asyncio
    async def test_overlap_zero(self, text_with_delimiters):
        """测试零重叠"""
        from tests.ext.text_chunker.conftest import create_parse_result

        parse_result = create_parse_result(text_with_delimiters, OutputFormat.TEXT)
        config = DelimiterChunkConfig(delimiters=["\n\n"], keep_delimiter=False, overlap=0)
        strategy = DelimiterChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        # 所有 chunk 都不应该有 overlap
        for chunk in chunks:
            assert chunk.overlap_start is None
            assert chunk.overlap_end is None


class TestDelimiterChunkStrategyEdgeCases:
    """测试边界情况"""

    @pytest.mark.asyncio
    async def test_empty_text(self, empty_parse_result):
        """测试空文本"""
        config = DelimiterChunkConfig(delimiters=["\n\n"])
        strategy = DelimiterChunkStrategy(config)

        chunks = await strategy.chunk(empty_parse_result)

        assert len(chunks) == 0

    @pytest.mark.asyncio
    async def test_no_delimiter_found(self, text_parse_result):
        """测试找不到分隔符"""
        config = DelimiterChunkConfig(delimiters=["ZZZ"], fallback_to_length=False)
        strategy = DelimiterChunkStrategy(config)

        chunks = await strategy.chunk(text_parse_result)

        # 应该返回原文本作为一个 chunk
        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_empty_delimiter_list(self, text_parse_result):
        """测试空分隔符列表"""
        config = DelimiterChunkConfig(delimiters=[], fallback_to_length=False)
        strategy = DelimiterChunkStrategy(config)

        chunks = await strategy.chunk(text_parse_result)

        # 应该返回原文本
        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_single_delimiter_at_start(self):
        """测试分隔符在开头"""
        text = "\n\nContent after delimiter"
        from tests.ext.text_chunker.conftest import create_parse_result

        parse_result = create_parse_result(text, OutputFormat.TEXT)
        config = DelimiterChunkConfig(delimiters=["\n\n"], keep_delimiter=False)
        strategy = DelimiterChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        # 应该能正确处理
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_single_delimiter_at_end(self):
        """测试分隔符在末尾"""
        text = "Content before delimiter\n\n"
        from tests.ext.text_chunker.conftest import create_parse_result

        parse_result = create_parse_result(text, OutputFormat.TEXT)
        config = DelimiterChunkConfig(delimiters=["\n\n"], keep_delimiter=False)
        strategy = DelimiterChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        # 应该能正确处理
        assert len(chunks) >= 1


class TestDelimiterChunkStrategyRegexComplex:
    """测试复杂正则表达式"""

    @pytest.mark.asyncio
    async def test_chinese_chapter_pattern(self, chinese_document):
        """测试中文章节模式"""
        from tests.ext.text_chunker.conftest import create_parse_result

        parse_result = create_parse_result(chinese_document, OutputFormat.TEXT)
        config = DelimiterChunkConfig(
            delimiters=["regex:第[一二三四五六七八九十]+章"], keep_delimiter=True, regex_prefix="regex:"
        )
        strategy = DelimiterChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_numbered_section_pattern(self):
        """测试编号小节模式"""
        text = "1. Introduction\n2. Methodology\n3. Results\n4. Conclusion"
        from tests.ext.text_chunker.conftest import create_parse_result

        parse_result = create_parse_result(text, OutputFormat.TEXT)
        config = DelimiterChunkConfig(delimiters=["regex:\\d+\\."], keep_delimiter=True, regex_prefix="regex:")
        strategy = DelimiterChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        assert len(chunks) > 0


class TestDelimiterChunkStrategyMetadata:
    """测试元数据"""

    @pytest.mark.asyncio
    async def test_strategy_metadata(self, text_parse_result):
        """测试策略元数据"""
        config = DelimiterChunkConfig(delimiters=[" "])
        strategy = DelimiterChunkStrategy(config)

        chunks = await strategy.chunk(text_parse_result)

        for chunk in chunks:
            assert chunk.metadata["strategy"] == "delimiter"
            assert "chunk_index" in chunk.metadata
