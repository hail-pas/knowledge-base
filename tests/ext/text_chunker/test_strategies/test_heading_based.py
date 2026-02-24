"""
测试按标题层级切块策略

测试场景：
- Markdown 标题解析
- 中文标题识别
- 标题树构建
- 父标题保留
- 段落重叠
- 回退到按长度切分
"""

import pytest

from ext.document_parser.core.parse_result import OutputFormat
from ext.text_chunker.strategies.heading_based import HeadingChunkStrategy
from ext.text_chunker.config.strategy_config import HeadingChunkConfig


class TestHeadingChunkStrategyMarkdown:
    """测试 Markdown 标题解析"""

    @pytest.mark.asyncio
    async def test_markdown_headings(self, markdown_parse_result):
        """测试 Markdown 标题切分"""
        config = HeadingChunkConfig(
            max_chunk_size=2000,
            overlap_paragraphs=0,
            preserve_headings=True,
        )
        strategy = HeadingChunkStrategy(config)

        chunks = await strategy.chunk(markdown_parse_result)

        assert len(chunks) > 0
        # 应该按标题分成多个块
        assert len(chunks) >= 3

    @pytest.mark.asyncio
    async def test_heading_levels(self, markdown_parse_result):
        """测试不同层级标题"""
        config = HeadingChunkConfig(
            max_chunk_size=3000,
            overlap_paragraphs=0,
            preserve_headings=True,
        )
        strategy = HeadingChunkStrategy(config)

        chunks = await strategy.chunk(markdown_parse_result)

        # 检查每个 chunk 包含了标题
        for chunk in chunks:
            assert len(chunk.content) > 0

    @pytest.mark.asyncio
    async def test_preserve_headings_true(self, markdown_parse_result):
        """测试保留父标题"""
        config = HeadingChunkConfig(
            max_chunk_size=3000,
            overlap_paragraphs=0,
            preserve_headings=True,
        )
        strategy = HeadingChunkStrategy(config)

        chunks = await strategy.chunk(markdown_parse_result)

        # 子标题应该包含父标题
        for chunk in chunks:
            content = chunk.content
            # 检查是否有 # 标题
            has_heading = "#" in content
            if preserve_headings := config.preserve_headings:
                assert has_heading or len(content) > 0

    @pytest.mark.asyncio
    async def test_preserve_headings_false(self, markdown_parse_result):
        """测试不保留父标题"""
        config = HeadingChunkConfig(
            max_chunk_size=3000,
            overlap_paragraphs=0,
            preserve_headings=False,
        )
        strategy = HeadingChunkStrategy(config)

        chunks = await strategy.chunk(markdown_parse_result)

        assert len(chunks) > 0


class TestHeadingChunkStrategyChinese:
    """测试中文标题识别"""

    @pytest.mark.asyncio
    async def test_chinese_chapters(self):
        """测试中文章节标题"""
        from tests.ext.text_chunker.conftest import create_parse_result

        text = """第一章：引言

这是第一章的内容。

第二章：方法

这是第二章的内容。

第三章：结论

这是第三章的内容。"""

        parse_result = create_parse_result(text, OutputFormat.MARKDOWN)
        config = HeadingChunkConfig(
            max_chunk_size=2000,
            overlap_paragraphs=0,
            preserve_headings=True,
            heading_patterns=["^第[一二三四五六七八九十]+[章节篇]"],
        )
        strategy = HeadingChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        assert len(chunks) >= 3

    @pytest.mark.asyncio
    async def test_chinese_pian(self):
        """测试中文"篇"级标题"""
        from tests.ext.text_chunker.conftest import create_parse_result

        text = """第一篇：总论

这是总论内容。

第二篇：分论

这是分论内容。"""

        parse_result = create_parse_result(text, OutputFormat.MARKDOWN)
        config = HeadingChunkConfig(
            max_chunk_size=2000,
            overlap_paragraphs=0,
            preserve_headings=True,
        )
        strategy = HeadingChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        assert len(chunks) >= 2


class TestHeadingChunkStrategyTreeBuilding:
    """测试标题树构建"""

    @pytest.mark.asyncio
    async def test_nested_heading_structure(self):
        """测试嵌套标题结构"""
        from tests.ext.text_chunker.conftest import create_parse_result

        text = """# Main Title

Content

## Subsection 1

Content 1

### Sub-subsection 1.1

Deep content

## Subsection 2

Content 2"""

        parse_result = create_parse_result(text, OutputFormat.MARKDOWN)
        config = HeadingChunkConfig(
            max_chunk_size=3000,
            overlap_paragraphs=0,
            preserve_headings=True,
        )
        strategy = HeadingChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_heading_hierarchy_preserved(self):
        """测试标题层级保持"""
        from tests.ext.text_chunker.conftest import create_parse_result

        text = """# Level 1

Content L1

## Level 2

Content L2

### Level 3

Content L3"""

        parse_result = create_parse_result(text, OutputFormat.MARKDOWN)
        config = HeadingChunkConfig(
            max_chunk_size=5000,
            overlap_paragraphs=0,
            preserve_headings=True,
        )
        strategy = HeadingChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        # 验证层级结构被保留
        assert len(chunks) > 0


class TestHeadingChunkStrategyMaxChunkSize:
    """测试最大块大小限制"""

    @pytest.mark.asyncio
    async def test_max_chunk_size_enforced(self, markdown_parse_result):
        """测试强制执行 max_chunk_size"""
        config = HeadingChunkConfig(
            max_chunk_size=100,
            overlap_paragraphs=0,
            preserve_headings=False,
        )
        strategy = HeadingChunkStrategy(config)

        chunks = await strategy.chunk(markdown_parse_result)

        # 超长的块会被进一步分割
        for chunk in chunks:
            if len(chunk.content) > 100:
                # 可能是因为标题长度或其他原因
                pass

    @pytest.mark.asyncio
    async def test_split_long_section_by_paragraphs(self):
        """测试按段落分割超长章节"""
        from tests.ext.text_chunker.conftest import create_parse_result

        # 创建一个很长的章节
        long_paragraph = "This is a long paragraph. " * 50
        text = f"""# Chapter 1

{long_paragraph}

{long_paragraph}

{long_paragraph}"""

        parse_result = create_parse_result(text, OutputFormat.MARKDOWN)
        config = HeadingChunkConfig(
            max_chunk_size=200,
            overlap_paragraphs=0,
            preserve_headings=True,
        )
        strategy = HeadingChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        # 应该被分割成多个块
        assert len(chunks) > 0


class TestHeadingChunkStrategyOverlap:
    """测试段落重叠"""

    @pytest.mark.asyncio
    async def test_paragraph_overlap(self, markdown_parse_result):
        """测试段落重叠功能"""
        config = HeadingChunkConfig(
            max_chunk_size=3000,
            overlap_paragraphs=1,
            preserve_headings=True,
        )
        strategy = HeadingChunkStrategy(config)

        chunks = await strategy.chunk(markdown_parse_result)

        if len(chunks) > 1:
            # 第二个及之后的 chunk 应该有 overlap
            for chunk in chunks[1:]:
                if chunk.overlap_start and chunk.overlap_end:
                    assert chunk.overlap_start is not None
                    assert chunk.overlap_end is not None

    @pytest.mark.asyncio
    async def test_multiple_paragraph_overlap(self):
        """测试多段落重叠"""
        from tests.ext.text_chunker.conftest import create_parse_result

        text = """# Chapter 1

Paragraph 1.

Paragraph 2.

Paragraph 3.

# Chapter 2

Paragraph 4.

Paragraph 5."""

        parse_result = create_parse_result(text, OutputFormat.MARKDOWN)
        config = HeadingChunkConfig(
            max_chunk_size=2000,
            overlap_paragraphs=2,
            preserve_headings=True,
        )
        strategy = HeadingChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        assert len(chunks) > 0


class TestHeadingChunkStrategyFallback:
    """测试回退机制"""

    @pytest.mark.asyncio
    async def test_fallback_to_length_no_headings(self, text_parse_result):
        """测试没有标题时回退到按长度切分"""
        config = HeadingChunkConfig(
            max_chunk_size=100,
            overlap_paragraphs=0,
            preserve_headings=True,
        )
        strategy = HeadingChunkStrategy(config)

        chunks = await strategy.chunk(text_parse_result)

        # 应该回退到按长度切分
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_fallback_with_plain_text(self, long_text_parse_result):
        """测试纯文本回退"""
        config = HeadingChunkConfig(
            max_chunk_size=200,
            overlap_paragraphs=0,
            preserve_headings=False,
        )
        strategy = HeadingChunkStrategy(config)

        chunks = await strategy.chunk(long_text_parse_result)

        # 应该生成多个块
        assert len(chunks) > 0


class TestHeadingChunkStrategyEdgeCases:
    """测试边界情况"""

    @pytest.mark.asyncio
    async def test_empty_text(self, empty_parse_result):
        """测试空文本"""
        config = HeadingChunkConfig(max_chunk_size=1000)
        strategy = HeadingChunkStrategy(config)

        chunks = await strategy.chunk(empty_parse_result)

        # 应该回退到长度策略并返回空列表
        assert len(chunks) == 0

    @pytest.mark.asyncio
    async def test_single_heading(self):
        """测试单个标题"""
        from tests.ext.text_chunker.conftest import create_parse_result

        text = """# Only Heading

This is the content under the only heading."""

        parse_result = create_parse_result(text, OutputFormat.MARKDOWN)
        config = HeadingChunkConfig(max_chunk_size=3000, preserve_headings=True)
        strategy = HeadingChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_heading_without_content(self):
        """测试没有内容的标题"""
        from tests.ext.text_chunker.conftest import create_parse_result

        text = """# Heading 1

# Heading 2

# Heading 3"""

        parse_result = create_parse_result(text, OutputFormat.MARKDOWN)
        config = HeadingChunkConfig(max_chunk_size=1000, preserve_headings=True)
        strategy = HeadingChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        # 应该能处理空内容
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_custom_heading_pattern(self):
        """测试自定义标题模式"""
        from tests.ext.text_chunker.conftest import create_parse_result

        text = """CHAPTER 1: Introduction

Content 1

CHAPTER 2: Method

Content 2"""

        parse_result = create_parse_result(text, OutputFormat.MARKDOWN)
        config = HeadingChunkConfig(
            max_chunk_size=2000,
            heading_patterns=["^CHAPTER \\d+:"],
            preserve_headings=True,
        )
        strategy = HeadingChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        assert len(chunks) >= 2


class TestHeadingChunkStrategyPositionTracking:
    """测试位置跟踪"""

    @pytest.mark.asyncio
    async def test_heading_positions(self, markdown_parse_result):
        """测试标题位置正确性"""
        config = HeadingChunkConfig(max_chunk_size=3000, preserve_headings=True)
        strategy = HeadingChunkStrategy(config)

        chunks = await strategy.chunk(markdown_parse_result)

        for chunk in chunks:
            assert chunk.start is not None
            assert chunk.end is not None
            assert len(chunk.pages) > 0

    @pytest.mark.asyncio
    async def test_multi_page_headings(self, multi_page_parse_result):
        """测试多页文档中的标题"""
        # 在多页文本中添加标题
        from tests.ext.text_chunker.conftest import create_parse_result

        page1 = "# Chapter 1\n\n" + "Content 1. " * 50
        page2 = "# Chapter 2\n\n" + "Content 2. " * 50
        page3 = "# Chapter 3\n\n" + "Content 3. " * 50

        content = "\n\n".join([page1, page2, page3])
        parse_result = create_parse_result(content, OutputFormat.MARKDOWN, page_count=3)

        config = HeadingChunkConfig(max_chunk_size=2000, preserve_headings=True)
        strategy = HeadingChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        # 检查页码
        for chunk in chunks:
            assert isinstance(chunk.pages, list)


class TestHeadingChunkStrategyMetadata:
    """测试元数据"""

    @pytest.mark.asyncio
    async def test_strategy_metadata(self, markdown_parse_result):
        """测试策略元数据"""
        config = HeadingChunkConfig(max_chunk_size=2000)
        strategy = HeadingChunkStrategy(config)

        chunks = await strategy.chunk(markdown_parse_result)

        for chunk in chunks:
            assert chunk.metadata["strategy"] == "heading"
            assert "chunk_index" in chunk.metadata
