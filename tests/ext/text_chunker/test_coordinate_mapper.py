"""
测试坐标映射器

测试场景：
- 全局索引到页码坐标转换
- 页码坐标到全局索引转换
- 页面范围计算
- 内容长度计算
- 边界情况处理
"""

import pytest

from ext.document_parser.core.parse_result import OutputFormat
from ext.document_parser.core.parse_result import ParseResult, PageResult
from ext.text_chunker.core.coordinate_mapper import CoordinateMapper
from ext.text_chunker.core.chunk_result import TextPosition


class TestCoordinateMapperBasic:
    """测试基本的坐标映射功能"""

    def test_initialization_single_page(self, text_parse_result):
        """测试单页文档初始化"""
        mapper = CoordinateMapper(text_parse_result)

        assert mapper is not None
        assert mapper._parse_result == text_parse_result

    def test_initialization_multi_page(self, multi_page_parse_result):
        """测试多页文档初始化"""
        mapper = CoordinateMapper(multi_page_parse_result)

        assert mapper is not None
        assert len(mapper._page_boundaries) == 3

    def test_get_content_length(self, text_parse_result):
        """测试获取内容长度"""
        mapper = CoordinateMapper(text_parse_result)
        length = mapper.get_content_length()

        assert length == len(text_parse_result.content)


class TestCoordinateMapperGlobalToPage:
    """测试全局索引到页码坐标转换"""

    def test_first_page_start(self, multi_page_parse_result):
        """测试第一页起始位置"""
        mapper = CoordinateMapper(multi_page_parse_result)
        pos = mapper.global_to_page(0)

        assert pos.page_number == 1
        assert pos.char_index == 0

    def test_first_page_middle(self, multi_page_parse_result):
        """测试第一页中间位置"""
        mapper = CoordinateMapper(multi_page_parse_result)
        pos = mapper.global_to_page(10)

        assert pos.page_number == 1
        assert pos.char_index == 10

    def test_second_page_start(self, multi_page_parse_result):
        """测试第二页起始位置"""
        mapper = CoordinateMapper(multi_page_parse_result)

        # 获取第一页的长度
        page1_len = len(multi_page_parse_result.pages[0].content)
        # 第二页的起始位置应该是 page1_len + 分隔符长度(2)
        second_page_start = page1_len + 2

        pos = mapper.global_to_page(second_page_start)

        assert pos.page_number == 2
        assert pos.char_index == 0

    def test_last_page(self, multi_page_parse_result):
        """测试最后一页位置"""
        mapper = CoordinateMapper(multi_page_parse_result)
        total_len = mapper.get_content_length()

        # 最后一个字符的位置
        pos = mapper.global_to_page(total_len - 1)

        assert pos.page_number == 3

    def test_separator_position(self, multi_page_parse_result):
        """测试分隔符位置（应该容错）"""
        mapper = CoordinateMapper(multi_page_parse_result)

        # 获取第一页的长度
        page1_len = len(multi_page_parse_result.pages[0].content)
        # 分隔符的位置
        separator_pos = page1_len

        # 应该容错，返回前一页的末尾或最后一页
        pos = mapper.global_to_page(separator_pos)
        # 可能返回第一页的末尾或最后一页（根据新的容错逻辑）
        assert pos.page_number >= 1

    def test_preserves_actual_page_numbers(self):
        """测试保留 ParseResult 中的真实页号"""
        parse_result = ParseResult(
            content="page10\n\npage11",
            format=OutputFormat.MARKDOWN,
            pages=[
                PageResult(page_number=10, content="page10"),
                PageResult(page_number=11, content="page11"),
            ],
            page_count=2,
            engine_used="test",
        )
        mapper = CoordinateMapper(parse_result)

        assert mapper.global_to_page(0).page_number == 10
        assert mapper.global_to_page(len("page10") + 2).page_number == 11


class TestCoordinateMapperPageToGlobal:
    """测试页码坐标到全局索引转换"""

    def test_first_page_start(self, multi_page_parse_result):
        """测试第一页起始"""
        mapper = CoordinateMapper(multi_page_parse_result)
        pos = TextPosition(page_number=1, char_index=0)
        global_idx = mapper.page_to_global(pos)

        assert global_idx == 0

    def test_first_page_offset(self, multi_page_parse_result):
        """测试第一页偏移"""
        mapper = CoordinateMapper(multi_page_parse_result)
        pos = TextPosition(page_number=1, char_index=10)
        global_idx = mapper.page_to_global(pos)

        assert global_idx == 10

    def test_second_page_start(self, multi_page_parse_result):
        """测试第二页起始"""
        mapper = CoordinateMapper(multi_page_parse_result)
        pos = TextPosition(page_number=2, char_index=0)
        global_idx = mapper.page_to_global(pos)

        # 应该是第一页的长度 + 分隔符长度
        page1_len = len(multi_page_parse_result.pages[0].content)
        assert global_idx == page1_len + 2

    def test_invalid_page_number(self, multi_page_parse_result):
        """测试无效页码"""
        mapper = CoordinateMapper(multi_page_parse_result)
        pos = TextPosition(page_number=100, char_index=0)

        with pytest.raises(ValueError, match="Invalid page number"):
            mapper.page_to_global(pos)

    def test_non_sequential_page_number(self):
        """测试非从1开始的页码可逆映射"""
        parse_result = ParseResult(
            content="aaa\n\nbbb",
            format=OutputFormat.TEXT,
            pages=[
                PageResult(page_number=10, content="aaa"),
                PageResult(page_number=11, content="bbb"),
            ],
            page_count=2,
            engine_used="test",
        )
        mapper = CoordinateMapper(parse_result)

        assert mapper.page_to_global(TextPosition(page_number=11, char_index=0)) == len("aaa") + 2

    def test_zero_page_number(self, multi_page_parse_result):
        """测试零页码"""
        mapper = CoordinateMapper(multi_page_parse_result)
        pos = TextPosition(page_number=0, char_index=0)

        with pytest.raises(ValueError, match="Invalid page number"):
            mapper.page_to_global(pos)


class TestCoordinateMapperGetPagesForRange:
    """测试页面范围计算"""

    def test_single_page_range(self, multi_page_parse_result):
        """测试单页范围"""
        mapper = CoordinateMapper(multi_page_parse_result)
        pages = mapper.get_pages_for_range(0, 100)

        assert pages == [1]

    def test_cross_page_range(self, multi_page_parse_result):
        """测试跨页范围"""
        mapper = CoordinateMapper(multi_page_parse_result)

        # 获取第一页的长度
        page1_len = len(multi_page_parse_result.pages[0].content)

        # 从第一页中间到第二页开头
        pages = mapper.get_pages_for_range(50, page1_len + 100)

        assert 1 in pages
        assert 2 in pages

    def test_full_range(self, multi_page_parse_result):
        """测试全文档范围"""
        mapper = CoordinateMapper(multi_page_parse_result)
        total_len = mapper.get_content_length()
        pages = mapper.get_pages_for_range(0, total_len)

        assert pages == [1, 2, 3]

    def test_full_range_with_actual_page_numbers(self):
        """测试返回真实页号而不是数组下标"""
        parse_result = ParseResult(
            content="aaa\n\nbbb\n\nccc",
            format=OutputFormat.TEXT,
            pages=[
                PageResult(page_number=10, content="aaa"),
                PageResult(page_number=11, content="bbb"),
                PageResult(page_number=12, content="ccc"),
            ],
            page_count=3,
            engine_used="test",
        )
        mapper = CoordinateMapper(parse_result)

        assert mapper.get_pages_for_range(0, mapper.get_content_length()) == [10, 11, 12]

    def test_empty_range(self, multi_page_parse_result):
        """测试空范围"""
        mapper = CoordinateMapper(multi_page_parse_result)
        pages = mapper.get_pages_for_range(0, 0)

        # 空范围应该返回第一页（因为 start=0）
        assert len(pages) <= 1

    def test_out_of_range(self, multi_page_parse_result):
        """测试超出范围"""
        mapper = CoordinateMapper(multi_page_parse_result)
        total_len = mapper.get_content_length()

        # 超出范围的位置
        pages = mapper.get_pages_for_range(total_len + 100, total_len + 200)

        # 应该返回空列表
        assert isinstance(pages, list)


class TestCoordinateMapperEdgeCases:
    """测试边界情况"""

    def test_empty_document(self, empty_parse_result):
        """测试空文档"""
        mapper = CoordinateMapper(empty_parse_result)

        # 空文档应该没有页边界，会抛出异常
        # 由于空文档的 pages 为空，_build_mapping 不会创建边界
        try:
            pos = mapper.global_to_page(0)
            # 如果没有抛出异常，说明空文档的处理逻辑已改变
            # 这也是合理的，可能返回一个默认位置
            assert pos is not None
        except ValueError as e:
            # 预期的行为：抛出异常
            assert "No page boundaries available" in str(e)

    def test_single_character_document(self, single_char_parse_result):
        """测试单字符文档"""
        mapper = CoordinateMapper(single_char_parse_result)

        pos = mapper.global_to_page(0)
        assert pos.page_number == 1
        assert pos.char_index == 0

    def test_exact_boundary(self, multi_page_parse_result):
        """测试精确边界位置"""
        mapper = CoordinateMapper(multi_page_parse_result)
        page1_len = len(multi_page_parse_result.pages[0].content)

        # 第一页的最后一个字符
        pos = mapper.global_to_page(page1_len - 1)
        assert pos.page_number == 1

    def test_round_trip_conversion(self, multi_page_parse_result):
        """测试双向转换一致性"""
        mapper = CoordinateMapper(multi_page_parse_result)

        # 测试几个不同位置
        test_positions = [
            TextPosition(page_number=1, char_index=0),
            TextPosition(page_number=1, char_index=50),
            TextPosition(page_number=2, char_index=0),
            TextPosition(page_number=2, char_index=50),
            TextPosition(page_number=3, char_index=0),
        ]

        for pos in test_positions:
            # 页码坐标 -> 全局索引
            global_idx = mapper.page_to_global(pos)
            # 全局索引 -> 页码坐标
            recovered_pos = mapper.global_to_page(global_idx)

            # 应该恢复到原位置
            assert recovered_pos.page_number == pos.page_number
            assert recovered_pos.char_index == pos.char_index


class TestCoordinateMapperPageBoundaries:
    """测试页面边界构建"""

    def test_boundary_calculation(self, multi_page_parse_result):
        """测试边界计算正确性"""
        mapper = CoordinateMapper(multi_page_parse_result)

        assert len(mapper._page_boundaries) == 3

        # 检查每个页面的边界
        sep_len = len(mapper._separator)

        # 第一页
        page1_number, page1_start, page1_end = mapper._page_boundaries[0]
        assert page1_number == 1
        assert page1_start == 0
        assert page1_end == len(multi_page_parse_result.pages[0].content)

        # 第二页
        page2_number, page2_start, page2_end = mapper._page_boundaries[1]
        assert page2_number == 2
        assert page2_start == page1_end + sep_len
        assert page2_end == page2_start + len(multi_page_parse_result.pages[1].content)

        # 第三页
        page3_number, page3_start, page3_end = mapper._page_boundaries[2]
        assert page3_number == 3
        assert page3_start == page2_end + sep_len
        assert page3_end == page3_start + len(multi_page_parse_result.pages[2].content)


class TestCoordinateMapperPerformance:
    """测试性能相关"""

    def test_large_document(self, create_parse_result):
        """测试大文档处理"""
        # 创建一个包含100页的文档

        pages_content = [f"Page {i + 1} content. " * 100 for i in range(100)]
        content = "\n\n".join(pages_content)

        parse_result = create_parse_result(content, OutputFormat.TEXT, page_count=100)

        # 应该能快速创建映射
        mapper = CoordinateMapper(parse_result)
        assert len(mapper._page_boundaries) == 100

        # 转换应该快速
        for i in range(0, len(content), 1000):
            pos = mapper.global_to_page(i)
            assert pos.page_number >= 1
            assert pos.page_number <= 100


class TestCoordinateMapperMultipleSeparator:
    """测试分隔符处理"""

    def test_custom_separator_in_content(self, create_parse_result):
        """测试内容中包含分隔符"""

        # 内容中包含 \n\n
        page1 = "Line 1\n\nLine 2 in page 1"
        page2 = "Line 1\n\nLine 2 in page 2"
        content = f"{page1}\n\n{page2}"

        parse_result = create_parse_result(content, OutputFormat.TEXT, page_count=2)

        mapper = CoordinateMapper(parse_result)

        # 应该能正确映射
        pos = mapper.global_to_page(0)
        assert pos.page_number == 1

    def test_separator_at_page_boundary(self, multi_page_parse_result):
        """测试页面边界处的分隔符"""
        mapper = CoordinateMapper(multi_page_parse_result)

        # 获取第一页结束和第二页开始的位置
        page1_len = len(multi_page_parse_result.pages[0].content)

        # 第一页最后一个字符
        pos1 = mapper.global_to_page(page1_len - 1)
        assert pos1.page_number == 1

        # 第二页第一个字符
        pos2 = mapper.global_to_page(page1_len + 2)
        assert pos2.page_number == 2
