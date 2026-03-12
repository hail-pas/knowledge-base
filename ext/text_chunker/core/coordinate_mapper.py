"""
坐标映射器

用于在全局索引和页码坐标之间进行转换
"""

from loguru import logger

from constant.symbol import PAGE_SEPARATOR
from ext.text_chunker.core.chunk_result import TextPosition
from ext.document_parser.core.parse_result import ParseResult


class CoordinateMapper:
    """全局索引 ↔ 页码坐标映射器

    用于将 ParseResult.content（完整拼接文本）中的全局索引
    转换为具体的页码和页内偏移量

    假设 content 是通过 "\\n\\n" 分隔符拼接各页面内容
    Example:
        page1: "abc" (len=3)
        page2: "def" (len=3)
        content = "abc\\n\\ndef"

        _page_boundaries = [(0, 3), (5, 8)]
    """

    def __init__(self, parse_result: ParseResult) -> None:
        """
        初始化坐标映射器

        Args:
            parse_result: 文档解析结果
        """
        self._parse_result = parse_result
        self._page_boundaries: list[tuple[int, int, int]] = []
        self._separator = PAGE_SEPARATOR
        self._build_mapping()

    def _build_mapping(self) -> None:
        """构建页码边界映射表

        计算每个页面的内容在完整content中的全局索引范围
        """
        if not self._parse_result.pages:
            logger.warning("ParseResult has no pages, mapping will be empty")
            return

        sep_len = len(self._separator)
        global_pos = 0

        for page in self._parse_result.pages:
            page_len = len(page.content)
            self._page_boundaries.append((page.page_number, global_pos, global_pos + page_len))
            global_pos += page_len + sep_len

    def global_to_page(self, global_index: int) -> TextPosition:
        """
        全局索引 → 页码坐标

        Args:
            global_index: 完整content中的字符索引

        Returns:
            TextPosition: 页码和页内偏移

        Raises:
            ValueError: 全局索引超出有效范围
        """
        if not self._page_boundaries:
            raise ValueError("No page boundaries available")

        # 如果索引为负数，设为0
        if global_index < 0:
            global_index = 0

        # 如果索引超出范围，取最后一个位置
        last_page_num, _, last_end = self._page_boundaries[-1]
        if global_index >= last_end:
            # 尝试容错：可能在分隔符上，返回前一页的末尾
            if global_index < last_end + len(self._separator):
                last_page_len = last_end - self._page_boundaries[-1][1]
                return TextPosition(page_number=last_page_num, char_index=last_page_len)
            # 超出范围太多，返回最后一页的末尾
            last_page_len = last_end - self._page_boundaries[-1][1]
            return TextPosition(page_number=last_page_num, char_index=last_page_len)

        for page_number, start, end in self._page_boundaries:
            if start <= global_index < end:
                char_index = global_index - start
                return TextPosition(page_number=page_number, char_index=char_index)

        # 如果都没匹配到，返回最后一页末尾（兜底）
        last_page_num, last_start, last_end = self._page_boundaries[-1]
        last_page_len = last_end - last_start
        return TextPosition(page_number=last_page_num, char_index=last_page_len)

    def page_to_global(self, position: TextPosition) -> int:
        """
        页码坐标 → 全局索引

        Args:
            position: 页码坐标

        Returns:
            全局索引

        Raises:
            ValueError: 页码超出有效范围
        """
        for page_number, start, end in self._page_boundaries:
            if page_number != position.page_number:
                continue
            page_len = end - start
            if position.char_index < 0 or position.char_index > page_len:
                raise ValueError(
                    f"Invalid char index {position.char_index} for page {position.page_number} with length {page_len}",
                )
            return start + position.char_index

        raise ValueError(f"Invalid page number: {position.page_number}")

    def get_pages_for_range(self, start: int, end: int) -> list[int]:
        """
        获取全局索引范围跨越的所有页码

        Args:
            start: 起始全局索引（包含）
            end: 结束全局索引（不包含）

        Returns:
            排序后的页码列表
        """
        if not self._page_boundaries:
            return []

        pages = set()

        try:
            start_pos = self.global_to_page(start)
            # end 是不包含的，所以要减1
            end_pos = self.global_to_page(max(end - 1, 0))

            started = False
            for page_number, _, _ in self._page_boundaries:
                if page_number == start_pos.page_number:
                    started = True
                if started:
                    pages.add(page_number)
                if page_number == end_pos.page_number:
                    break

        except ValueError:
            logger.warning(f"Failed to map range [{start}, {end}) to pages")

        return sorted(pages)

    def get_content_length(self) -> int:
        """
        获取完整content的长度

        Returns:
            字符数
        """
        if not self._page_boundaries:
            return len(self._parse_result.content)

        _, _, last_end = self._page_boundaries[-1]
        sep_count = len(self._page_boundaries) - 1
        sep_len = len(self._separator)
        return last_end + sep_count * sep_len
