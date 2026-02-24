"""
按标题层级切块策略

按照文档标题层级进行切块，保留父级标题
"""

import re
from dataclasses import dataclass, field
from loguru import logger
from typing import Any

from ext.document_parser.core.parse_result import ParseResult
from ext.text_chunker.config.strategy_config import HeadingChunkConfig
from ext.text_chunker.strategies.base import BaseChunkStrategy


@dataclass
class Heading:
    """标题信息"""

    level: int  # 标题层级（1-6）
    title: str  # 标题文本
    position: int  # 在文档中的全局位置
    parent: "Heading | None" = None  # 父标题
    children: list["Heading"] = field(default_factory=list)  # 子标题


class HeadingChunkStrategy(BaseChunkStrategy[HeadingChunkConfig]):
    """按标题层级切块策略

    根据文档标题层级进行切块，保留所有父级标题，
    超长时按段落分割
    """

    async def chunk(self, parse_result: ParseResult) -> list:
        """
        执行按标题层级切块

        Args:
            parse_result: 文档解析结果

        Returns:
            切块结果列表
        """
        from ext.text_chunker.core.coordinate_mapper import CoordinateMapper

        self._set_mapper(CoordinateMapper(parse_result))

        text = parse_result.content

        # 1. 解析标题结构
        headings = self._parse_headings(text)

        if not headings:
            logger.warning("No headings found, falling back to length-based chunking")
            # 回退到按长度切分
            return await self._fallback_to_length(parse_result)

        # 2. 构建标题树
        heading_tree = self._build_heading_tree(headings)

        # 3. 按标题切分内容
        chunks_data = self._split_by_headings(text, heading_tree)

        # 4. 构建ChunkResult列表
        results = []
        chunk_index = 0

        for chunk_text, start_pos, end_pos in chunks_data:
            overlap_start = None
            overlap_end = None

            # 计算overlap（基于段落数）
            if self.config.overlap_paragraphs > 0 and chunk_index > 0:
                overlap_end = start_pos
                overlap_start = self._calculate_overlap_start(text, start_pos, self.config.overlap_paragraphs)

            chunk = self._build_chunk(
                content=chunk_text,
                global_start=start_pos,
                global_end=end_pos,
                overlap_start=overlap_start,
                overlap_end=overlap_end,
                metadata={"chunk_index": chunk_index, "strategy": "heading"},
            )
            results.append(chunk)
            chunk_index += 1

        logger.info(f"Chunked text into {len(results)} chunks using heading strategy")
        return results

    def _parse_headings(self, text: str) -> list[Heading]:
        """
        解析文档中的所有标题

        Args:
            text: 文档文本

        Returns:
            标题列表（按出现顺序）
        """
        headings = []
        heading_patterns = self.config.heading_patterns

        for pattern in heading_patterns:
            # 编译正则
            regex = re.compile(pattern, re.MULTILINE)

            for match in regex.finditer(text):
                title = match.group(0).strip()

                # 判断标题层级
                level = 1
                if pattern.startswith("^#"):
                    # Markdown标题
                    level = len(match.group(1)) if match.groups() else title.count("#")
                elif "章节" in title or "篇" in title:
                    # 中文章节
                    if "篇" in title:
                        level = 1
                    elif "章" in title:
                        level = 2
                    else:
                        level = 3

                headings.append(Heading(level=level, title=title, position=match.start(), parent=None))

        # 按位置排序
        headings.sort(key=lambda h: h.position)

        logger.debug(f"Parsed {len(headings)} headings")
        return headings

    def _build_heading_tree(self, headings: list[Heading]) -> list[Heading]:
        """
        构建标题层级树

        Args:
            headings: 标题列表

        Returns:
            根级标题列表
        """
        if not headings:
            return []

        root_headings: list[Heading] = []
        stack: list[Heading] = []

        for heading in headings:
            # 弹出栈中比当前标题层级高或相等的标题
            while stack and stack[-1].level >= heading.level:
                stack.pop()

            # 设置父标题
            if stack:
                heading.parent = stack[-1]
                stack[-1].children.append(heading)
            else:
                root_headings.append(heading)

            # 入栈
            stack.append(heading)

        logger.debug(f"Built heading tree with {len(root_headings)} root headings")
        return root_headings

    def _split_by_headings(self, text: str, heading_tree: list[Heading]) -> list[tuple[str, int, int]]:
        """
        按标题切分文本

        Args:
            text: 文档文本
            heading_tree: 标题树

        Returns:
            (chunk_text, start_pos, end_pos) 列表
        """
        chunks = []

        def process_heading(heading: Heading, parent_titles: list[str]):
            """处理单个标题及其内容"""
            # 收集所有父级标题
            all_titles = parent_titles + [heading.title]

            # 确定内容范围
            start_pos = heading.position + len(heading.title)

            # 找到结束位置（下一个同级或更高级标题，或文档结尾）
            end_pos = len(text)

            # 在同级标题中找下一个
            if heading.parent:
                siblings = heading.parent.children
                idx = siblings.index(heading)
                if idx < len(siblings) - 1:
                    end_pos = siblings[idx + 1].position
            else:
                # 根级标题，在根级列表中找下一个
                # 这里简化处理，假设heading_tree是排序的
                pass

            # 提取内容
            content = text[start_pos:end_pos].strip()

            # 添加父级标题（如果配置要求）
            if self.config.preserve_headings and all_titles:
                heading_text = "\n".join(all_titles) + "\n\n"
                content = heading_text + content

            # 检查是否超长
            if len(content) > self.config.max_chunk_size:
                # 超长，按段落分割
                sub_chunks = self._split_by_paragraphs(content, self.config.max_chunk_size)

                # 计算每个子块的全局位置
                global_offset = heading.position
                for sub_chunk in sub_chunks:
                    chunk_start = global_offset
                    global_offset += len(sub_chunk)
                    chunk_end = global_offset
                    chunks.append((sub_chunk, chunk_start, chunk_end))
            else:
                chunks.append((content, heading.position, end_pos))

            # 递归处理子标题
            for child in heading.children:
                process_heading(child, all_titles)

        # 处理所有根级标题
        for root_heading in heading_tree:
            process_heading(root_heading, [])

        return chunks

    def _split_by_paragraphs(self, text: str, max_size: int) -> list[str]:
        """
        按段落切分文本

        Args:
            text: 输入文本
            max_size: 最大切块大小

        Returns:
            切块列表
        """
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            test_chunk = current_chunk + ("\n\n" if current_chunk else "") + para

            if len(test_chunk) <= max_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _calculate_overlap_start(self, text: str, chunk_start: int, num_paragraphs: int) -> int:
        """
        计算overlap起始位置（向前回溯N个段落）

        Args:
            text: 文档文本
            chunk_start: 当前chunk起始位置
            num_paragraphs: 回溯段落数

        Returns:
            overlap起始位置
        """
        if num_paragraphs <= 0:
            return chunk_start

        # 向前查找段落分隔符
        paragraph_count = 0
        pos = chunk_start - 1

        while pos >= 0 and paragraph_count < num_paragraphs:
            if text[pos : pos + 2] == "\n\n":
                paragraph_count += 1
            pos -= 1

        return max(pos + 2, 0)  # +2 跳过 \n\n

    async def _fallback_to_length(self, parse_result: ParseResult) -> list:
        """
        回退到按长度切分

        Args:
            parse_result: 文档解析结果

        Returns:
            切块结果列表
        """
        from ext.text_chunker.strategies.length_based import LengthChunkStrategy
        from ext.text_chunker.config.strategy_config import LengthChunkConfig

        logger.info("Falling back to length-based chunking")

        config = LengthChunkConfig(chunk_size=self.config.max_chunk_size, overlap=0)
        strategy = LengthChunkStrategy(config)
        return await strategy.chunk(parse_result)
