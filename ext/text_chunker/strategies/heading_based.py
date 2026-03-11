"""
按标题层级切块策略

按照正文区间切块，并在 chunk 内容中补上当前标题及其祖先标题。
"""

import re
from dataclasses import dataclass, field
from loguru import logger

from ext.document_parser.core.parse_result import ParseResult
from ext.text_chunker.config.strategy_config import HeadingChunkConfig
from ext.text_chunker.strategies.base import BaseChunkStrategy
from ext.text_chunker.core.coordinate_mapper import CoordinateMapper


@dataclass
class Heading:
    """标题信息"""

    level: int
    title: str
    start: int
    line_end: int
    parent: "Heading | None" = None
    children: list["Heading"] = field(default_factory=list)


@dataclass
class HeadingBodyChunk:
    """正文切块信息"""

    content: str
    body_start: int
    body_end: int
    overlap_start: int | None = None
    overlap_end: int | None = None


class HeadingChunkStrategy(BaseChunkStrategy[HeadingChunkConfig]):
    """按标题层级切块策略"""

    async def chunk(self, parse_result: ParseResult) -> list:

        self._set_mapper(CoordinateMapper(parse_result))

        text = parse_result.content
        headings = self._parse_headings(text)

        if not headings:
            logger.warning("No headings found, falling back to length-based chunking")
            return await self._fallback_to_length(parse_result)

        self._build_heading_tree(headings)

        results = []
        chunk_index = 0

        for chunk_data in self._split_by_headings(text, headings):
            chunk = self._build_chunk(
                content=chunk_data.content,
                global_start=chunk_data.body_start,
                global_end=chunk_data.body_end,
                overlap_start=chunk_data.overlap_start,
                overlap_end=chunk_data.overlap_end,
                metadata={"chunk_index": chunk_index, "strategy": "heading"},
            )
            results.append(chunk)
            chunk_index += 1

        logger.info(f"Chunked text into {len(results)} chunks using heading strategy")
        return results

    def _parse_headings(self, text: str) -> list[Heading]:
        headings: list[Heading] = []

        for match in re.finditer(r"(?m)^[^\n]*(?:\n|$)", text):
            line = match.group(0).rstrip("\n")
            stripped_line = line.strip()
            if not stripped_line:
                continue

            level = self._detect_heading_level(stripped_line)
            if level is None:
                continue

            headings.append(
                Heading(
                    level=level,
                    title=stripped_line,
                    start=match.start(),
                    line_end=match.start() + len(line),
                )
            )

        logger.debug(f"Parsed {len(headings)} headings")
        return headings

    def _detect_heading_level(self, line: str) -> int | None:
        for pattern in self.config.heading_patterns:
            if not re.search(pattern, line):
                continue

            markdown_match = re.match(r"^(#{1,6})\s+", line)
            if markdown_match:
                return len(markdown_match.group(1))

            if "篇" in line:
                return 1
            if "章" in line:
                return 2
            if "节" in line:
                return 3

            return 1

        return None

    def _build_heading_tree(self, headings: list[Heading]) -> None:
        stack: list[Heading] = []

        for heading in headings:
            while stack and stack[-1].level >= heading.level:
                stack.pop()

            if stack:
                heading.parent = stack[-1]
                stack[-1].children.append(heading)

            stack.append(heading)

    def _split_by_headings(self, text: str, headings: list[Heading]) -> list[HeadingBodyChunk]:
        chunks: list[HeadingBodyChunk] = []

        leading_chunks = self._build_prefixed_chunks(text, 0, headings[0].start, [])
        chunks.extend(leading_chunks)

        for index, heading in enumerate(headings):
            next_heading_start = headings[index + 1].start if index + 1 < len(headings) else len(text)
            section_end = heading.children[0].start if heading.children else next_heading_start
            heading_path = self._get_heading_path(heading)
            chunks.extend(self._build_prefixed_chunks(text, heading.line_end, section_end, heading_path))

        return chunks

    def _build_prefixed_chunks(
        self,
        text: str,
        range_start: int,
        range_end: int,
        heading_path: list[str],
    ) -> list[HeadingBodyChunk]:
        body_start = self._skip_leading_whitespace(text, range_start, range_end)
        body_end = self._trim_trailing_whitespace(text, body_start, range_end)

        if body_start >= body_end:
            return []

        prefix = ""
        if self.config.preserve_headings and heading_path:
            prefix = "\n".join(heading_path) + "\n\n"

        body_budget = self.config.max_chunk_size - len(prefix) if self.config.max_chunk_size > len(prefix) else 1
        body_ranges = self._split_body_range(text, body_start, body_end, body_budget, self.config.overlap_paragraphs)

        return [
            HeadingBodyChunk(
                content=prefix + text[chunk_start:chunk_end],
                body_start=chunk_start,
                body_end=chunk_end,
                overlap_start=overlap_start,
                overlap_end=overlap_end,
            )
            for chunk_start, chunk_end, overlap_start, overlap_end in body_ranges
        ]

    def _split_body_range(
        self,
        text: str,
        start: int,
        end: int,
        max_size: int,
        overlap_paragraphs: int,
    ) -> list[tuple[int, int, int | None, int | None]]:
        atoms = self._build_body_atoms(text, start, end, max_size)
        if not atoms:
            return []

        results: list[tuple[int, int, int | None, int | None]] = []
        chunk_start_idx = 0

        while chunk_start_idx < len(atoms):
            chunk_end_idx = chunk_start_idx
            chunk_start = atoms[chunk_start_idx][0]
            chunk_end = atoms[chunk_start_idx][1]

            while chunk_end_idx + 1 < len(atoms):
                candidate_end = atoms[chunk_end_idx + 1][1]
                if candidate_end - chunk_start > max_size:
                    break
                chunk_end_idx += 1
                chunk_end = candidate_end

            overlap_start = None
            overlap_end = None
            if chunk_start_idx > 0 and overlap_paragraphs > 0:
                overlap_count = min(overlap_paragraphs, chunk_end_idx - chunk_start_idx + 1)
                overlap_start = atoms[chunk_start_idx][0]
                overlap_end = atoms[chunk_start_idx + overlap_count - 1][1]

            results.append((chunk_start, chunk_end, overlap_start, overlap_end))

            if chunk_end_idx == len(atoms) - 1:
                break

            next_start_idx = max(chunk_end_idx - overlap_paragraphs + 1, chunk_start_idx + 1)
            chunk_start_idx = next_start_idx

        return results

    def _build_body_atoms(self, text: str, start: int, end: int, max_size: int) -> list[tuple[int, int]]:
        paragraphs = self._extract_paragraph_ranges(text, start, end)
        atoms: list[tuple[int, int]] = []

        for para_start, para_end in paragraphs:
            if para_end - para_start <= max_size:
                atoms.append((para_start, para_end))
                continue

            split_start = para_start
            while split_start < para_end:
                split_end = min(split_start + max_size, para_end)
                atoms.append((split_start, split_end))
                split_start = split_end

        return atoms

    def _extract_paragraph_ranges(self, text: str, start: int, end: int) -> list[tuple[int, int]]:
        paragraph_ranges: list[tuple[int, int]] = []
        cursor = start

        while cursor < end:
            separator = re.search(r"\n\s*\n", text[cursor:end])
            paragraph_end = cursor + separator.start() if separator else end

            trimmed_start = self._skip_leading_whitespace(text, cursor, paragraph_end)
            trimmed_end = self._trim_trailing_whitespace(text, trimmed_start, paragraph_end)

            if trimmed_start < trimmed_end:
                paragraph_ranges.append((trimmed_start, trimmed_end))

            if not separator:
                break

            cursor = cursor + separator.end()

        return paragraph_ranges

    def _skip_leading_whitespace(self, text: str, start: int, end: int) -> int:
        while start < end and text[start].isspace():
            start += 1
        return start

    def _trim_trailing_whitespace(self, text: str, start: int, end: int) -> int:
        while end > start and text[end - 1].isspace():
            end -= 1
        return end

    def _get_heading_path(self, heading: Heading) -> list[str]:
        titles = []
        current: Heading | None = heading

        while current is not None:
            titles.append(current.title)
            current = current.parent

        return list(reversed(titles))

    async def _fallback_to_length(self, parse_result: ParseResult) -> list:
        from ext.text_chunker.config.strategy_config import LengthChunkConfig
        from ext.text_chunker.strategies.length_based import LengthChunkStrategy

        logger.info("Falling back to length-based chunking")

        config = LengthChunkConfig(chunk_size=self.config.max_chunk_size, overlap=0)
        strategy = LengthChunkStrategy(config)
        return await strategy.chunk(parse_result)
