from __future__ import annotations

import re
from pathlib import Path

from markitdown import MarkItDown
from markitdown import __version__ as markitdown_version
from ext.document_parser.core.engine_base import BaseEngine
from ext.document_parser.core.parse_result import OutputFormat, ParseResult, PageResult


class MarkitdownEngine(BaseEngine):
    engine_name = "markitdown"
    supported_formats = [
        # Office documents
        ".docx",
        ".pptx",
        ".xlsx",
        ".doc",
        ".ppt",
        ".xls",
        # PDF
        ".pdf",
        # Images (with OCR support)
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".tiff",
        # Audio (transcription)
        ".wav",
        ".mp3",
        ".m4a",
        # Web and text
        ".html",
        ".htm",
        ".txt",
        ".md",
        ".rst",
        ".xml",
        # Archives
        ".zip",
        ".epub",
    ]

    async def parse(self, file_path: str, options: dict | None = None) -> ParseResult:
        md = MarkItDown(enable_plugins=False)
        result = md.convert(str(file_path))

        content = result.text_content if hasattr(result, "text_content") else str(result)
        title = getattr(result, "title", None)

        # Try to detect pagination
        pages = self._detect_pages(content)
        page_count = len(pages)

        metadata = {
            "title": title if title else Path(file_path).name,
            "file_path": str(file_path),
            "file_name": Path(file_path).name,
        }

        parse_metadata = {
            "pagination_method": "detected" if page_count > 1 else "single_page",
            "page_markers_found": page_count > 1,
            "engine": "amarkitdown",
        }

        # Get version if available
        try:
            parse_metadata["engine_version"] = markitdown_version
        except (ImportError, AttributeError):
            parse_metadata["engine_version"] = "unknown"

        return ParseResult(
            content=content,
            format=OutputFormat.MARKDOWN,
            pages=pages,
            page_count=page_count,
            metadata=metadata,
            parse_metadata=parse_metadata,
            confidence=0.85,
            engine_used=self.engine_name,
        )

    def _detect_pages(self, content: str) -> list[PageResult]:
        """Detect page boundaries in markdown content."""
        page_patterns = [
            r"\n---+\s*Page\s+(\d+)\s*---+\n",
            r"\n\n#+\s*Page\s+(\d+)\s*\n",
            r"\n\n\s*Page\s+(\d+)\s*\n\n",
            r"\n-{3,}\s*\n\s*Page\s+(\d+)\s*\n-{3,}\n",
            r"\n={3,}\s*\n\s*Page\s+(\d+)\s*\n={3,}\n",
        ]

        # Try to detect page markers
        for pattern in page_patterns:
            matches = list(re.finditer(pattern, content, re.IGNORECASE))
            if len(matches) > 1:
                return self._split_by_markers(content, matches)

        # If no clear page markers detected, return single page
        return [
            PageResult(
                page_number=1,
                content=content,
                tables=[],
                images=[],
                metadata={"note": "No page markers detected, returning as single page"},
            ),
        ]

    def _split_by_markers(self, content: str, matches: list[re.Match]) -> list[PageResult]:
        """Split content into pages based on detected markers."""
        pages = []
        prev_end = 0

        for i, match in enumerate(matches):
            start = match.start()
            page_content = content[prev_end:start].strip()

            if page_content:
                pages.append(
                    PageResult(
                        page_number=i + 1,
                        content=page_content,
                        tables=[],
                        images=[],
                        metadata={"page_marker": match.group(0)},
                    ),
                )

            prev_end = match.end()

        # Add remaining content as last page
        if prev_end < len(content):
            remaining_content = content[prev_end:].strip()
            if remaining_content:
                pages.append(
                    PageResult(
                        page_number=len(matches) + 1,
                        content=remaining_content,
                        tables=[],
                        images=[],
                        metadata={"note": "Final page"},
                    ),
                )

        # If splitting failed, return single page
        if not pages:
            return [
                PageResult(
                    page_number=1,
                    content=content,
                    tables=[],
                    images=[],
                    metadata={"note": "Page splitting failed, returning as single page"},
                ),
            ]

        return pages
