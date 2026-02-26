from __future__ import annotations

import trafilatura

from ext.document_parser.core.engine_base import BaseEngine
from ext.document_parser.core.parse_result import OutputFormat, ParseResult, PageResult


class TrafilaturaEngine(BaseEngine):
    engine_name = "trafilatura"
    supported_formats = [".html", ".htm"]

    async def parse(self, file_path: str, options: dict | None = None) -> ParseResult:
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        text = trafilatura.extract(
            html_content,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )

        pages_result = [
            PageResult(
                page_number=1,
                content=text or "",
                tables=[],
                images=[],
                metadata={"source": "trafilatura"},
            )
        ]

        return ParseResult(
            content=text or "",
            format=OutputFormat.TEXT,
            pages=pages_result,
            page_count=1,
            metadata={"source": "trafilatura"},
            confidence=0.90,
            engine_used="trafilatura",
        )


class MarkdownEngine(BaseEngine):
    engine_name = "markdown"
    supported_formats = [".md", ".markdown"]

    async def parse(self, file_path: str, options: dict | None = None) -> ParseResult:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        pages_result = [
            PageResult(
                page_number=1,
                content=content,
                tables=[],
                images=[],
                metadata={},
            )
        ]

        result = ParseResult(
            content=content,
            format=OutputFormat.MARKDOWN,
            pages=pages_result,
            page_count=1,
            metadata={"source": "markdown"},
            confidence=1.0,
            engine_used="markdown",
        )

        return result
