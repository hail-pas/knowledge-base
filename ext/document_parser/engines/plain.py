from __future__ import annotations

from pathlib import Path

import aiofiles

from ext.document_parser.core.engine_base import BaseEngine
from ext.document_parser.core.parse_result import PageResult, ParseResult, OutputFormat


class TextEngine(BaseEngine):
    engine_name = "text"
    supported_formats = [".txt"]

    async def parse(self, file_path: str, options: dict | None = None) -> ParseResult:
        async with aiofiles.open(file_path, encoding="utf-8") as f:
            content = await f.read()

        file_path_obj = Path(file_path)
        file_size = file_path_obj.stat().st_size

        pages_result = [
            PageResult(
                page_number=1,
                content=content,
                tables=[],
                images=[],
                metadata={},
            ),
        ]

        return ParseResult(
            content=content,
            format=OutputFormat.TEXT,
            pages=pages_result,
            page_count=1,
            metadata={
                "file_name": file_path_obj.name,
                "file_size": file_size,
            },
            confidence=1.0,
            engine_used="text",
        )
