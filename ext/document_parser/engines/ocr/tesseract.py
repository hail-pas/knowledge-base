from __future__ import annotations

from PIL import Image
import pytesseract

from ext.document_parser.core.engine_base import BaseEngine
from ext.document_parser.core.parse_result import OutputFormat, ParseResult, PageResult


class TesseractOCREngine(BaseEngine):
    engine_name = "tesseract"
    supported_formats = [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]

    async def parse(self, file_path: str, options: dict | None = None) -> ParseResult:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)

        pages_result = [
            PageResult(
                page_number=1,
                content=text,
                tables=[],
                images=[],
                metadata={"engine": "tesseract"},
            )
        ]

        return ParseResult(
            content=text,
            format=OutputFormat.TEXT,
            pages=pages_result,
            page_count=1,
            metadata={"engine": "tesseract"},
            confidence=0.75,
            engine_used="tesseract",
        )
