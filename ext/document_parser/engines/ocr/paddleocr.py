from __future__ import annotations

import io
from pathlib import Path

from paddleocr import PaddleOCR
from pdf2image import convert_from_path

from ext.document_parser.core.engine_base import BaseEngine
from ext.document_parser.core.parse_result import PageResult, ParseResult, OutputFormat


class PaddleOCREngine(BaseEngine):
    engine_name = "paddleocr"
    supported_formats = [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]

    def __init__(self) -> None:
        self.ocr = None

    async def parse(self, file_path: str, options: dict | None = None) -> ParseResult:
        if self.ocr is None:
            self.ocr = PaddleOCR(use_angle_cls=True, lang="ch")

        ext = Path(file_path).suffix.lower()

        if ext == ".pdf":
            return await self._parse_pdf(file_path)
        return await self._parse_image(file_path)

    async def _parse_pdf(self, file_path: str) -> ParseResult:
        images = convert_from_path(file_path)
        all_text = []
        pages_result = []
        total_confidence = []

        for page_num, image in enumerate(images):
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="PNG")
            img_bytes = img_byte_arr.getvalue()

            result = self.ocr.ocr(img_bytes)

            text_lines = []
            page_confidence = []

            if result and result[0]:
                for line in result[0]:
                    if line:
                        bbox, (text, confidence) = line
                        text_lines.append(text)
                        page_confidence.append(confidence)

            page_text = "\n".join(text_lines)
            all_text.append(page_text)
            total_confidence.extend(page_confidence)

            pages_result.append(
                PageResult(
                    page_number=page_num + 1,
                    content=page_text,
                    tables=[],
                    images=[],
                ),
            )

        avg_confidence = sum(total_confidence) / len(total_confidence) if total_confidence else 0.75

        return ParseResult(
            content="\n\n".join(all_text),
            format=OutputFormat.TEXT,
            pages=pages_result,
            page_count=len(images),
            metadata={"avg_ocr_confidence": avg_confidence},
            confidence=avg_confidence,
            engine_used="paddleocr",
        )

    async def _parse_image(self, file_path: str) -> ParseResult:
        result = self.ocr.ocr(file_path)

        text_lines = []
        confidence_scores = []

        if result and result[0]:
            for line in result[0]:
                if line:
                    bbox, (text, confidence) = line
                    text_lines.append(text)
                    confidence_scores.append(confidence)

        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.75

        pages_result = [
            PageResult(
                page_number=1,
                content="\n".join(text_lines),
                tables=[],
                images=[],
                metadata={"avg_ocr_confidence": avg_confidence},
            ),
        ]

        return ParseResult(
            content="\n".join(text_lines),
            format=OutputFormat.TEXT,
            pages=pages_result,
            page_count=1,
            metadata={"avg_ocr_confidence": avg_confidence},
            confidence=avg_confidence,
            engine_used="paddleocr",
        )
