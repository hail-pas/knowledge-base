from __future__ import annotations

import os
from pathlib import Path

import numpy as np
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
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            self.ocr = PaddleOCR(use_angle_cls=True, lang="ch")

        ext = Path(file_path).suffix.lower()

        if ext == ".pdf":
            return await self._parse_pdf(file_path)
        return await self._parse_image(file_path)

    async def _parse_pdf(self, file_path: str) -> ParseResult:
        images = convert_from_path(file_path, dpi=100)
        all_text = []
        pages_result = []
        total_confidence = []

        for page_num, image in enumerate(images):
            image_array = np.array(image)
            result = self.ocr.ocr(image_array)
            text_lines, page_confidence = self._extract_lines_from_result(result)

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
        text_lines, confidence_scores = self._extract_lines_from_result(result)

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

    def _extract_lines_from_result(self, result: object) -> tuple[list[str], list[float]]:
        text_lines: list[str] = []
        confidence_scores: list[float] = []

        if not result:
            return text_lines, confidence_scores

        # PaddleOCR 3.x returns a list of dict items with rec_texts/rec_scores.
        if isinstance(result, list) and result and isinstance(result[0], dict):
            for item in result:
                texts = item.get("rec_texts") or []
                scores = item.get("rec_scores") or []
                for text, score in zip(texts, scores, strict=False):
                    if text:
                        text_lines.append(str(text))
                        confidence_scores.append(float(score))
            return text_lines, confidence_scores

        # PaddleOCR 2.x returns nested tuples: [ [ [bbox, (text, confidence)], ... ] ]
        if isinstance(result, list) and result:
            first_page = result[0]
            if isinstance(first_page, list):
                for line in first_page:
                    if not line or len(line) < 2:
                        continue
                    _, text_confidence = line[0], line[1]
                    if not isinstance(text_confidence, list | tuple) or len(text_confidence) < 2:
                        continue
                    text, confidence = text_confidence[0], text_confidence[1]
                    if text:
                        text_lines.append(str(text))
                        confidence_scores.append(float(confidence))

        return text_lines, confidence_scores
