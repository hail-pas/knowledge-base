from __future__ import annotations

import pdfplumber

from ext.document_parser.core.engine_base import BaseEngine
from ext.document_parser.core.parse_result import OutputFormat, ParseResult, PageResult, TableFormat


class PDFPlumberEngine(BaseEngine):
    engine_name = "pdfplumber"
    supported_formats = [".pdf"]

    async def parse(self, file_path: str, options: dict | None = None) -> ParseResult:
        all_text = []
        pages_result = []

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                all_text.append(text)

                tables = page.extract_tables(
                    {
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                    }
                )

                table_formats = []
                for table in tables:
                    if table:
                        headers = table[0] if table else []
                        rows = []
                        raw = []

                        for i, row in enumerate(table):
                            if row:
                                raw.append(row)
                                if i > 0 and headers:
                                    row_dict = {}
                                    for j, header in enumerate(headers):
                                        if header is not None and j < len(row):
                                            row_dict[str(header)] = row[j]
                                    if row_dict:
                                        rows.append(row_dict)

                        table_formats.append(
                            TableFormat(
                                headers=[str(h) for h in headers if h],
                                rows=rows,
                                raw=raw,
                            )
                        )

                pages_result.append(
                    PageResult(
                        page_number=page_num + 1,
                        content=text,
                        tables=table_formats,
                    )
                )

        return ParseResult(
            content="\n\n".join([t for t in all_text if t]),
            format=OutputFormat.TEXT,
            pages=pages_result,
            page_count=len(pages_result),
            metadata={},
            confidence=0.88,
            engine_used="pdfplumber",
        )
