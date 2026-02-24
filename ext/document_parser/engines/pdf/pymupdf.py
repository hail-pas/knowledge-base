from __future__ import annotations

from pathlib import Path
import fitz

from ext.document_parser.core.engine_base import BaseEngine
from ext.document_parser.core.parse_result import OutputFormat, ParseResult, PageResult, TableFormat


class PyMUPDFEngine(BaseEngine):
    engine_name = "pymupdf"
    supported_formats = [".pdf"]

    async def parse(self, file_path: str, options: dict | None = None) -> ParseResult:
        doc = fitz.open(file_path)
        pages_result = []
        all_text = []

        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            all_text.append(text)

            tables = page.find_tables()
            table_formats = []

            for table in tables:
                table_data = table.extract()
                if table_data:
                    headers = table_data[0] if table_data else []
                    rows = []
                    raw = []

                    for i, row in enumerate(table_data):
                        if row:
                            raw.append(row)
                            if i > 0 and headers:
                                row_dict = {}
                                for j, header in enumerate(headers):
                                    if header and j < len(row):
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
                    metadata={"bbox": str(page.rect)},
                )
            )

        result = ParseResult(
            content="\n\n".join(all_text),
            format=OutputFormat.TEXT,
            pages=pages_result,
            page_count=len(doc),
            metadata={
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
            },
            parse_metadata={
                "engine": "pymupdf",
                "version": fitz.version,
                "pages_processed": len(doc),
            },
            confidence=0.85,
            engine_used="pymupdf",
        )

        doc.close()
        return result
