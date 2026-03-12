from __future__ import annotations

import json

import pandas as pd
import aiofiles

from ext.document_parser.core.engine_base import BaseEngine
from ext.document_parser.core.parse_result import (
    PageResult,
    ParseResult,
    TableFormat,
    OutputFormat,
)


class CSVEngine(BaseEngine):
    engine_name = "csv"
    supported_formats = [".csv"]

    async def parse(self, file_path: str, options: dict | None = None) -> ParseResult:
        data_frame = pd.read_csv(file_path)

        headers = data_frame.columns.tolist()
        rows = data_frame.to_dict("records")
        raw = data_frame.to_numpy().tolist()

        table_format = TableFormat(
            headers=headers,
            rows=rows,
            raw=raw,
        )

        text_content = f"# CSV Data\n\nHeaders: {', '.join(headers)}\n\n"
        text_content += f"Total rows: {len(rows)}\n\n"

        pages_result = [
            PageResult(
                page_number=1,
                content=text_content,
                tables=[table_format],
                images=[],
                metadata={},
            ),
        ]

        return ParseResult(
            content=text_content,
            format=OutputFormat.TEXT,
            pages=pages_result,
            page_count=1,
            structured_data={"table": table_format.model_dump()},
            metadata={"row_count": len(rows), "column_count": len(headers)},
            confidence=1.0,
            engine_used="csv",
        )


class JSONEngine(BaseEngine):
    engine_name = "json"
    supported_formats = [".json"]

    async def parse(self, file_path: str, options: dict | None = None) -> ParseResult:
        async with aiofiles.open(file_path, encoding="utf-8") as f:
            data = json.loads(await f.read())

        text_content = json.dumps(data, indent=2, ensure_ascii=False)

        pages_result = [
            PageResult(
                page_number=1,
                content=text_content,
                tables=[],
                images=[],
                metadata={"json_type": type(data).__name__},
            ),
        ]

        return ParseResult(
            content=text_content,
            format=OutputFormat.JSON,
            pages=pages_result,
            page_count=1,
            structured_data=data,
            metadata={"json_type": type(data).__name__},
            confidence=1.0,
            engine_used="json",
        )
