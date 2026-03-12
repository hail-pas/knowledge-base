from __future__ import annotations

import tempfile
import contextlib
from pathlib import Path

from config.main import local_configs
from ext.document_parser.core.engine_base import BaseEngine
from ext.document_parser.core.parse_result import ParseResult, OutputFormat
from ext.document_parser.config.engine_registry import get_engine


class URLEngine(BaseEngine):
    engine_name = "url"
    supported_formats = []

    CONTENT_TYPE_ENGINE_MAP = {
        "text/html": ["trafilatura"],
        "application/pdf": ["pymupdf", "pdfplumber", "markitdown"],
        "image/png": ["markitdown", "paddleocr"],
        "image/jpeg": ["markitdown", "paddleocr"],
        "image/gif": ["markitdown"],
        "image/bmp": ["markitdown"],
        "image/tiff": ["paddleocr", "markitdown"],
        "application/vnd.ms-powerpoint": ["pptx", "markitdown"],
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": ["pptx", "markitdown"],
        "application/msword": ["docx", "markitdown"],
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ["docx", "markitdown"],
        "application/vnd.ms-excel": ["xlsx", "markitdown"],
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ["xlsx", "markitdown"],
        "text/plain": ["markdown", "markitdown"],
        "text/csv": ["csv"],
        "application/json": ["json"],
        "text/markdown": ["markdown", "markitdown"],
    }

    EXTENSION_MAP = {
        ".html": "trafilatura",
        ".htm": "trafilatura",
        ".pdf": ["pymupdf", "pdfplumber", "markitdown"],
        ".png": ["markitdown", "paddleocr"],
        ".jpg": ["markitdown", "paddleocr"],
        ".jpeg": ["markitdown", "paddleocr"],
        ".gif": ["markitdown"],
        ".bmp": ["markitdown"],
        ".tiff": ["paddleocr", "markitdown"],
        ".pptx": ["pptx", "markitdown"],
        ".ppt": ["pptx", "markitdown"],
        ".docx": ["docx", "markitdown"],
        ".doc": ["docx", "markitdown"],
        ".xlsx": ["xlsx", "markitdown"],
        ".xls": ["xlsx", "markitdown"],
        ".txt": ["markdown", "markitdown"],
        ".md": ["markdown", "markitdown"],
        ".csv": ["csv"],
        ".json": ["json"],
    }

    def can_parse(self, file_path: str) -> bool:
        return file_path.startswith(("http://", "https://"))

    async def parse(self, file_path: str, options: dict | None = None) -> ParseResult:
        tmp_path = None
        try:
            client = local_configs.extensions.httpx.instance
            response = await client.get(file_path, follow_redirects=True, timeout=30.0)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
            content = response.content

            # Determine file extension from content-type or URL
            ext = self._get_extension_from_content_type(content_type) or self._get_extension_from_url(file_path)

            # Create temp file with appropriate extension
            suffix = ext if ext else ".tmp"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            # Determine which engines to use
            engine_names = self._get_engines_for_content(content_type, ext)

            # Try each engine
            last_error = None
            for engine_name in engine_names:
                try:
                    engine_instance = get_engine(engine_name)
                    if not engine_instance:
                        continue

                    result = await engine_instance.parse(tmp_path, options)

                    # Add URL metadata
                    result.metadata["source_url"] = file_path
                    result.metadata["content_type"] = content_type
                    result.metadata["downloaded_file"] = tmp_path

                    return result

                except Exception as e:
                    last_error = e
                    continue

            # If all engines failed, raise error
            raise Exception(f"All delegated engines failed for URL {file_path}. Last error: {last_error}")

        finally:
            # Cleanup temp file
            if tmp_path and Path(tmp_path).exists():
                with contextlib.suppress(Exception):
                    Path(tmp_path).unlink()

    def _get_extension_from_content_type(self, content_type: str) -> str | None:
        """Map content-type to file extension."""
        content_type_ext_map = {
            "application/pdf": ".pdf",
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/gif": ".gif",
            "image/bmp": ".bmp",
            "image/tiff": ".tiff",
            "text/html": ".html",
            "text/plain": ".txt",
            "text/markdown": ".md",
            "text/csv": ".csv",
            "application/json": ".json",
            "application/vnd.ms-powerpoint": ".ppt",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
            "application/msword": ".doc",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
            "application/vnd.ms-excel": ".xls",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        }
        return content_type_ext_map.get(content_type)

    def _get_extension_from_url(self, url: str) -> str | None:
        """Extract extension from URL."""
        path = Path(url)
        if path.suffix:
            return path.suffix.lower()
        return None

    def _get_engines_for_content(self, content_type: str, ext: str | None) -> list[str]:
        """Determine which engines to try for the given content."""
        # First, try content-type based mapping
        engines = self.CONTENT_TYPE_ENGINE_MAP.get(content_type, [])

        # If no engines from content-type, try extension-based mapping
        if not engines and ext:
            ext_engines = self.EXTENSION_MAP.get(ext.lower(), [])
            if ext_engines:
                engines = ext_engines

        # Default to markitdown as universal fallback
        if not engines:
            engines = ["markitdown"]

        # Normalize to list
        if isinstance(engines, str):
            engines = [engines]

        return engines
