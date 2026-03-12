from __future__ import annotations

from pathlib import Path

import pytest

from ext.document_parser.config.engine_registry import clear_cache
from ext.document_parser.core.parser import DocumentParser
from ext.document_parser.core.parse_result import OutputFormat
from ext.document_parser.processors.cleaners import TextCleaner
from ext.document_parser.processors.sanitizers import EmailSanitizer
from tests.ext.document_parser.helpers import require_module, sample_file


@pytest.fixture(autouse=True)
def clear_engine_instances():
    clear_cache()
    yield
    clear_cache()


@pytest.mark.asyncio
async def test_document_parser_uses_text_engine_for_txt():
    parser = DocumentParser()

    result = await parser.parse(str(sample_file("test.txt")))

    assert result.engine_used == "text"
    assert result.page_count == 1
    assert "Test plain text file." in result.content


@pytest.mark.asyncio
async def test_document_parser_uses_pdf_engine_for_two_page_pdf():
    require_module("fitz")

    parser = DocumentParser()
    result = await parser.parse(str(sample_file("test.pdf")))

    assert result.engine_used in {"pymupdf", "pdfplumber", "paddleocr"}
    assert result.page_count == 2
    assert "PDF Page 2" in result.content


@pytest.mark.asyncio
async def test_document_parser_can_force_markitdown_engine():
    require_module("markitdown")

    parser = DocumentParser()
    result = await parser.parse(str(sample_file("test.pptx")), engine="markitdown")

    assert result.engine_used == "markitdown"
    assert result.format == OutputFormat.MARKDOWN
    assert result.page_count == 2


@pytest.mark.asyncio
async def test_document_parser_converts_text_output_to_markdown():
    parser = DocumentParser()

    result = await parser.parse(str(sample_file("test.txt")), output_format=OutputFormat.MARKDOWN)

    assert result.engine_used == "text"
    assert result.format == OutputFormat.MARKDOWN
    assert "## Test plain text file." in result.content


@pytest.mark.asyncio
async def test_document_parser_applies_processors(tmp_path: Path):
    parser = DocumentParser()
    fixture = tmp_path / "processor_input.txt"
    fixture.write_text('Contact "Alice" via alice@example.com.\n\n', encoding="utf-8")

    result = await parser.parse(
        str(fixture),
        processors=[EmailSanitizer(), TextCleaner()],
    )

    assert result.content == 'Contact "Alice" via [EMAIL_REDACTED].'


@pytest.mark.asyncio
async def test_document_parser_routes_urls_to_url_engine(monkeypatch: pytest.MonkeyPatch):
    class FakeResult:
        engine_used = "url"
        content = "url content"
        format = OutputFormat.TEXT
        pages = []
        page_count = 1
        metadata = {}

    class FakeURLEngine:
        async def parse(self, file_path: str, options: dict | None = None) -> FakeResult:
            assert file_path == "https://example.test/data"
            assert options == {"x": 1}
            return FakeResult()

    parser = DocumentParser()

    monkeypatch.setattr("ext.document_parser.core.parser.get_engine", lambda name: FakeURLEngine() if name == "url" else None)

    result = await parser.parse("https://example.test/data", options={"x": 1})

    assert result.engine_used == "url"
    assert result.content == "url content"


def test_document_parser_lists_supported_formats_and_engines():
    parser = DocumentParser()

    formats = parser.list_supported_formats()
    engines = parser.list_available_engines()

    assert ".pdf" in formats
    assert ".txt" in formats
    assert "text" in engines
    assert "json" in engines
