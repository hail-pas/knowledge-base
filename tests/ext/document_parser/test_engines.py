from __future__ import annotations

from pathlib import Path

import pytest

from ext.document_parser.core.parse_result import OutputFormat
from ext.document_parser.engines.amarkitdown.amarkitdown import MarkitdownEngine
from ext.document_parser.engines.office.engines import DocxEngine, PPTXEngine, XLSXEngine
from ext.document_parser.engines.ocr.paddleocr import PaddleOCREngine
from ext.document_parser.engines.ocr.tesseract import TesseractOCREngine
from ext.document_parser.engines.pdf.pdfplumber import PDFPlumberEngine
from ext.document_parser.engines.pdf.pymupdf import PyMUPDFEngine
from ext.document_parser.engines.plain import TextEngine
from ext.document_parser.engines.structured.engines import CSVEngine, JSONEngine
from ext.document_parser.engines.web.engines import MarkdownEngine, TrafilaturaEngine
from ext.document_parser.engines.web.url import URLEngine
from tests.ext.document_parser.helpers import require_command, require_module, sample_file


@pytest.mark.asyncio
async def test_text_engine_parses_txt_fixture():
    require_module("aiofiles")

    result = await TextEngine().parse(str(sample_file("test.txt")))

    assert result.engine_used == "text"
    assert result.page_count == 1
    assert "Test plain text file." in result.content
    assert "Second line for parser validation." in result.content


@pytest.mark.asyncio
async def test_pymupdf_engine_parses_two_page_pdf_fixture():
    require_module("fitz")

    result = await PyMUPDFEngine().parse(str(sample_file("test.pdf")))

    assert result.engine_used == "pymupdf"
    assert result.page_count == 2
    assert "PDF Page 1" in result.content
    assert "PDF Page 2" in result.content
    assert len(result.pages) == 2


@pytest.mark.asyncio
async def test_pdfplumber_engine_parses_two_page_pdf_fixture():
    require_module("pdfplumber")

    result = await PDFPlumberEngine().parse(str(sample_file("test.pdf")))

    assert result.engine_used == "pdfplumber"
    assert result.page_count == 2
    assert "page one" in result.content.lower()
    assert "page two" in result.content.lower()


@pytest.mark.asyncio
async def test_docx_engine_parses_docx_fixture():
    require_module("docx")

    result = await DocxEngine().parse(str(sample_file("test.docx")))

    assert result.engine_used == "docx"
    assert result.page_count == 1
    assert result.metadata["paragraph_count"] >= 4
    assert result.metadata["table_count"] == 1
    assert "Test Document Page 1" in result.content
    assert "Test Document Page 2" in result.content


@pytest.mark.asyncio
async def test_xlsx_engine_parses_multi_sheet_fixture():
    require_module("openpyxl")

    result = await XLSXEngine().parse(str(sample_file("test.xlsx")))

    assert result.engine_used == "xlsx"
    assert result.page_count == 2
    assert result.metadata["sheet_count"] == 2
    assert [page.metadata["sheet_name"] for page in result.pages] == ["Scores", "Summary"]


@pytest.mark.asyncio
async def test_pptx_engine_parses_two_slide_fixture():
    require_module("pptx")

    result = await PPTXEngine().parse(str(sample_file("test.pptx")))

    assert result.engine_used == "pptx"
    assert result.page_count == 2
    assert result.metadata["slide_count"] == 2
    assert "Test Slide 1" in result.pages[0].content
    assert "Test Slide 2" in result.pages[1].content


@pytest.mark.asyncio
async def test_markdown_engine_parses_markdown_fixture():
    require_module("aiofiles")

    result = await MarkdownEngine().parse(str(sample_file("test.md")))

    assert result.engine_used == "markdown"
    assert result.format == OutputFormat.MARKDOWN
    assert result.page_count == 1
    assert "# Test Markdown" in result.content


@pytest.mark.asyncio
async def test_trafilatura_engine_extracts_html_fixture():
    require_module("trafilatura")

    result = await TrafilaturaEngine().parse(str(sample_file("test.html")))

    assert result.engine_used == "trafilatura"
    assert result.page_count == 1
    assert "Test HTML" in result.content
    assert "Second paragraph for extraction." in result.content


@pytest.mark.asyncio
async def test_csv_engine_parses_csv_fixture():
    require_module("pandas")

    result = await CSVEngine().parse(str(sample_file("test.csv")))

    assert result.engine_used == "csv"
    assert result.page_count == 1
    assert result.metadata == {"row_count": 2, "column_count": 3}
    assert result.structured_data is not None
    assert result.structured_data["table"]["rows"][0]["name"] == "alice"


@pytest.mark.asyncio
async def test_json_engine_parses_json_fixture():
    require_module("aiofiles")

    result = await JSONEngine().parse(str(sample_file("test.json")))

    assert result.engine_used == "json"
    assert result.format == OutputFormat.JSON
    assert result.page_count == 1
    assert result.structured_data is not None
    assert result.structured_data["items"] == [1, 2, 3]


@pytest.mark.asyncio
async def test_markitdown_engine_detects_multiple_slides():
    require_module("markitdown")

    result = await MarkitdownEngine().parse(str(sample_file("test.pptx")))

    assert result.engine_used == "markitdown"
    assert result.format == OutputFormat.MARKDOWN
    assert result.page_count == 2
    assert result.pages[0].page_number == 1
    assert result.pages[1].page_number == 2
    assert "Test Slide 1" in result.pages[0].content
    assert "Test Slide 2" in result.pages[1].content


@pytest.mark.asyncio
async def test_tesseract_engine_parses_png_fixture():
    require_module("pytesseract")
    require_module("PIL")
    require_command("tesseract")

    result = await TesseractOCREngine().parse(str(sample_file("test.png")))

    assert result.engine_used == "tesseract"
    assert result.page_count == 1
    assert "OCR" in result.content.upper()
    assert "LINE" in result.content.upper()


@pytest.mark.asyncio
async def test_paddleocr_engine_parses_png_fixture(monkeypatch: pytest.MonkeyPatch):
    require_module("paddleocr")

    monkeypatch.setenv("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    result = await PaddleOCREngine().parse(str(sample_file("test.png")))

    assert result.engine_used == "paddleocr"
    assert result.page_count == 1
    assert result.confidence > 0
    assert "OCR" in result.content.upper()


@pytest.mark.asyncio
async def test_paddleocr_engine_parses_two_page_scan_pdf(monkeypatch: pytest.MonkeyPatch):
    require_module("paddleocr")
    require_module("pdf2image")
    require_command("pdftoppm")

    monkeypatch.setenv("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    result = await PaddleOCREngine().parse(str(sample_file("test_scan.pdf")))

    assert result.engine_used == "paddleocr"
    assert result.page_count == 2
    assert len(result.pages) == 2
    assert "SCAN" in result.content.upper()


@pytest.mark.asyncio
async def test_url_engine_downloads_and_delegates_to_real_engine(monkeypatch: pytest.MonkeyPatch):
    class FakeResponse:
        def __init__(self, file_path: Path, content_type: str) -> None:
            self.content = file_path.read_bytes()
            self.headers = {"content-type": content_type}

        def raise_for_status(self) -> None:
            return None

    class FakeClient:
        async def get(self, url: str, *, follow_redirects: bool, timeout: float) -> FakeResponse:
            assert url == "https://example.test/test.pdf"
            assert follow_redirects is True
            assert timeout == 30.0
            return FakeResponse(sample_file("test.pdf"), "application/pdf")

    from config.main import local_configs

    monkeypatch.setattr(local_configs.extensions.httpx, "_client", FakeClient())

    result = await URLEngine().parse("https://example.test/test.pdf")

    assert result.page_count == 2
    assert result.metadata["source_url"] == "https://example.test/test.pdf"
    assert result.metadata["content_type"] == "application/pdf"
    assert not Path(result.metadata["downloaded_file"]).exists()
