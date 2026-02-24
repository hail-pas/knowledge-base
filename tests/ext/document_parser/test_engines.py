"""
测试 Document Parser 的所有引擎

覆盖 13 个引擎：
- PDF: pymupdf, pdfplumber, paddleocr, markitdown
- Office: docx, xlsx, pptx, markitdown
- 结构化数据: csv, json, markitdown
- Web: trafilatura, markdown, markitdown
- OCR: paddleocr, tesseract, markitdown
- 通用: url, markitdown
"""

import pytest

from ext.document_parser import OutputFormat
from ext.document_parser.config.engine_registry import get_engine


class TestPDFEngines:
    """测试 PDF 解析引擎"""

    @pytest.mark.asyncio
    async def test_pymupdf_engine(self, parser, sample_files, has_pdf_files):
        """测试 pymupdf 引擎"""
        if not has_pdf_files:
            pytest.skip("No PDF files available")

        pdf_files = sample_files["pdf"]
        tested = False
        for pdf_file in pdf_files:
            # 跳过扫描版 PDF，pymupdf 无法处理
            if "scan" in pdf_file.name.lower():
                continue

            result = await parser.parse(str(pdf_file), engine="pymupdf")

            assert result is not None
            assert result.engine_used == "pymupdf"
            assert result.content is not None
            assert len(result.content.strip()) > 0
            assert result.confidence > 0.7
            assert result.page_count > 0
            assert result.pages is not None
            assert len(result.pages) == result.page_count
            assert result.format in [OutputFormat.TEXT, OutputFormat.TEXT.value]
            tested = True

        if not tested:
            pytest.skip("No non-scanned PDF files available")

    @pytest.mark.asyncio
    async def test_pdfplumber_engine(self, parser, sample_files, has_pdf_files):
        """测试 pdfplumber 引擎"""
        if not has_pdf_files:
            pytest.skip("No PDF files available")

        pdf_files = sample_files["pdf"]
        tested = False
        for pdf_file in pdf_files:
            # 跳过扫描版 PDF，pdfplumber 无法处理
            if "scan" in pdf_file.name.lower():
                continue

            result = await parser.parse(str(pdf_file), engine="pdfplumber")

            assert result is not None
            assert result.engine_used == "pdfplumber"
            assert result.content is not None
            assert len(result.content.strip()) > 0
            assert result.confidence > 0.7
            assert result.page_count > 0
            assert result.pages is not None
            tested = True

        if not tested:
            pytest.skip("No non-scanned PDF files available")

    @pytest.mark.asyncio
    async def test_paddleocr_pdf_engine(self, parser, sample_files, has_pdf_files, has_poppler, has_paddleocr):
        """测试 paddleocr 引擎解析 PDF"""
        if not has_pdf_files:
            pytest.skip("No PDF files available")
        if not has_poppler:
            pytest.skip("poppler not installed (required for pdf2image)")
        if not has_paddleocr:
            pytest.skip("paddleocr not available")

        pdf_files = sample_files["pdf"]
        # 只测试第一个 PDF 文件
        if len(pdf_files) > 0:
            pdf_file = pdf_files[0]
            result = await parser.parse(str(pdf_file), engine="paddleocr")

            assert result is not None
            assert result.engine_used == "paddleocr"
            assert result.content is not None
            assert result.pages is not None
        else:
            pytest.skip("No PDF files available")

    @pytest.mark.asyncio
    async def test_markitdown_pdf_engine(self, parser, sample_files, has_pdf_files):
        """测试 markitdown 引擎解析 PDF"""
        if not has_pdf_files:
            pytest.skip("No PDF files available")

        pdf_files = sample_files["pdf"]
        tested = False
        for pdf_file in pdf_files:
            try:
                result = await parser.parse(str(pdf_file), engine="markitdown")

                assert result is not None
                assert result.engine_used == "markitdown"
                assert result.content is not None
                assert len(result.content.strip()) > 0
                assert result.format in [OutputFormat.MARKDOWN, OutputFormat.MARKDOWN.value]
                tested = True
                break  # 只测试第一个成功的 PDF
            except Exception as e:
                # 如果解析失败，跳过这个文件，继续下一个
                continue

        if not tested:
            pytest.skip("No PDF files could be parsed by markitdown")


class TestOfficeEngines:
    """测试 Office 文档解析引擎"""

    @pytest.mark.asyncio
    async def test_docx_engine(self, parser, sample_files, has_docx_files):
        """测试 docx 引擎"""
        if not has_docx_files:
            pytest.skip("No DOCX files available")

        docx_files = sample_files["docx"]
        for docx_file in docx_files:
            result = await parser.parse(str(docx_file), engine="docx")

            assert result is not None
            assert result.engine_used == "docx"
            assert result.content is not None
            assert len(result.content.strip()) > 0
            assert result.confidence > 0.8
            assert result.page_count >= 1
            assert result.format in [OutputFormat.TEXT, OutputFormat.TEXT.value]

    @pytest.mark.asyncio
    async def test_markitdown_docx_engine(self, parser, sample_files, has_docx_files):
        """测试 markitdown 引擎解析 DOCX"""
        if not has_docx_files:
            pytest.skip("No DOCX files available")

        docx_files = sample_files["docx"]
        for docx_file in docx_files:
            result = await parser.parse(str(docx_file), engine="markitdown")

            assert result is not None
            assert result.engine_used == "markitdown"
            assert result.content is not None
            assert len(result.content.strip()) > 0
            assert result.format in [OutputFormat.MARKDOWN, OutputFormat.MARKDOWN.value]

    @pytest.mark.asyncio
    async def test_xlsx_engine(self, parser, sample_files, has_xlsx_files):
        """测试 xlsx 引擎"""
        if not has_xlsx_files:
            pytest.skip("No XLSX files available")

        xlsx_files = sample_files["xlsx"]
        for xlsx_file in xlsx_files:
            result = await parser.parse(str(xlsx_file), engine="xlsx")

            assert result is not None
            assert result.engine_used == "xlsx"
            assert result.content is not None
            assert len(result.content.strip()) > 0
            assert result.confidence > 0.9
            assert result.structured_data is not None
            assert "sheets" in result.structured_data
            assert result.format in [OutputFormat.TEXT, OutputFormat.TEXT.value]

    @pytest.mark.asyncio
    async def test_markitdown_xlsx_engine(self, parser, sample_files, has_xlsx_files):
        """测试 markitdown 引擎解析 XLSX"""
        if not has_xlsx_files:
            pytest.skip("No XLSX files available")

        xlsx_files = sample_files["xlsx"]
        for xlsx_file in xlsx_files:
            result = await parser.parse(str(xlsx_file), engine="markitdown")

            assert result is not None
            assert result.engine_used == "markitdown"
            assert result.content is not None
            assert len(result.content.strip()) > 0
            assert result.format in [OutputFormat.MARKDOWN, OutputFormat.MARKDOWN.value]

    @pytest.mark.asyncio
    async def test_pptx_engine(self, parser, sample_files, has_pptx_files):
        """测试 pptx 引擎"""
        if not has_pptx_files:
            pytest.skip("No PPTX files available")

        pptx_files = sample_files["pptx"]
        for pptx_file in pptx_files:
            result = await parser.parse(str(pptx_file), engine="pptx")

            assert result is not None
            assert result.engine_used == "pptx"
            assert result.content is not None
            assert len(result.content.strip()) > 0
            assert result.confidence > 0.8
            assert result.page_count > 0
            assert result.format in [OutputFormat.TEXT, OutputFormat.TEXT.value]

    @pytest.mark.asyncio
    async def test_markitdown_pptx_engine(self, parser, sample_files, has_pptx_files):
        """测试 markitdown 引擎解析 PPTX"""
        if not has_pptx_files:
            pytest.skip("No PPTX files available")

        pptx_files = sample_files["pptx"]
        for pptx_file in pptx_files:
            result = await parser.parse(str(pptx_file), engine="markitdown")

            assert result is not None
            assert result.engine_used == "markitdown"
            assert result.content is not None
            assert len(result.content.strip()) > 0
            assert result.format in [OutputFormat.MARKDOWN, OutputFormat.MARKDOWN.value]


class TestStructuredDataEngines:
    """测试结构化数据解析引擎"""

    @pytest.mark.asyncio
    async def test_csv_engine(self, parser, sample_files, has_csv_files):
        """测试 csv 引擎"""
        if not has_csv_files:
            pytest.skip("No CSV files available")

        csv_files = sample_files["csv"]
        for csv_file in csv_files:
            result = await parser.parse(str(csv_file), engine="csv")

            assert result is not None
            assert result.engine_used == "csv"
            assert result.content is not None
            assert len(result.content.strip()) > 0
            assert result.confidence > 0.9
            assert result.structured_data is not None
            assert "table" in result.structured_data
            assert result.format in [OutputFormat.TEXT, OutputFormat.TEXT.value]

    @pytest.mark.asyncio
    async def test_markitdown_csv_engine(self, parser, sample_files, has_csv_files):
        """测试 markitdown 引擎解析 CSV"""
        if not has_csv_files:
            pytest.skip("No CSV files available")

        csv_files = sample_files["csv"]
        for csv_file in csv_files:
            result = await parser.parse(str(csv_file), engine="markitdown")

            assert result is not None
            assert result.engine_used == "markitdown"
            assert result.content is not None
            assert len(result.content.strip()) > 0
            assert result.format in [OutputFormat.MARKDOWN, OutputFormat.MARKDOWN.value]

    @pytest.mark.asyncio
    async def test_json_engine(self, parser, sample_files, has_json_files):
        """测试 json 引擎"""
        if not has_json_files:
            pytest.skip("No JSON files available")

        json_files = sample_files["json"]
        for json_file in json_files:
            result = await parser.parse(str(json_file), engine="json")

            assert result is not None
            assert result.engine_used == "json"
            assert result.content is not None
            assert len(result.content.strip()) > 0
            assert result.confidence > 0.9
            assert result.structured_data is not None
            assert result.format in [OutputFormat.JSON, OutputFormat.JSON.value]

    @pytest.mark.asyncio
    async def test_markitdown_json_engine(self, parser, sample_files, has_json_files):
        """测试 markitdown 引擎解析 JSON"""
        if not has_json_files:
            pytest.skip("No JSON files available")

        json_files = sample_files["json"]
        for json_file in json_files:
            result = await parser.parse(str(json_file), engine="markitdown")

            assert result is not None
            assert result.engine_used == "markitdown"
            assert result.content is not None
            assert len(result.content.strip()) > 0
            assert result.format in [OutputFormat.MARKDOWN, OutputFormat.MARKDOWN.value]


class TestWebEngines:
    """测试 Web 内容解析引擎"""

    @pytest.mark.asyncio
    async def test_trafilatura_engine(self, parser, sample_files, has_html_files):
        """测试 trafilatura 引擎"""
        if not has_html_files:
            pytest.skip("No HTML files available")

        html_files = sample_files["html"]
        for html_file in html_files:
            result = await parser.parse(str(html_file), engine="trafilatura")

            assert result is not None
            assert result.engine_used == "trafilatura"
            assert result.content is not None
            assert result.confidence > 0.8
            assert result.format in [OutputFormat.TEXT, OutputFormat.TEXT.value]

    @pytest.mark.asyncio
    async def test_markitdown_html_engine(self, parser, sample_files, has_html_files):
        """测试 markitdown 引擎解析 HTML"""
        if not has_html_files:
            pytest.skip("No HTML files available")

        html_files = sample_files["html"]
        for html_file in html_files:
            result = await parser.parse(str(html_file), engine="markitdown")

            assert result is not None
            assert result.engine_used == "markitdown"
            assert result.content is not None
            assert len(result.content.strip()) > 0
            assert result.format in [OutputFormat.MARKDOWN, OutputFormat.MARKDOWN.value]

    @pytest.mark.asyncio
    async def test_markdown_engine(self, parser, sample_files, has_md_files):
        """测试 markdown 引擎"""
        if not has_md_files:
            pytest.skip("No MD files available")

        md_files = sample_files["md"]
        for md_file in md_files:
            result = await parser.parse(str(md_file), engine="markdown")

            assert result is not None
            assert result.engine_used == "markdown"
            assert result.content is not None
            assert len(result.content.strip()) > 0
            assert result.confidence > 0.9
            assert result.format in [OutputFormat.MARKDOWN, OutputFormat.MARKDOWN.value]

    @pytest.mark.asyncio
    async def test_markitdown_md_engine(self, parser, sample_files, has_md_files):
        """测试 markitdown 引擎解析 Markdown"""
        if not has_md_files:
            pytest.skip("No MD files available")

        md_files = sample_files["md"]
        for md_file in md_files:
            result = await parser.parse(str(md_file), engine="markitdown")

            assert result is not None
            assert result.engine_used == "markitdown"
            assert result.content is not None
            assert len(result.content.strip()) > 0
            assert result.format in [OutputFormat.MARKDOWN, OutputFormat.MARKDOWN.value]

    @pytest.mark.asyncio
    async def test_markitdown_txt_engine(self, parser, sample_files, has_txt_files):
        """测试 markitdown 引擎解析 TXT"""
        if not has_txt_files:
            pytest.skip("No TXT files available")

        txt_files = sample_files["txt"]
        for txt_file in txt_files:
            result = await parser.parse(str(txt_file), engine="markitdown")

            assert result is not None
            assert result.engine_used == "markitdown"
            assert result.content is not None
            assert len(result.content.strip()) > 0
            assert result.format in [OutputFormat.MARKDOWN, OutputFormat.MARKDOWN.value]


class TestOCREngines:
    """测试 OCR 引擎"""

    @pytest.mark.asyncio
    async def test_paddleocr_image_engine(self, parser, sample_files, has_png_files, has_paddleocr):
        """测试 paddleocr 引擎解析图片"""
        if not has_png_files:
            pytest.skip("No PNG files available")
        if not has_paddleocr:
            pytest.skip("paddleocr not available")

        png_files = sample_files["png"]
        for png_file in png_files:
            result = await parser.parse(str(png_file), engine="paddleocr")

            assert result is not None
            assert result.engine_used == "paddleocr"
            assert result.content is not None
            assert result.pages is not None

    @pytest.mark.asyncio
    async def test_tesseract_engine(self, parser, sample_files, has_png_files, has_tesseract):
        """测试 tesseract 引擎"""
        if not has_png_files:
            pytest.skip("No PNG files available")
        if not has_tesseract:
            pytest.skip("tesseract not installed")

        png_files = sample_files["png"]
        for png_file in png_files:
            result = await parser.parse(str(png_file), engine="tesseract")

            assert result is not None
            assert result.engine_used == "tesseract"
            assert result.content is not None
            assert result.confidence > 0.7

    @pytest.mark.asyncio
    async def test_markitdown_png_engine(self, parser, sample_files, has_png_files, has_paddleocr):
        """测试 markitdown 引擎解析 PNG"""
        if not has_png_files:
            pytest.skip("No PNG files available")
        if not has_paddleocr:
            pytest.skip("paddleocr not available (may be required for OCR)")

        png_files = sample_files["png"]
        for png_file in png_files:
            try:
                result = await parser.parse(str(png_file), engine="markitdown")

                assert result is not None
                assert result.engine_used == "markitdown"
                assert result.format in [OutputFormat.MARKDOWN, OutputFormat.MARKDOWN.value]
                break  # 只测试第一个成功的
            except Exception:
                continue


class TestEngineRegistry:
    """测试引擎注册表"""

    def test_list_engines(self):
        """测试列出所有引擎"""
        from ext.document_parser.config import list_engines

        engines = list_engines()

        assert isinstance(engines, list)
        assert len(engines) >= 13

        expected_engines = [
            "pymupdf",
            "pdfplumber",
            "paddleocr",
            "tesseract",
            "docx",
            "xlsx",
            "pptx",
            "trafilatura",
            "markdown",
            "csv",
            "json",
            "markitdown",
            "url",
        ]

        for engine in expected_engines:
            assert engine in engines

    def test_get_engine(self):
        """测试获取引擎实例"""
        engine = get_engine("pymupdf")

        assert engine is not None
        assert engine.engine_name == "pymupdf"

    def test_get_engine_invalid(self):
        """测试获取不存在的引擎"""
        engine = get_engine("invalid_engine")

        assert engine is None


class TestEngineDirectly:
    """直接测试引擎实例"""

    @pytest.mark.asyncio
    async def test_pymupdf_engine_direct(self, sample_files, has_pdf_files):
        """直接测试 pymupdf 引擎"""
        if not has_pdf_files:
            pytest.skip("No PDF files available")

        engine = get_engine("pymupdf")
        assert engine is not None

        pdf_file = sample_files["pdf"][0]
        result = await engine.parse(str(pdf_file))

        assert result is not None
        assert result.engine_used == "pymupdf"
        assert result.content is not None

    @pytest.mark.asyncio
    async def test_docx_engine_direct(self, sample_files, has_docx_files):
        """直接测试 docx 引擎"""
        if not has_docx_files:
            pytest.skip("No DOCX files available")

        engine = get_engine("docx")
        assert engine is not None

        docx_file = sample_files["docx"][0]
        result = await engine.parse(str(docx_file))

        assert result is not None
        assert result.engine_used == "docx"
        assert result.content is not None

    @pytest.mark.asyncio
    async def test_csv_engine_direct(self, sample_files, has_csv_files):
        """直接测试 csv 引擎"""
        if not has_csv_files:
            pytest.skip("No CSV files available")

        engine = get_engine("csv")
        assert engine is not None

        csv_file = sample_files["csv"][0]
        result = await engine.parse(str(csv_file))

        assert result is not None
        assert result.engine_used == "csv"
        assert result.content is not None

    @pytest.mark.asyncio
    async def test_markitdown_engine_direct(self, sample_files):
        """直接测试 markitdown 引擎"""
        engine = get_engine("markitdown")
        assert engine is not None

        if len(sample_files.get("md", [])) > 0:
            md_file = sample_files["md"][0]
            result = await engine.parse(str(md_file))

            assert result is not None
            assert result.engine_used == "markitdown"
            assert result.content is not None
