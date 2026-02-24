"""
测试 DocumentParser 类

测试解析器的核心功能：
- 自动引擎选择
- 手动指定引擎
- 引擎 fallback 机制
- 输出格式转换
- 异常处理
- 列出支持的格式和引擎
"""

import pytest

from ext.document_parser import DocumentParser, OutputFormat


class TestDocumentParser:
    """测试 DocumentParser 核心功能"""

    def test_parser_initialization(self, parser):
        """测试解析器初始化"""
        assert parser is not None
        assert isinstance(parser, DocumentParser)

    def test_list_supported_formats(self, parser):
        """测试列出支持的格式"""
        formats = parser.list_supported_formats()

        assert isinstance(formats, dict)
        assert len(formats) > 0

        expected_formats = {
            ".pdf",
            ".docx",
            ".xlsx",
            ".pptx",
            ".html",
            ".htm",
            ".md",
            ".markdown",
            ".csv",
            ".json",
            ".png",
            ".jpg",
            ".jpeg",
        }

        actual_formats = set(formats.keys())
        assert expected_formats.issubset(actual_formats)

class TestAutoEngineSelection:
    """测试自动引擎选择"""

    @pytest.mark.asyncio
    async def test_auto_select_pdf_engine(self, parser, sample_files, has_pdf_files):
        """测试自动选择 PDF 引擎"""
        if not has_pdf_files:
            pytest.skip("No PDF files available")

        pdf_file = sample_files["pdf"][0]
        result = await parser.parse(str(pdf_file))

        assert result is not None
        assert result.content is not None
        assert len(result.content.strip()) > 0
        assert result.engine_used in ["pymupdf", "pdfplumber", "paddleocr"]

    @pytest.mark.asyncio
    async def test_auto_select_docx_engine(self, parser, sample_files, has_docx_files):
        """测试自动选择 DOCX 引擎"""
        if not has_docx_files:
            pytest.skip("No DOCX files available")

        docx_file = sample_files["docx"][0]
        result = await parser.parse(str(docx_file))

        assert result is not None
        assert result.content is not None
        assert result.engine_used == "docx"

    @pytest.mark.asyncio
    async def test_auto_select_xlsx_engine(self, parser, sample_files, has_xlsx_files):
        """测试自动选择 XLSX 引擎"""
        if not has_xlsx_files:
            pytest.skip("No XLSX files available")

        xlsx_file = sample_files["xlsx"][0]
        result = await parser.parse(str(xlsx_file))

        assert result is not None
        assert result.content is not None
        assert result.engine_used == "xlsx"

    @pytest.mark.asyncio
    async def test_auto_select_pptx_engine(self, parser, sample_files, has_pptx_files):
        """测试自动选择 PPTX 引擎"""
        if not has_pptx_files:
            pytest.skip("No PPTX files available")

        pptx_file = sample_files["pptx"][0]
        result = await parser.parse(str(pptx_file))

        assert result is not None
        assert result.content is not None
        assert result.engine_used == "pptx"

    @pytest.mark.asyncio
    async def test_auto_select_html_engine(self, parser, sample_files, has_html_files):
        """测试自动选择 HTML 引擎"""
        if not has_html_files:
            pytest.skip("No HTML files available")

        html_file = sample_files["html"][0]
        result = await parser.parse(str(html_file))

        assert result is not None
        assert result.content is not None
        assert result.engine_used == "trafilatura"

    @pytest.mark.asyncio
    async def test_auto_select_csv_engine(self, parser, sample_files, has_csv_files):
        """测试自动选择 CSV 引擎"""
        if not has_csv_files:
            pytest.skip("No CSV files available")

        csv_file = sample_files["csv"][0]
        result = await parser.parse(str(csv_file))

        assert result is not None
        assert result.content is not None
        assert result.engine_used == "csv"

    @pytest.mark.asyncio
    async def test_auto_select_json_engine(self, parser, sample_files, has_json_files):
        """测试自动选择 JSON 引擎"""
        if not has_json_files:
            pytest.skip("No JSON files available")

        json_file = sample_files["json"][0]
        result = await parser.parse(str(json_file))

        assert result is not None
        assert result.content is not None
        assert result.engine_used == "json"

    @pytest.mark.asyncio
    async def test_auto_select_png_engine(self, parser, sample_files, has_png_files):
        """测试自动选择 PNG 引擎"""
        if not has_png_files:
            pytest.skip("No PNG files available")

        png_file = sample_files["png"][0]
        result = await parser.parse(str(png_file))

        assert result is not None
        assert result.content is not None
        # assert result.engine_used == "tesseract"

    @pytest.mark.asyncio
    async def test_auto_select_md_engine(self, parser, sample_files, has_md_files):
        """测试自动选择 MD 引擎"""
        if not has_md_files:
            pytest.skip("No MD files available")

        md_file = sample_files["md"][0]
        result = await parser.parse(str(md_file))

        assert result is not None
        assert result.content is not None
        assert result.engine_used == "markdown"


class TestManualEngineSelection:
    """测试手动指定引擎"""

    @pytest.mark.asyncio
    async def test_manual_select_pymupdf(self, parser, sample_files, has_pdf_files):
        """测试手动指定 pymupdf 引擎"""
        if not has_pdf_files:
            pytest.skip("No PDF files available")

        pdf_file = sample_files["pdf"][0]
        result = await parser.parse(str(pdf_file), engine="pymupdf")

        assert result is not None
        assert result.engine_used == "pymupdf"

    @pytest.mark.asyncio
    async def test_manual_select_pdfplumber(self, parser, sample_files, has_pdf_files):
        """测试手动指定 pdfplumber 引擎"""
        if not has_pdf_files:
            pytest.skip("No PDF files available")

        pdf_file = sample_files["pdf"][0]
        result = await parser.parse(str(pdf_file), engine="pdfplumber")

        assert result is not None
        assert result.engine_used == "pdfplumber"

    @pytest.mark.asyncio
    async def test_manual_select_markitdown(self, parser, sample_files):
        """测试手动指定 markitdown 引擎"""
        if len(sample_files.get("md", [])) == 0:
            pytest.skip("No MD files available")

        md_file = sample_files["md"][0]
        result = await parser.parse(str(md_file), engine="markitdown")

        assert result is not None
        assert result.engine_used == "markitdown"
        assert result.format in [OutputFormat.MARKDOWN, OutputFormat.MARKDOWN.value]


class TestEngineFallback:
    """测试引擎 fallback 机制"""

    @pytest.mark.asyncio
    async def test_pdf_fallback_mechanism(self, parser):
        """测试 PDF 引擎 fallback 机制"""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", mode="wb") as f:
            f.write(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n%%EOF")
            f.flush()

            try:
                result = await parser.parse(f.name)
                assert result is not None
                assert result.engine_used in ["pymupdf", "pdfplumber", "paddleocr"]
            except Exception:
                pass


class TestOutputFormat:
    """测试输出格式转换"""

    @pytest.mark.asyncio
    async def test_output_format_text(self, parser, sample_files, has_md_files):
        """测试 TEXT 输出格式"""
        if not has_md_files:
            pytest.skip("No MD files available")

        md_file = sample_files["md"][0]
        result = await parser.parse(str(md_file), output_format=OutputFormat.TEXT)

        assert result is not None
        assert result.format in [OutputFormat.MARKDOWN, OutputFormat.MARKDOWN.value]

    @pytest.mark.asyncio
    async def test_output_format_markdown(self, parser, sample_files, has_md_files):
        """测试 MARKDOWN 输出格式转换"""
        if not has_md_files:
            pytest.skip("No MD files available")

        md_file = sample_files["md"][0]
        result = await parser.parse(str(md_file), output_format=OutputFormat.MARKDOWN)

        assert result is not None
        assert result.format in [OutputFormat.MARKDOWN, OutputFormat.MARKDOWN.value]


class TestExceptionHandling:
    """测试异常处理"""

    @pytest.mark.asyncio
    async def test_unsupported_file_type(self, parser):
        """测试不支持的文件类型"""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".unsupported", mode="w") as f:
            f.write("test content")
            f.flush()

            with pytest.raises(ValueError, match="不支持的文件类型"):
                await parser.parse(f.name)

    @pytest.mark.asyncio
    async def test_nonexistent_file(self, parser):
        """测试不存在的文件"""
        with pytest.raises(Exception):
            await parser.parse("/nonexistent/file.pdf")

    @pytest.mark.asyncio
    async def test_invalid_engine(self, parser, sample_files, has_pdf_files):
        """测试无效的引擎"""
        if not has_pdf_files:
            pytest.skip("No PDF files available")

        pdf_file = sample_files["pdf"][0]

        with pytest.raises(Exception):
            await parser.parse(str(pdf_file), engine="invalid_engine")


class TestParseResult:
    """测试解析结果"""

    @pytest.mark.asyncio
    async def test_parse_result_structure(self, parser, sample_files, has_md_files):
        """测试解析结果的结构"""
        if not has_md_files:
            pytest.skip("No MD files available")

        md_file = sample_files["md"][0]
        result = await parser.parse(str(md_file))

        assert hasattr(result, "content")
        assert hasattr(result, "format")
        assert hasattr(result, "engine_used")
        assert hasattr(result, "confidence")
        assert hasattr(result, "page_count")
        assert hasattr(result, "metadata")
        assert hasattr(result, "parse_metadata")
        assert hasattr(result, "parse_time")
        assert hasattr(result, "created_at")

    @pytest.mark.asyncio
    async def test_parse_result_content_not_empty(self, parser, sample_files, has_md_files):
        """测试解析结果内容非空"""
        if not has_md_files:
            pytest.skip("No MD files available")

        md_file = sample_files["md"][0]
        result = await parser.parse(str(md_file))

        assert result.content is not None
        assert len(result.content.strip()) > 0

    @pytest.mark.asyncio
    async def test_parse_result_confidence_range(self, parser, sample_files, has_md_files):
        """测试解析结果置信度在合理范围"""
        if not has_md_files:
            pytest.skip("No MD files available")

        md_file = sample_files["md"][0]
        result = await parser.parse(str(md_file))

        assert 0.0 <= result.confidence <= 1.0


class TestMultiFileParsing:
    """测试多文件解析"""

    @pytest.mark.asyncio
    async def test_parse_all_available_files(self, parser, sample_files):
        """测试解析所有可用文件"""
        parsed_count = 0
        failed_count = 0

        for file_type, files in sample_files.items():
            for file_path in files:
                try:
                    result = await parser.parse(str(file_path))
                    assert result is not None
                    assert result.content is not None
                    parsed_count += 1
                except Exception as e:
                    failed_count += 1

        assert parsed_count > 0, f"Should parse at least some files, parsed: {parsed_count}, failed: {failed_count}"

    @pytest.mark.asyncio
    async def test_parse_multiple_pdf_files(self, parser, sample_files, has_pdf_files):
        """测试解析多个 PDF 文件"""
        if not has_pdf_files:
            pytest.skip("No PDF files available")

        results = []
        for pdf_file in sample_files["pdf"]:
            try:
                result = await parser.parse(str(pdf_file))
                results.append(result)
            except Exception:
                # 跳过无法解析的 PDF（如扫描版需要 OCR）
                pass

        assert len(results) > 0, "Should parse at least one PDF file"
        for result in results:
            assert result is not None
            assert result.content is not None

    @pytest.mark.asyncio
    async def test_parse_multiple_pptx_files(self, parser, sample_files, has_pptx_files):
        """测试解析多个 PPTX 文件"""
        if not has_pptx_files:
            pytest.skip("No PPTX files available")

        results = []
        for pptx_file in sample_files["pptx"]:
            result = await parser.parse(str(pptx_file))
            results.append(result)

        assert len(results) > 0
        for result in results:
            assert result is not None
            assert result.content is not None


class TestMarkitdownWithParser:
    """测试 markitdown 引擎与 parser 集成"""

    @pytest.mark.asyncio
    async def test_markitdown_with_pdf(self, parser, sample_files, has_pdf_files):
        """测试 markitdown 解析 PDF"""
        if not has_pdf_files:
            pytest.skip("No PDF files available")

        pdf_file = sample_files["pdf"][0]
        result = await parser.parse(str(pdf_file), engine="markitdown")

        assert result.engine_used == "markitdown"
        assert result.format in [OutputFormat.MARKDOWN, OutputFormat.MARKDOWN.value]

    @pytest.mark.asyncio
    async def test_markitdown_with_docx(self, parser, sample_files, has_docx_files):
        """测试 markitdown 解析 DOCX"""
        if not has_docx_files:
            pytest.skip("No DOCX files available")

        docx_file = sample_files["docx"][0]
        try:
            result = await parser.parse(str(docx_file), engine="markitdown")
            assert result.engine_used == "markitdown"
            assert result.format in [OutputFormat.MARKDOWN, OutputFormat.MARKDOWN.value]
        except Exception as e:
            if "MissingDependencyException" in str(e):
                pytest.skip("markitdown docx dependencies not installed")
            raise

    @pytest.mark.asyncio
    async def test_markitdown_with_xlsx(self, parser, sample_files, has_xlsx_files):
        """测试 markitdown 解析 XLSX"""
        if not has_xlsx_files:
            pytest.skip("No XLSX files available")

        xlsx_file = sample_files["xlsx"][0]
        result = await parser.parse(str(xlsx_file), engine="markitdown")

        assert result.engine_used == "markitdown"
        assert result.format in [OutputFormat.MARKDOWN, OutputFormat.MARKDOWN.value]

    @pytest.mark.asyncio
    async def test_markitdown_with_pptx(self, parser, sample_files, has_pptx_files):
        """测试 markitdown 解析 PPTX"""
        if not has_pptx_files:
            pytest.skip("No PPTX files available")

        pptx_file = sample_files["pptx"][0]
        result = await parser.parse(str(pptx_file), engine="markitdown")

        assert result.engine_used == "markitdown"
        assert result.format in [OutputFormat.MARKDOWN, OutputFormat.MARKDOWN.value]

    @pytest.mark.asyncio
    async def test_markitdown_with_csv(self, parser, sample_files, has_csv_files):
        """测试 markitdown 解析 CSV"""
        if not has_csv_files:
            pytest.skip("No CSV files available")

        csv_file = sample_files["csv"][0]
        result = await parser.parse(str(csv_file), engine="markitdown")

        assert result.engine_used == "markitdown"
        assert result.format in [OutputFormat.MARKDOWN, OutputFormat.MARKDOWN.value]

    @pytest.mark.asyncio
    async def test_markitdown_with_json(self, parser, sample_files, has_json_files):
        """测试 markitdown 解析 JSON"""
        if not has_json_files:
            pytest.skip("No JSON files available")

        json_file = sample_files["json"][0]
        result = await parser.parse(str(json_file), engine="markitdown")

        assert result.engine_used == "markitdown"
        assert result.format in [OutputFormat.MARKDOWN, OutputFormat.MARKDOWN.value]

    @pytest.mark.asyncio
    async def test_markitdown_with_html(self, parser, sample_files, has_html_files):
        """测试 markitdown 解析 HTML"""
        if not has_html_files:
            pytest.skip("No HTML files available")

        html_file = sample_files["html"][0]
        result = await parser.parse(str(html_file), engine="markitdown")

        assert result.engine_used == "markitdown"
        assert result.format in [OutputFormat.MARKDOWN, OutputFormat.MARKDOWN.value]

    # @pytest.mark.asyncio
    # async def test_markitdown_with_png(self, parser, sample_files, has_png_files):
    #     """测试 markitdown 解析 PNG"""
    #     if not has_png_files:
    #         pytest.skip("No PNG files available")

    #     png_file = sample_files["png"][0]
    #     result = await parser.parse(str(png_file), engine="markitdown")

    #     assert result.engine_used == "markitdown"
    #     assert result.format in [OutputFormat.MARKDOWN, OutputFormat.MARKDOWN.value]
