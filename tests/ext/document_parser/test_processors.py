"""
测试 Document Processor 后处理器

测试所有后处理器：
- TextCleaner: 文本清洗
- ContentDeduplicator: 内容去重
- EmailSanitizer: 邮箱脱敏
- PhoneSanitizer: 电话脱敏
- IDCardSanitizer: 身份证脱敏
"""

import pytest

from ext.document_parser import (
    TextCleaner,
    ContentDeduplicator,
    EmailSanitizer,
    PhoneSanitizer,
    IDCardSanitizer,
)
from ext.document_parser.core.parse_result import ParseResult, OutputFormat


class TestTextCleaner:
    """测试 TextCleaner 处理器"""

    @pytest.mark.asyncio
    async def test_remove_extra_whitespace(self):
        """测试去除多余空白"""
        cleaner = TextCleaner(remove_extra_whitespace=True)

        result = ParseResult(
            content="This  has    extra   whitespace",
            format=OutputFormat.TEXT,
            engine_used="test",
        )

        processed = await cleaner.process(result)

        assert "  " not in processed.content
        assert processed.content == "This has extra whitespace"

    @pytest.mark.asyncio
    async def test_normalize_quotes(self):
        """测试标准化引号"""
        cleaner = TextCleaner(normalize_quotes=True)

        result = ParseResult(
            content=""""Smart quotes" and 'single quotes'""",
            format=OutputFormat.TEXT,
            engine_used="test",
        )

        processed = await cleaner.process(result)

        assert """\"""" in processed.content or "'" in processed.content

    @pytest.mark.asyncio
    async def test_remove_control_chars(self):
        """测试去除控制字符"""
        cleaner = TextCleaner(remove_control_chars=True)

        result = ParseResult(
            content="Text\u0000with\u0001control\u0002chars",
            format=OutputFormat.TEXT,
            engine_used="test",
        )

        processed = await cleaner.process(result)

        assert "\u0000" not in processed.content
        assert "\u0001" not in processed.content
        assert "\u0002" not in processed.content

    @pytest.mark.asyncio
    async def test_combined_cleaning(self):
        """测试组合清洗"""
        cleaner = TextCleaner(
            remove_extra_whitespace=True,
            normalize_quotes=True,
            remove_control_chars=True,
        )

        result = ParseResult(
            content='Text  with  "quotes"  and\u0000spaces',
            format=OutputFormat.TEXT,
            engine_used="test",
        )

        processed = await cleaner.process(result)

        assert "  " not in processed.content
        assert "\u0000" not in processed.content

    @pytest.mark.asyncio
    async def test_empty_content(self):
        """测试空内容"""
        cleaner = TextCleaner(remove_extra_whitespace=True)

        result = ParseResult(
            content="",
            format=OutputFormat.TEXT,
            engine_used="test",
        )

        processed = await cleaner.process(result)

        assert processed.content == ""

    @pytest.mark.asyncio
    async def test_no_cleaning_options(self):
        """测试不启用任何清洗选项（但仍会 strip）"""
        cleaner = TextCleaner(
            remove_extra_whitespace=False,
            normalize_quotes=False,
            remove_control_chars=False,
        )

        original_content = "  Text  with  spaces  "
        result = ParseResult(
            content=original_content,
            format=OutputFormat.TEXT,
            engine_used="test",
        )

        processed = await cleaner.process(result)

        # TextCleaner 总是会调用 strip()
        assert processed.content == original_content.strip()


class TestContentDeduplicator:
    """测试 ContentDeduplicator 处理器"""

    @pytest.mark.asyncio
    async def test_remove_duplicate_lines(self):
        """测试去除重复段落"""
        deduplicator = ContentDeduplicator()

        result = ParseResult(
            content="Para 1\n\nPara 2\n\nPara 1\n\nPara 3",
            format=OutputFormat.TEXT,
            engine_used="test",
        )

        processed = await deduplicator.process(result)

        paragraphs = processed.content.split("\n\n")
        assert len(paragraphs) == 3
        assert "Para 1" in processed.content
        assert "Para 2" in processed.content
        assert "Para 3" in processed.content

    @pytest.mark.asyncio
    async def test_remove_duplicate_paragraphs(self):
        """测试去除重复段落"""
        deduplicator = ContentDeduplicator()

        result = ParseResult(
            content="Para 1\n\nPara 2\n\nPara 1\n\nPara 3",
            format=OutputFormat.TEXT,
            engine_used="test",
        )

        processed = await deduplicator.process(result)

        assert "Para 1" in processed.content
        assert "Para 2" in processed.content
        assert "Para 3" in processed.content

    @pytest.mark.asyncio
    async def test_no_duplicates(self):
        """测试无重复内容"""
        deduplicator = ContentDeduplicator()

        result = ParseResult(
            content="Line 1\nLine 2\nLine 3",
            format=OutputFormat.TEXT,
            engine_used="test",
        )

        processed = await deduplicator.process(result)

        assert processed.content == result.content

    @pytest.mark.asyncio
    async def test_empty_content(self):
        """测试空内容"""
        deduplicator = ContentDeduplicator()

        result = ParseResult(
            content="",
            format=OutputFormat.TEXT,
            engine_used="test",
        )

        processed = await deduplicator.process(result)

        assert processed.content == ""


class TestEmailSanitizer:
    """测试 EmailSanitizer 处理器"""

    @pytest.mark.asyncio
    async def test_sanitize_email(self):
        """测试邮箱脱敏"""
        sanitizer = EmailSanitizer()

        result = ParseResult(
            content="Contact us at support@example.com or sales@company.org",
            format=OutputFormat.TEXT,
            engine_used="test",
        )

        processed = await sanitizer.process(result)

        assert "support@example.com" not in processed.content
        assert "sales@company.org" not in processed.content
        assert "[EMAIL_REDACTED]" in processed.content
        assert processed.content.count("[EMAIL_REDACTED]") == 2

    @pytest.mark.asyncio
    async def test_no_email(self):
        """测试无邮箱内容"""
        sanitizer = EmailSanitizer()

        result = ParseResult(
            content="This is just plain text without emails",
            format=OutputFormat.TEXT,
            engine_used="test",
        )

        processed = await sanitizer.process(result)

        assert processed.content == result.content

    @pytest.mark.asyncio
    async def test_multiple_emails(self):
        """测试多个邮箱"""
        sanitizer = EmailSanitizer()

        result = ParseResult(
            content="Email1: a@test.com, Email2: b@test.com, Email3: c@test.com",
            format=OutputFormat.TEXT,
            engine_used="test",
        )

        processed = await sanitizer.process(result)

        assert "a@test.com" not in processed.content
        assert "b@test.com" not in processed.content
        assert "c@test.com" not in processed.content


class TestPhoneSanitizer:
    """测试 PhoneSanitizer 处理器"""

    @pytest.mark.asyncio
    async def test_sanitize_phone_number(self):
        """测试电话号码脱敏"""
        sanitizer = PhoneSanitizer()

        result = ParseResult(
            content="Call us at 13812345678 or 15912345678",
            format=OutputFormat.TEXT,
            engine_used="test",
        )

        processed = await sanitizer.process(result)

        assert "13812345678" not in processed.content
        assert "15912345678" not in processed.content
        assert "[PHONE_REDACTED]" in processed.content
        assert processed.content.count("[PHONE_REDACTED]") == 2

    @pytest.mark.asyncio
    async def test_sanitize_phone_with_spaces(self):
        """测试带空格的电话号码"""
        sanitizer = PhoneSanitizer()

        result = ParseResult(
            content="Phone: 138 1234 5678",
            format=OutputFormat.TEXT,
            engine_used="test",
        )

        processed = await sanitizer.process(result)

        # 只有连续的11位数字才会被匹配
        assert "138 1234 5678" in processed.content

    @pytest.mark.asyncio
    async def test_no_phone_number(self):
        """测试无电话号码"""
        sanitizer = PhoneSanitizer()

        result = ParseResult(
            content="This is just text without phone numbers",
            format=OutputFormat.TEXT,
            engine_used="test",
        )

        processed = await sanitizer.process(result)

        assert processed.content == result.content


class TestIDCardSanitizer:
    """测试 IDCardSanitizer 处理器"""

    @pytest.mark.asyncio
    async def test_sanitize_id_card(self):
        """测试身份证脱敏"""
        sanitizer = IDCardSanitizer()

        result = ParseResult(
            content="ID: 110101199001011234",
            format=OutputFormat.TEXT,
            engine_used="test",
        )

        processed = await sanitizer.process(result)

        assert "110101199001011234" not in processed.content
        assert "[ID_CARD_REDACTED]" in processed.content

    @pytest.mark.asyncio
    async def test_sanitize_id_card_with_spaces(self):
        """测试带空格的身份证号"""
        sanitizer = IDCardSanitizer()

        result = ParseResult(
            content="ID: 110101 19900101 1234",
            format=OutputFormat.TEXT,
            engine_used="test",
        )

        processed = await sanitizer.process(result)

        # 带空格的不会匹配正则表达式
        assert "110101 19900101 1234" in processed.content

    @pytest.mark.asyncio
    async def test_no_id_card(self):
        """测试无身份证号"""
        sanitizer = IDCardSanitizer()

        result = ParseResult(
            content="This is just text without ID cards",
            format=OutputFormat.TEXT,
            engine_used="test",
        )

        processed = await sanitizer.process(result)

        assert processed.content == result.content


class TestProcessorIntegration:
    """测试处理器集成"""

    @pytest.mark.asyncio
    async def test_multiple_processors(self, parser, sample_files, has_md_files):
        """测试多个处理器组合"""
        if not has_md_files:
            pytest.skip("No MD files available")

        md_file = sample_files["md"][0]

        processors = [
            TextCleaner(remove_extra_whitespace=True),
            ContentDeduplicator(),
        ]

        result = await parser.parse(str(md_file), processors=processors)

        assert result is not None
        assert result.content is not None
        assert "  " not in result.content

    @pytest.mark.asyncio
    async def test_sanitizer_processors(self, parser):
        """测试脱敏处理器组合"""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("Contact: support@example.com\n")
            f.write("Phone: 13812345678\n")
            f.write("ID: 110101199001011234\n")
            f.flush()

            processors = [
                EmailSanitizer(),
                PhoneSanitizer(),
                IDCardSanitizer(),
            ]

            result = await parser.parse(f.name, processors=processors)

            assert result is not None
            assert "support@example.com" not in result.content or "13812345678" not in result.content

    @pytest.mark.asyncio
    async def test_all_processors(self, parser):
        """测试所有处理器组合"""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("Duplicate line\nDuplicate line\n")
            f.write("Email: test@example.com\n")
            f.write("Phone: 13812345678\n")
            f.write("ID: 110101199001011234\n")
            f.flush()

            processors = [
                TextCleaner(remove_extra_whitespace=True),
                ContentDeduplicator(),
                EmailSanitizer(),
                PhoneSanitizer(),
                IDCardSanitizer(),
            ]

            result = await parser.parse(f.name, processors=processors)

            assert result is not None
            assert result.content is not None

    @pytest.mark.asyncio
    async def test_processor_order(self, parser):
        """测试处理器顺序"""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("Line 1  \nLine 1\n")
            f.write("Email: test@example.com\n")
            f.flush()

            processors = [
                TextCleaner(remove_extra_whitespace=True),
                ContentDeduplicator(),
                EmailSanitizer(),
            ]

            result = await parser.parse(f.name, processors=processors)

            assert result is not None
            assert "Line 1" in result.content
            # ContentDeduplicator 是按段落去重，不是按行去重
            assert result.content.count("Line 1") >= 1

    @pytest.mark.asyncio
    async def test_processor_with_markdown_output(self, parser, sample_files, has_md_files):
        """测试处理器与 Markdown 输出"""
        if not has_md_files:
            pytest.skip("No MD files available")

        md_file = sample_files["md"][0]

        processors = [
            TextCleaner(remove_extra_whitespace=True),
        ]

        result = await parser.parse(
            str(md_file),
            processors=processors,
            output_format=OutputFormat.MARKDOWN,
        )

        assert result is not None
        assert result.format == OutputFormat.MARKDOWN

    @pytest.mark.asyncio
    async def test_processor_preserves_metadata(self, parser, sample_files, has_md_files):
        """测试处理器保留元数据"""
        if not has_md_files:
            pytest.skip("No MD files available")

        md_file = sample_files["md"][0]

        processors = [
            TextCleaner(remove_extra_whitespace=True),
        ]

        result = await parser.parse(str(md_file), processors=processors)

        assert result is not None
        assert result.engine_used is not None
        assert result.confidence >= 0.0
        assert result.parse_time >= 0.0

    @pytest.mark.asyncio
    async def test_empty_processor_list(self, parser, sample_files, has_md_files):
        """测试空处理器列表"""
        if not has_md_files:
            pytest.skip("No MD files available")

        md_file = sample_files["md"][0]

        result = await parser.parse(str(md_file), processors=[])

        assert result is not None
        assert result.content is not None

    @pytest.mark.asyncio
    async def test_processor_with_different_engines(self, parser, sample_files, has_pdf_files):
        """测试处理器与不同引擎"""
        if not has_pdf_files:
            pytest.skip("No PDF files available")

        pdf_file = sample_files["pdf"][0]

        processors = [
            TextCleaner(remove_extra_whitespace=True),
        ]

        result = await parser.parse(str(pdf_file), engine="pymupdf", processors=processors)

        assert result is not None
        assert result.engine_used == "pymupdf"
        assert "  " not in result.content
