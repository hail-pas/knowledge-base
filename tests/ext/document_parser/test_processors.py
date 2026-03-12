from __future__ import annotations

import pytest

from ext.document_parser.core.parse_result import ParseResult
from ext.document_parser.processors.cleaners import TextCleaner
from ext.document_parser.processors.deduplicator import ContentDeduplicator
from ext.document_parser.processors.sanitizers import (
    EmailSanitizer,
    IDCardSanitizer,
    PhoneSanitizer,
)


def make_result(content: str) -> ParseResult:
    return ParseResult(content=content, engine_used="test")


@pytest.mark.asyncio
async def test_text_cleaner_normalizes_whitespace_and_control_chars():
    result = make_result("Line 1\t\tLine 2\x0b\n\n")

    cleaned = await TextCleaner().process(result)

    assert cleaned.content == "Line 1 Line 2"


@pytest.mark.asyncio
async def test_email_phone_and_id_sanitizers_redact_sensitive_text():
    result = make_result("email alice@example.com phone 13800138000 id 11010519491231002X")

    result = await EmailSanitizer().process(result)
    result = await PhoneSanitizer().process(result)
    result = await IDCardSanitizer().process(result)

    assert "[EMAIL_REDACTED]" in result.content
    assert "[PHONE_REDACTED]" in result.content
    assert "[ID_CARD_REDACTED]" in result.content


@pytest.mark.asyncio
async def test_content_deduplicator_keeps_unique_paragraphs_only():
    result = make_result("same\n\nsame\n\nother")

    deduplicated = await ContentDeduplicator().process(result)

    assert deduplicated.content == "same\n\nother"
