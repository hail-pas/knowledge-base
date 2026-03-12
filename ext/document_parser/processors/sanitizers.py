import re

from ext.document_parser.processors.base import BaseProcessor
from ext.document_parser.core.parse_result import ParseResult


class EmailSanitizer(BaseProcessor):
    async def process(self, result: ParseResult) -> ParseResult:
        pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        result.content = re.sub(pattern, "[EMAIL_REDACTED]", result.content)
        return result


class PhoneSanitizer(BaseProcessor):
    async def process(self, result: ParseResult) -> ParseResult:
        pattern = r"(?<!\d)1[3-9]\d{9}(?!\d)"
        result.content = re.sub(pattern, "[PHONE_REDACTED]", result.content)
        return result


class IDCardSanitizer(BaseProcessor):
    async def process(self, result: ParseResult) -> ParseResult:
        pattern = r"\b[1-9]\d{5}(18|19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b"
        result.content = re.sub(pattern, "[ID_CARD_REDACTED]", result.content)
        return result
