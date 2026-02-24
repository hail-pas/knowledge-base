import re

from ext.document_parser.processors.base import BaseProcessor
from ext.document_parser.core.parse_result import ParseResult


class TextCleaner(BaseProcessor):
    def __init__(
        self,
        remove_extra_whitespace: bool = True,
        normalize_quotes: bool = True,
        remove_control_chars: bool = True,
    ):
        self.remove_extra_whitespace = remove_extra_whitespace
        self.normalize_quotes = normalize_quotes
        self.remove_control_chars = remove_control_chars

    async def process(self, result: ParseResult) -> ParseResult:
        content = result.content

        if self.remove_extra_whitespace:
            content = re.sub(r"\s+", " ", content)

        if self.normalize_quotes:
            content = content.replace('"', '"').replace('"', '"')
            content = content.replace(""", "'").replace(""", "'")

        if self.remove_control_chars:
            content = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]", "", content)

        result.content = content.strip()
        return result
