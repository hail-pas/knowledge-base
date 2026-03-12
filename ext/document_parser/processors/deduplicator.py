from ext.document_parser.processors.base import BaseProcessor
from ext.document_parser.core.parse_result import ParseResult


class ContentDeduplicator(BaseProcessor):
    async def process(self, result: ParseResult) -> ParseResult:
        paragraphs = result.content.split("\n\n")
        unique_paragraphs = []
        seen = set()

        for paragraph in paragraphs:
            stripped_paragraph = paragraph.strip()
            if not stripped_paragraph:
                continue

            para_hash = hash(stripped_paragraph)
            if para_hash not in seen:
                seen.add(para_hash)
                unique_paragraphs.append(stripped_paragraph)

        result.content = "\n\n".join(unique_paragraphs)
        return result
