from ext.document_parser.processors.base import BaseProcessor
from ext.document_parser.core.parse_result import ParseResult


class ContentDeduplicator(BaseProcessor):
    async def process(self, result: ParseResult) -> ParseResult:
        paragraphs = result.content.split("\n\n")
        unique_paragraphs = []
        seen = set()

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_hash = hash(para)
            if para_hash not in seen:
                seen.add(para_hash)
                unique_paragraphs.append(para)

        result.content = "\n\n".join(unique_paragraphs)
        return result
