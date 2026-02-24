from abc import ABC, abstractmethod

from ext.document_parser.core.parse_result import ParseResult


class BaseProcessor(ABC):
    @abstractmethod
    async def process(self, result: ParseResult) -> ParseResult:
        pass
