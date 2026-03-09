from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from ext.document_parser.core.parse_result import ParseResult


class BaseEngine(ABC):
    engine_name: str
    supported_formats: list[str]

    @abstractmethod
    async def parse(self, file_path: str, options: dict[str, Any] | None = None) -> ParseResult:
        pass

    def can_parse(self, file_path: str) -> bool:
        return any(file_path.lower().endswith(ext) for ext in self.supported_formats)
