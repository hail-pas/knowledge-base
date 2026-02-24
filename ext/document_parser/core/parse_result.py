from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from typing import Any, Optional


class OutputFormat(Enum):
    TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"


class TableFormat(BaseModel):
    headers: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    raw: list[list[Any]] = Field(default_factory=list)


class PageResult(BaseModel):
    page_number: int
    content: str
    tables: list[TableFormat] = Field(default_factory=list)
    images: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParseResult(BaseModel):
    content: str
    format: OutputFormat = Field(default=OutputFormat.TEXT)
    structured_data: Optional[Any] = Field(default=None)
    pages: Optional[list[PageResult]] = Field(default=None)
    page_count: int = Field(default=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    parse_metadata: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    engine_used: str
    parse_time: float = Field(default=0.0)
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        use_enum_values = False
