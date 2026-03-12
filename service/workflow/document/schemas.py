from typing import Any, Dict, Literal, Optional

from pydantic import Field, BaseModel

from ext.document_parser.core.parse_result import OutputFormat

# Reuse existing chunking config models
from ext.text_chunker.config.strategy_config import (
    JsonChunkConfig,
    LengthChunkConfig,
    HeadingChunkConfig,
    DelimiterChunkConfig,
)


class DocumentTaskBaseInput(BaseModel):
    document_id: int = Field(..., description="Document primary key")


class DocumentParseTaskInput(DocumentTaskBaseInput):
    """Input parameters for DocumentParseTask"""

    # Parser configuration
    engine: str | None = Field(
        default=None,
        description=(
            "Parser engine to use. None = auto-detect based on file extension. "
            "Available: pymupdf, pdfplumber, docx, xlsx, pptx, trafilatura, "
            "markdown, csv, json, paddleocr, tesseract, url"
        ),
    )
    output_format: OutputFormat = Field(default=OutputFormat.AUTO, description="Output format for parsed content")
    options: dict[str, Any] | None = Field(
        default=None,
        description="Additional parser-specific options (passed to engine)",
    )


class DocumentSummarizeTaskInput(DocumentTaskBaseInput):
    """Input parameters for GenerateTagsTask (placeholder)"""


class DocumentChunkTaskInput(DocumentTaskBaseInput):
    """Input parameters for DocumentChunkTask - reuses existing chunk config models"""

    # Chunking configuration
    strategy: Literal["auto", "length", "heading", "delimiter", "json"] = Field(
        default="auto",
        description=(
            "Chunking strategy. 'auto' selects based on document format. "
            "Options: auto, length, heading, delimiter, json"
        ),
    )

    # Strategy-specific configs (reuse existing models)
    length_config: LengthChunkConfig | None = Field(
        default=None,
        description="Length-based chunking config (used when strategy='length')",
    )
    heading_config: HeadingChunkConfig | None = Field(
        default=None,
        description="Heading-based chunking config (used when strategy='heading')",
    )
    delimiter_config: DelimiterChunkConfig | None = Field(
        default=None,
        description="Delimiter-based chunking config (used when strategy='delimiter')",
    )
    json_config: JsonChunkConfig | None = Field(
        default=None,
        description="JSON chunking config (used when strategy='json')",
    )

    def get_chunk_config(self) -> dict:
        """Get the config dict for the specified strategy"""
        if self.strategy == "length":
            return self.length_config.model_dump() if self.length_config else {}
        if self.strategy == "heading":
            return self.heading_config.model_dump() if self.heading_config else {}
        if self.strategy == "delimiter":
            return self.delimiter_config.model_dump() if self.delimiter_config else {}
        if self.strategy == "json":
            return self.json_config.model_dump() if self.json_config else {}
        # auto
        return {}  # Will use preset defaults


class IndexChunkTaskInput(DocumentTaskBaseInput):
    """Input parameters for IndexChunkTask"""

    # Indexing configuration
    batch_size: int = Field(default=20, ge=1, description="Batch size for bulk insert operations")
    concurrent_batches: int = Field(default=2, ge=1, description="Number of concurrent batch operations")


class GenerateTagsTaskInput(DocumentTaskBaseInput):
    """Input parameters for GenerateTagsTask (placeholder)"""


class GenerateFAQTaskInput(DocumentTaskBaseInput):
    """Input parameters for GenerateFAQTask (placeholder)"""

    # Future parameters for FAQ generation
    max_faq: int | None = Field(default=5, description="Maximum number of FAQ pairs to generate")
    llm_model_config_id: int | None = Field(default=0, description="指定使用的大模型配置")
