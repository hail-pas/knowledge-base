from pydantic import BaseModel, Field
from typing import Optional, Literal, Any, Dict

from ext.document_parser.core.parse_result import OutputFormat

# Reuse existing chunking config models
from ext.text_chunker.config.strategy_config import (
    LengthChunkConfig,
    HeadingChunkConfig,
    DelimiterChunkConfig,
    JsonChunkConfig,
)


class DocumentParseTaskInput(BaseModel):
    """Input parameters for DocumentParseTask"""

    document_id: int = Field(..., description="Document primary key")

    # Parser configuration
    engine: Optional[str] = Field(
        default=None,
        description=(
            "Parser engine to use. None = auto-detect based on file extension. "
            "Available: pymupdf, pdfplumber, docx, xlsx, pptx, trafilatura, "
            "markdown, csv, json, paddleocr, tesseract, url"
        ),
    )
    output_format: OutputFormat = Field(default=OutputFormat.TEXT, description="Output format for parsed content")
    options: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional parser-specific options (passed to engine)"
    )


class DocumentChunkTaskInput(BaseModel):
    """Input parameters for DocumentChunkTask - reuses existing chunk config models"""

    document_id: int = Field(..., description="Document primary key")

    # Chunking configuration
    strategy: Literal["auto", "length", "heading", "delimiter", "json"] = Field(
        default="auto",
        description=(
            "Chunking strategy. 'auto' selects based on document format. "
            "Options: auto, length, heading, delimiter, json"
        ),
    )

    # Strategy-specific configs (reuse existing models)
    length_config: Optional[LengthChunkConfig] = Field(
        default=None, description="Length-based chunking config (used when strategy='length')"
    )
    heading_config: Optional[HeadingChunkConfig] = Field(
        default=None, description="Heading-based chunking config (used when strategy='heading')"
    )
    delimiter_config: Optional[DelimiterChunkConfig] = Field(
        default=None, description="Delimiter-based chunking config (used when strategy='delimiter')"
    )
    json_config: Optional[JsonChunkConfig] = Field(
        default=None, description="JSON chunking config (used when strategy='json')"
    )

    def get_chunk_config(self) -> dict:
        """Get the config dict for the specified strategy"""
        if self.strategy == "length":
            return self.length_config.model_dump() if self.length_config else {}
        elif self.strategy == "heading":
            return self.heading_config.model_dump() if self.heading_config else {}
        elif self.strategy == "delimiter":
            return self.delimiter_config.model_dump() if self.delimiter_config else {}
        elif self.strategy == "json":
            return self.json_config.model_dump() if self.json_config else {}
        else:  # auto
            return {}  # Will use preset defaults


class IndexChunkTaskInput(BaseModel):
    """Input parameters for IndexChunkTask"""

    document_id: int = Field(..., description="Document primary key")

    # Indexing configuration
    batch_size: int = Field(default=20, ge=1, description="Batch size for bulk insert operations")
    concurrent_batches: int = Field(default=2, ge=1, description="Number of concurrent batch operations")


class GenerateTagsTaskInput(BaseModel):
    """Input parameters for GenerateTagsTask (placeholder)"""

    document_id: int = Field(..., description="Document primary key")


class GenerateFAQTaskInput(BaseModel):
    """Input parameters for GenerateFAQTask (placeholder)"""

    document_id: int = Field(..., description="Document primary key")
    # Future parameters for FAQ generation
    max_faq: Optional[int] = Field(default=5, description="Maximum number of FAQ pairs to generate")
    llm_model_config_id: Optional[int] = Field(default=0, description="指定使用的大模型配置")
