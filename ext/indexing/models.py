from pydantic import Field

from core.types import StrEnum
from ext.indexing.base import BaseIndexModel

class SourceTypeEnum(StrEnum):
    document_content = "document_content"
    document_faq = "document_faq"


class _DocumentBaseIndexModel(BaseIndexModel):

    collection_id: int
    file_id: int
    file_name: str
    tags: list[str]
    source_type: str

    extras: dict
    # tenant_id: str


class DocumentContentSparseIndex(_DocumentBaseIndexModel):
    content: str

    # chunk position
    db_chunk_id: int
    start_page: int
    start_char_index: int
    end_page: int
    end_char_index: int

    class Meta:  # type: ignore
        index_name: str = "document_content"
        dense_vector_field: str | None = None
        # partition_key = "tenant_id"


class DocumentContentDenseIndex(DocumentContentSparseIndex):

    id: int = Field(default_factory=lambda: DocumentContentDenseIndex._get_id_default(), index_metadata={}) # type: ignore
    dense_vector: list[float]

    class Meta:  # type: ignore
        index_name: str = "document_content"
        # partition_key = "tenant_id"


class DocumentGenerateFAQDenseIndex(_DocumentBaseIndexModel):

    id: int = Field(default_factory=lambda: DocumentContentDenseIndex._get_id_default(), index_metadata={}) # type: ignore
    question: str
    dense_vector: list[float]
    answer: str
    db_faq_id: int

    class Meta:  # type: ignore
        index_name: str = "document_gfaq"
        # partition_key = "tenant_id"
