from pydantic import Field

from core.types import StrEnum
from ext.indexing.base import BaseIndexModel

class SourceTypeEnum(StrEnum):
    document_content = "document_content"
    document_faq = "document_faq"
    table = "table"


class _DocumentBaseIndexModel(BaseIndexModel):

    file_id: int
    file_name: str
    tags: list[str]
    source_type: str

    extras: dict
    # tenant_id: str


class DocumentContentDenseIndex(_DocumentBaseIndexModel):

    id: int = Field(default_factory=lambda: BaseIndexModel._get_id_default(), index_metadata={}) # type: ignore
    content: str

    # chunk position
    start_page: int
    start_char_index: int
    end_page: int
    end_char_index: int

    class Meta:  # type: ignore
        index_name: str = "document_content"
        dense_vector_field: str | None = None
        # partition_key = "tenant_id"


class DocumentGenerateFAQDenseIndex(_DocumentBaseIndexModel):

    id: int = Field(default_factory=lambda: BaseIndexModel._get_id_default(), index_metadata={}) # type: ignore
    question: str
    answer: str

    class Meta:  # type: ignore
        index_name: str = "document_content"
        dense_vector_field: str | None = None
        # partition_key = "tenant_id"


class DocumentContentSparseIndex(_DocumentBaseIndexModel):
    content: str

    # chunk position
    start_page: int
    start_char_index: int
    end_page: int
    end_char_index: int

    class Meta:  # type: ignore
        index_name: str = "document_content"
        dense_vector_field: str | None = None
        # partition_key = "tenant_id"
