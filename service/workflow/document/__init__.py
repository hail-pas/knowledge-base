from ext.workflow import schedule_workflow
from ext.document_parser.core.parse_result import OutputFormat
from ext.text_chunker.config.strategy_config import (
    LengthChunkConfig,
    HeadingChunkConfig,
    DelimiterChunkConfig,
    JsonChunkConfig,
)

from service.workflow.document.config import DOCUMENT_PROCESSING_WORKFLOW_DEFAULTS
from service.workflow.document.schemas import (
    DocumentParseTaskInput,
    DocumentChunkTaskInput,
    IndexChunkTaskInput,
    GenerateTagsTaskInput,
    GenerateFAQTaskInput,
)

from service.workflow.document import tasks


async def process_document(
    document_id: int,
    execute_mode: str = "celery",
    # Parse parameters
    parse_engine: str | None = None,
    parse_output_format: str = "text",
    parse_options: dict | None = None,
    # Chunk parameters
    chunk_strategy: str = "auto",
    length_config: LengthChunkConfig | None = None,
    heading_config: HeadingChunkConfig | None = None,
    delimiter_config: DelimiterChunkConfig | None = None,
    json_config: JsonChunkConfig | None = None,
    # Index parameters
    index_batch_size: int = 100,
    index_concurrent_batches: int = 5,
    # Tag/FAQ parameters
    max_tags: int = 10,
    max_faq: int = 5,
) -> str:
    """
    Start document processing workflow with custom parameters

    Args:
        document_id: Document primary key
        execute_mode: "celery" (fire-and-forget) or "direct" (wait complete)
        parse_engine: Parser engine (None = auto-detect)
        parse_output_format: Output format ("text" or "markdown")
        parse_options: Additional parser options
        chunk_strategy: Chunking strategy ("auto", "length", "heading", "delimiter", "json")
        length_config: Length chunking config (used when strategy="length")
        heading_config: Heading chunking config (used when strategy="heading")
        delimiter_config: Delimiter chunking config (used when strategy="delimiter")
        json_config: JSON chunking config (used when strategy="json")
        index_batch_size: Batch size for indexing
        index_concurrent_batches: Concurrent batches for indexing
        max_tags: Maximum tags to generate (placeholder)
        max_faq: Maximum FAQ pairs to generate (placeholder)

    Returns:
        workflow_uid (str)

    Example:
        >>> # Default processing
        >>> workflow_uid = await process_document(document_id=123)
        >>>
        >>> # Custom chunking with existing config model
        >>> workflow_uid = await process_document(
        ...     document_id=123,
        ...     chunk_strategy="heading",
        ...     heading_config=HeadingChunkConfig(max_chunk_size=3000)
        ... )
        >>>
        >>> # Full custom configuration
        >>> workflow_uid = await process_document(
        ...     document_id=123,
        ...     parse_engine="pymupdf",
        ...     parse_output_format="markdown",
        ...     chunk_strategy="length",
        ...     length_config=LengthChunkConfig(chunk_size=500, overlap=100),
        ...     index_batch_size=200
        ... )
    """
    # Build workflow config with custom parameters
    workflow_config = {
        "parse_document": {
            "input": DocumentParseTaskInput(
                document_id=document_id, engine=parse_engine, output_format=parse_output_format, options=parse_options  # type: ignore
            ).model_dump(),
            "execute_params": {"task_name": "workflow_document.DocumentParseTask"},
            "depends_on": [],
        },
        "chunk_document": {
            "input": DocumentChunkTaskInput(
                document_id=document_id,
                strategy=chunk_strategy,  # type: ignore
                length_config=length_config,
                heading_config=heading_config,
                delimiter_config=delimiter_config,
                json_config=json_config,
            ).model_dump(),
            "execute_params": {"task_name": "workflow_document.DocumentChunkTask"},
            "depends_on": ["parse_document"],
        },
        "index_chunks": {
            "input": IndexChunkTaskInput(
                document_id=document_id, batch_size=index_batch_size, concurrent_batches=index_concurrent_batches
            ).model_dump(),
            "execute_params": {"task_name": "workflow_document.IndexChunkTask"},
            "depends_on": ["chunk_document"],
        },
        "generate_tags": {
            "input": GenerateTagsTaskInput(document_id=document_id, max_tags=max_tags).model_dump(),
            "execute_params": {"task_name": "workflow_document.GenerateTagsTask"},
            "depends_on": ["parse_document"],
        },
        "generate_faq": {
            "input": GenerateFAQTaskInput(document_id=document_id, max_faq=max_faq).model_dump(),
            "execute_params": {"task_name": "workflow_document.GenerateFAQTask"},
            "depends_on": ["parse_document"],
        },
    }

    workflow_uid = await schedule_workflow(
        config=workflow_config, config_format="dict", initial_inputs={}, execute_mode=execute_mode  # type: ignore
    )

    return workflow_uid


__all__ = [
    "process_document",
    "DOCUMENT_PROCESSING_WORKFLOW_DEFAULTS",
    # Also export schemas for users who want to build custom workflows
    "DocumentParseTaskInput",
    "DocumentChunkTaskInput",
    "IndexChunkTaskInput",
    "GenerateTagsTaskInput",
    "GenerateFAQTaskInput",
]
