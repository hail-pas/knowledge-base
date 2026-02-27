from service.workflow.document.schemas import (
    DocumentParseTaskInput,
    DocumentChunkTaskInput,
    IndexChunkTaskInput,
    GenerateTagsTaskInput,
    GenerateFAQTaskInput,
)

# Default configuration with auto-detection and defaults
DOCUMENT_PROCESSING_WORKFLOW_DEFAULTS = {
    "parse_document": {
        "input": DocumentParseTaskInput(document_id=0).model_dump(),
        "execute_params": {"task_name": "workflow_document.DocumentParseTask"},
        "depends_on": [],
    },
    "chunk_document": {
        "input": DocumentChunkTaskInput(document_id=0).model_dump(),
        "execute_params": {"task_name": "workflow_document.DocumentChunkTask"},
        "depends_on": ["parse_document"],
    },
    "index_chunks": {
        "input": IndexChunkTaskInput(document_id=0).model_dump(),
        "execute_params": {"task_name": "workflow_document.IndexChunkTask"},
        "depends_on": ["chunk_document"],
    },
    "generate_tags": {
        "input": GenerateTagsTaskInput(document_id=0).model_dump(),
        "execute_params": {"task_name": "workflow_document.GenerateTagsTask"},
        "depends_on": ["parse_document"],
    },
    "generate_faq": {
        "input": GenerateFAQTaskInput(document_id=0).model_dump(),
        "execute_params": {"task_name": "workflow_document.GenerateFAQTask"},
        "depends_on": ["parse_document"],
    },
}
