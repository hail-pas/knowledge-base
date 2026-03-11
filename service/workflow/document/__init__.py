from typing import Literal
from ext.workflow import schedule_workflow
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
    workflow_uid: str | None,
    document_id: int,
    workflow_template: dict,
    config_format: str = "dict",
    execute_mode: Literal["celery", "direct"] = "celery",
) -> str:
    workflow_uid = await schedule_workflow(
        workflow_uid=workflow_uid,
        config=workflow_template, config_format=config_format, initial_inputs={"document_id": document_id}, execute_mode=execute_mode  # type: ignore
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
