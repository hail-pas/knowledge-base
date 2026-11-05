from typing import Any

from loguru import logger

from ext.workflow import ActivityTaskTemplate, activity_task
from ext.workflow.manager import WorkflowManager
from ext.ext_tortoise.enums import ActivityStatusEnum, DocumentStatusEnum, WorkflowStatusEnum
from ext.ext_tortoise.models.knowledge_base import Document
from service.document.indexing import DocumentIndexer
from service.document.pipeline import DocumentPipeline
from service.document.status import update_document_status
from service.workflow.document.schemas import (
    DocumentParseTaskInput,
    DocumentSummarizeTaskInput,
    DocumentChunkTaskInput,
    IndexChunkTaskInput,
    GenerateTagsTaskInput,
    GenerateFAQTaskInput,
)


class DocumentWorkflowTask(ActivityTaskTemplate):
    """Workflow adapter: lifecycle/status synchronization only."""

    def _get_document_id(self) -> int:
        document_id = self.input.get("document_id")
        if not isinstance(document_id, int):
            raise ValueError("Invalid or missing document_id in activity input")
        return document_id

    async def _set_running(self) -> None:
        is_current = await update_document_status(
            self._get_document_id(),
            self.activity.workflow_uid,
            DocumentStatusEnum.processing,
        )
        if not is_current:
            raise RuntimeError("Workflow no longer owns this document")
        await super()._set_running()

    async def _set_completed(self, output: dict[str, Any]) -> None:
        await super()._set_completed(output)
        await self._sync_document_status()

    async def _handle_exception(self, exception: Exception) -> None:
        await super()._handle_exception(exception)
        await self._sync_document_status()

    async def _sync_document_status(self) -> None:
        if not await Document.filter(
            id=self._get_document_id(),
            current_workflow_uid=self.activity.workflow_uid,
        ).exists():
            return
        workflow = await WorkflowManager.get_workflow_by_uid(self.activity.workflow_uid)
        activities = await WorkflowManager.get_activities_by_workflow(self.activity.workflow_uid)
        statuses = [ActivityStatusEnum(activity.status) for activity in activities]
        workflow_status = WorkflowStatusEnum(workflow.status)

        if workflow_status in {WorkflowStatusEnum.failed, WorkflowStatusEnum.canceled} or any(
            status in {ActivityStatusEnum.failed, ActivityStatusEnum.canceled} for status in statuses
        ):
            target = DocumentStatusEnum.failure
        elif statuses and all(status == ActivityStatusEnum.completed for status in statuses):
            target = DocumentStatusEnum.success
        else:
            target = DocumentStatusEnum.processing
        await update_document_status(self._get_document_id(), self.activity.workflow_uid, target)


async def _get_document(document_id: int) -> Document:
    document = await Document.get(id=document_id).prefetch_related(
        "collection__embedding_model_config", "file_source"
    )
    return document


@activity_task(prefix="workflow_document")  # type: ignore
class DocumentParseTask(DocumentWorkflowTask):
    """Execute parsing through the document application pipeline."""

    async def execute(self) -> dict[str, Any]:
        task_input = DocumentParseTaskInput(**self.input)
        document = await _get_document(task_input.document_id)
        return await DocumentPipeline(document).parse(
            engine=task_input.engine,
            output_format=task_input.output_format,
            options=task_input.options,
            workflow_uid=self.activity.workflow_uid,
        )


@activity_task(prefix="workflow_document")  # type: ignore
class DocumentSummarizeTask(DocumentWorkflowTask):
    async def execute(self) -> dict[str, Any]:
        task_input = DocumentSummarizeTaskInput(**self.input)
        updated = await Document.filter(
            id=task_input.document_id,
            current_workflow_uid=self.activity.workflow_uid,
        ).update(short_summary="", long_summary="")
        if not updated:
            raise RuntimeError("Workflow no longer owns this document")
        return {"short_summary": "", "long_summary": ""}


@activity_task(prefix="workflow_document")  # type: ignore
class DocumentChunkTask(DocumentWorkflowTask):
    """Execute chunking through the document application pipeline."""

    async def execute(self) -> dict[str, Any]:
        task_input = DocumentChunkTaskInput(**self.input)
        document = await _get_document(task_input.document_id)
        return await DocumentPipeline(document).chunk(
            strategy=task_input.strategy,
            config=task_input.get_chunk_config(),
            workflow_uid=self.activity.workflow_uid,
        )


@activity_task(prefix="workflow_document")  # type: ignore
class IndexChunkTask(DocumentWorkflowTask):
    """Build sparse and dense indices for persisted document chunks."""

    async def execute(self) -> dict[str, Any]:
        task_input = IndexChunkTaskInput(**self.input)
        document = await _get_document(task_input.document_id)
        return await DocumentIndexer(document.collection).build(
            document,
            batch_size=task_input.batch_size,
            concurrent_batches=task_input.concurrent_batches,
            workflow_uid=self.activity.workflow_uid,
        )


@activity_task(prefix="workflow_document")  # type: ignore
class GenerateTagsTask(DocumentWorkflowTask):
    async def execute(self) -> dict[str, Any]:
        task_input = GenerateTagsTaskInput(**self.input)
        logger.info("Tag generation is not implemented for document {}", task_input.document_id)
        return {"tags": [], "message": "Tag generation is not implemented"}


@activity_task(prefix="workflow_document")  # type: ignore
class GenerateFAQTask(DocumentWorkflowTask):
    async def execute(self) -> dict[str, Any]:
        task_input = GenerateFAQTaskInput(**self.input)
        logger.info("FAQ generation is not implemented for document {}", task_input.document_id)
        return {"faq_count": 0, "message": "FAQ generation is not implemented"}
