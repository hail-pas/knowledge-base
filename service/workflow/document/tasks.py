import os
import tempfile
from typing import Any
from pathlib import Path

from loguru import logger

from ext.workflow import ActivityTaskTemplate, activity_task
from ext.embedding import EmbeddingModelFactory
from ext.file_source import FileSourceFactory
from ext.text_chunker import TextChunker
from ext.indexing.types import FilterClause
from ext.document_parser import DocumentParser
from ext.indexing.models import CollectionIndexModelHelper
from ext.workflow.manager import WorkflowManager
from ext.ext_tortoise.enums import DocumentStatusEnum, WorkflowStatusEnum
from service.workflow.document.utils import (
    delete_parsed_content,
    upload_parsed_content,
    cleanup_document_pages,
    update_document_status,
    cleanup_document_chunks,
    get_document_for_workflow,
)
from service.workflow.document.schemas import (
    IndexChunkTaskInput,
    GenerateFAQTaskInput,
    GenerateTagsTaskInput,
    DocumentChunkTaskInput,
    DocumentParseTaskInput,
    DocumentSummarizeTaskInput,
)
from ext.document_parser.core.parse_result import PageResult, ParseResult, OutputFormat
from ext.ext_tortoise.models.knowledge_base import (
    DocumentChunk,
    DocumentPages,
)


class DocumentWorkflowTask(ActivityTaskTemplate):
    """Common lifecycle hooks for document workflow tasks."""

    async def _set_running(self) -> None:
        await update_document_status(self._get_document_id(), DocumentStatusEnum.processing)
        await super()._set_running()

    async def _set_completed(self, output: dict[str, Any]) -> None:
        await super()._set_completed(output)
        await self._sync_document_status_with_workflow()

    async def _handle_exception(self, exception: Exception) -> None:
        await super()._handle_exception(exception)
        await self._sync_document_status_with_workflow()

    def _get_document_id(self) -> int:
        document_id = self.input.get("document_id")
        if not isinstance(document_id, int):
            raise ValueError("Invalid or missing document_id in activity input")
        return document_id

    async def _sync_document_status_with_workflow(self) -> None:
        workflow = await WorkflowManager.get_workflow_by_uid(self.activity.workflow_uid)
        workflow_status = WorkflowStatusEnum(workflow.status)

        target_status: DocumentStatusEnum
        if workflow_status == WorkflowStatusEnum.completed:
            target_status = DocumentStatusEnum.success
        elif workflow_status in [WorkflowStatusEnum.failed, WorkflowStatusEnum.canceled]:
            target_status = DocumentStatusEnum.failure
        else:
            target_status = DocumentStatusEnum.processing

        await update_document_status(self._get_document_id(), target_status)


@activity_task(prefix="workflow_document")  # type: ignore
class DocumentParseTask(DocumentWorkflowTask):
    """Parse document and save parsed content"""

    async def execute(self) -> dict[str, Any]:
        # Validate and parse input
        task_input = DocumentParseTaskInput(**self.input)
        document = await get_document_for_workflow(task_input.document_id)

        # Get file content from file source
        provider = await FileSourceFactory.create(document.file_source)

        # 下载文件到临时文件，处理完之后删除
        # 当uri为 http 地址时，则不需要下载，直接使用 document.uri 进行下一步解析，否则使用本地临时文件路径

        temp_file_path: str | None = None
        if document.uri.startswith("http"):
            file_path = document.uri
        else:
            temp_fd, temp_file_path = tempfile.mkstemp(suffix=document.extension)
            os.close(temp_fd)
            file_path = temp_file_path
            await provider.download_to_local(document.uri, file_path)

        try:
            # Parse document with configured parameters
            parser = DocumentParser()
            parse_result = await parser.parse(
                file_path=file_path,
                engine=task_input.engine,  # None = auto-detect
                output_format=task_input.output_format,
                options=task_input.options,
            )

            if document.parsed_uri:
                await provider.delete_file(document.parsed_uri)

            await cleanup_document_pages(task_input.document_id)

            # Upload parsed content
            parsed_uri = await upload_parsed_content(
                document.file_source,
                task_input.document_id,
                parse_result.content,
                parse_result.format.value,
            )

            # Update document
            document.parsed_uri = parsed_uri
            await document.save()

            # Save pages if exists and more than 1 page
            page_count = 0
            has_pages = False
            if parse_result.pages:
                pages_data = []
                for page in parse_result.pages:
                    metadata = page.metadata or {}
                    metadata.update(parse_result.metadata)
                    metadata.update(parse_result.parse_metadata)
                    metadata.update({"engine_used": parse_result.engine_used})

                    pages_data.append(
                        DocumentPages(
                            document_id=task_input.document_id,
                            page_number=page.page_number,
                            content=page.content,
                            tables=[t.model_dump() for t in (page.tables or [])],
                            images=list(page.images or []),
                            metadata=metadata,
                        ),
                    )

                # 太长了,content 可以不存储到数据库，可以直接从parsed_uri文件获取
                if len(pages_data) == 1 and len(pages_data[0].content) > 65535:
                    pages_data[0].content = ""
                await DocumentPages.bulk_create(pages_data)
                page_count = len(parse_result.pages)
                has_pages = True

            logger.info(
                f"Document parsed: {document.file_name}, "
                f"engine={parse_result.engine_used}, "
                f"format={parse_result.format.value}, pages={page_count}",
            )

            return {
                "parsed_uri": parsed_uri,
                "page_count": page_count,
                "has_pages": has_pages,
                "format": parse_result.format.value,
                "engine_used": parse_result.engine_used,
            }
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    async def _handle_exception(self, exception: Exception) -> None:
        """Cleanup on error"""
        task_input = DocumentParseTaskInput(**self.input)
        document = await get_document_for_workflow(task_input.document_id)

        # Delete uploaded content
        if document.parsed_uri:
            await delete_parsed_content(document.file_source, document.parsed_uri)
            document.parsed_uri = None  # type: ignore
            await document.save()

        # Cleanup pages
        await cleanup_document_pages(task_input.document_id)

        await super()._handle_exception(exception)


@activity_task(prefix="workflow_document")  # type: ignore
class DocumentSummarizeTask(DocumentWorkflowTask):
    async def execute(self) -> dict[str, Any]:
        task_input = DocumentSummarizeTaskInput(**self.input)

        document = await get_document_for_workflow(task_input.document_id)

        short_summary = ""
        long_summary = ""

        document.short_summary = short_summary
        document.long_summary = long_summary

        await document.save(update_fields=["short_summary", "long_summary"])

        return {
            "short_summary": short_summary,
            "long_summary": long_summary,
        }

    # async def _handle_exception(self, exception: Exception) -> None:
    #     """Cleanup on error"""
    #     task_input = DocumentSummarizeTaskInput(**self.input)
    #     await super()._handle_exception(exception)


@activity_task(prefix="workflow_document")  # type: ignore
class DocumentChunkTask(DocumentWorkflowTask):
    """Chunk parsed document content"""

    async def execute(self) -> dict[str, Any]:
        # Validate and parse input
        task_input = DocumentChunkTaskInput(**self.input)
        document = await get_document_for_workflow(task_input.document_id)

        # Read parsed content
        provider = await FileSourceFactory.create(document.file_source)
        content_bytes = await provider.get_file(document.parsed_uri)
        content = content_bytes.decode("utf-8")

        db_pages = await DocumentPages.filter(document_id=document.id).order_by("page_number")

        if not db_pages:
            pages = [PageResult(page_number=1, content=content)]
        else:
            pages = [
                PageResult(
                    page_number=page.page_number,
                    content=page.content,
                    metadata=page.metadata,
                    tables=page.tables,
                    images=page.images,
                )
                for page in db_pages
            ]
            # Keep the same page separator as document parsers and CoordinateMapper.
            content = DocumentPages.pages_to_content(db_pages)

        # Detect format from uri
        ext = Path(document.parsed_uri).suffix
        output_format = OutputFormat.MARKDOWN if ext == ".md" else OutputFormat.TEXT

        # Create ParseResult
        parse_result = ParseResult(
            content=content,
            format=output_format,
            pages=pages,
            page_count=len(pages),
            engine_used="",
        )

        # Get chunking config
        chunk_config = task_input.get_chunk_config()

        # Chunk content with configured strategy
        chunker = TextChunker()
        chunks = await chunker.chunk(
            parse_result,
            strategy=task_input.strategy,
            config=chunk_config if chunk_config else None,
        )

        await cleanup_document_chunks(task_input.document_id)

        # Save chunks
        chunk_records = []
        for chunk in chunks:
            chunk_records.append(
                DocumentChunk(
                    document_id=task_input.document_id,
                    content=chunk.content,
                    pages=chunk.pages,
                    min_page=chunk.start.page_number,
                    max_page=chunk.end.page_number,
                    start=chunk.start.model_dump(),
                    end=chunk.end.model_dump(),
                    overlap_start=chunk.overlap_start.model_dump() if chunk.overlap_start else None,
                    overlap_end=chunk.overlap_end.model_dump() if chunk.overlap_end else None,
                    metadata=chunk.metadata or {},
                ),
            )

        await DocumentChunk.bulk_create(chunk_records)

        saved_chunks = await DocumentChunk.filter(document_id=task_input.document_id).only("id").order_by("id")

        logger.info(f"Document chunked: {len(saved_chunks)} chunks created, strategy={task_input.strategy}")  # type: ignore

        return {
            "chunk_count": len(saved_chunks),  # type: ignore
            "chunk_ids": [c.id for c in saved_chunks],  # type: ignore
            "strategy_used": task_input.strategy,
        }

    async def _handle_exception(self, exception: Exception) -> None:
        """Cleanup on error"""
        task_input = DocumentChunkTaskInput(**self.input)
        await cleanup_document_chunks(task_input.document_id)
        await super()._handle_exception(exception)


@activity_task(prefix="workflow_document")  # type: ignore
class IndexChunkTask(DocumentWorkflowTask):
    """Index document chunks into search backend"""

    async def execute(self) -> dict[str, Any]:
        # Validate and parse input
        task_input = IndexChunkTaskInput(**self.input)
        document = await get_document_for_workflow(task_input.document_id)

        # Fetch all chunks
        chunks = await DocumentChunk.filter(document_id=task_input.document_id)

        if not chunks:
            logger.warning(f"No chunks to index for document {task_input.document_id}")
            return {"indexed_chunks": 0, "sparse_index_count": 0, "dense_index_count": 0}

        # Get embedding model
        embedding_config = document.collection.embedding_model_config
        embedding_model = await EmbeddingModelFactory.create(embedding_config)

        # Generate embeddings for all chunks in batches to control memory usage.
        texts = [chunk.content for chunk in chunks]
        embeddings = []
        for offset in range(0, len(texts), task_input.batch_size):
            batch_texts = texts[offset : offset + task_input.batch_size]
            batch_embeddings = await embedding_model.embed_batch(batch_texts)
            embeddings.extend(batch_embeddings)

        assert len(embeddings[0]) == embedding_config.dimension

        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embedding count mismatch: expected {len(chunks)}, got {len(embeddings)}",
            )

        # Create index helper
        helper = CollectionIndexModelHelper(document.collection)
        dense_model = await helper.get_dense_model()

        # Delete from both indices
        filter_clause = FilterClause(equals={"file_id": task_input.document_id})
        await helper.sparse_model.delete_by_query(filter_clause)
        await dense_model.delete_by_query(filter_clause)

        # Prepare sparse index documents
        sparse_docs = []
        for chunk in chunks:
            sparse_docs.append(
                helper.sparse_model(
                    db_chunk_id=chunk.id,
                    collection_id=document.collection_id,  # type: ignore
                    file_id=task_input.document_id,
                    file_name=document.file_name,
                    tags=[],
                    source_type="document_content",
                    extras={},
                    content=chunk.content,
                    start_page=chunk.min_page or 0,
                    start_char_index=0,
                    end_page=chunk.max_page or 0,
                    end_char_index=len(chunk.content),
                ),
            )

        # Prepare dense index documents
        dense_docs = []
        for chunk, emb in zip(chunks, embeddings, strict=False):
            dense_docs.append(
                dense_model(
                    db_chunk_id=chunk.id,
                    collection_id=document.collection_id,  # type: ignore
                    file_id=task_input.document_id,
                    file_name=document.file_name,
                    tags=[],
                    source_type="document_content",
                    extras={},
                    content=chunk.content,
                    start_page=chunk.min_page or 0,
                    start_char_index=0,
                    end_page=chunk.max_page or 0,
                    end_char_index=len(chunk.content),
                    dense_vector=emb,
                ),
            )

        # Bulk insert with configured batch size
        await helper.sparse_model.bulk_insert(
            sparse_docs,
            batch_size=task_input.batch_size,
            concurrent_batches=task_input.concurrent_batches,
        )
        await dense_model.bulk_insert(
            dense_docs,
            batch_size=task_input.batch_size,
            concurrent_batches=task_input.concurrent_batches,
        )

        logger.info(
            f"Indexed {len(chunks)} chunks for document {task_input.document_id} "
            f"(batch_size={task_input.batch_size}, "
            f"concurrent_batches={task_input.concurrent_batches})",
        )

        return {
            "indexed_chunks": len(chunks),
            "sparse_index_count": len(sparse_docs),
            "dense_index_count": len(dense_docs),
        }

    async def _handle_exception(self, exception: Exception) -> None:
        """Cleanup indexed chunks on error"""
        task_input = IndexChunkTaskInput(**self.input)
        document = await get_document_for_workflow(task_input.document_id)

        try:
            helper = CollectionIndexModelHelper(document.collection)
            dense_model = await helper.get_dense_model()

            # Delete from both indices
            filter_clause = FilterClause(equals={"file_id": task_input.document_id})
            await helper.sparse_model.delete_by_query(filter_clause)
            await dense_model.delete_by_query(filter_clause)

            logger.info(f"Cleaned up indexed chunks for document {task_input.document_id}")
        except Exception as e:
            logger.error(f"Failed to cleanup indexed chunks: {e}")

        await super()._handle_exception(exception)


@activity_task(prefix="workflow_document")  # type: ignore
class GenerateTagsTask(DocumentWorkflowTask):
    """Generate tags from parsed document (placeholder)"""

    async def execute(self) -> dict[str, Any]:
        task_input = GenerateTagsTaskInput(**self.input)
        logger.info(
            f"Generating tags for document {task_input.document_id} (placeholder), ",
        )

        return {"tags": [], "message": "Placeholder - tag generation not implemented"}


@activity_task(prefix="workflow_document")  # type: ignore
class GenerateFAQTask(DocumentWorkflowTask):
    """Generate FAQ from parsed document (placeholder)"""

    async def execute(self) -> dict[str, Any]:
        task_input = GenerateFAQTaskInput(**self.input)
        logger.info(
            f"Generating FAQ for document {task_input.document_id} (placeholder), "
            f"max_faq={task_input.max_faq}, style={task_input.llm_model_config_id}",
        )

        return {"faq_count": 0, "message": "Placeholder - FAQ generation not implemented"}
