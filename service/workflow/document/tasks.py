import tempfile
from typing import Any
from pathlib import Path
from loguru import logger

from ext.workflow import ActivityTaskTemplate, activity_task
from ext.document_parser import DocumentParser
from ext.text_chunker import TextChunker
from ext.document_parser.core.parse_result import ParseResult, OutputFormat, PageResult
from ext.embedding import EmbeddingModelFactory
from ext.indexing.types import FilterClause
from ext.indexing.models import CollectionIndexModelHelper
from ext.file_source import FileSourceFactory
from service.workflow.document.schemas import (
    DocumentParseTaskInput,
    DocumentChunkTaskInput,
    IndexChunkTaskInput,
    GenerateTagsTaskInput,
    GenerateFAQTaskInput,
)
from service.workflow.document.utils import (
    get_document_for_workflow,
    upload_parsed_content,
    delete_parsed_content,
    cleanup_document_pages,
    cleanup_document_chunks,
)
from ext.ext_tortoise.models.knowledge_base import (
    DocumentPages,
    DocumentChunk,
)


@activity_task(prefix="workflow_document")  # type: ignore
class DocumentParseTask(ActivityTaskTemplate):
    """Parse document and save parsed content"""

    async def execute(self) -> dict[str, Any]:
        # Validate and parse input
        task_input = DocumentParseTaskInput(**self.input)
        document = await get_document_for_workflow(task_input.document_id)

        # Get file content from file source
        provider = await FileSourceFactory.create(document.file_source)

        # 下载文件到临时文件，处理完之后删除
        # 当uri为 http 地址时，则不需要下载，直接使用 document.uri 进行下一步解析，否则使用本地临时文件路径

        if document.uri.startswith("http"):
            file_path = document.uri
        else:
            file_path = tempfile.mkstemp(suffix=document.extension)[1]
            await provider.download_to_local(document.uri, file_path)

        # Parse document with configured parameters
        parser = DocumentParser()
        parse_result = await parser.parse(
            file_path=file_path,
            engine=task_input.engine,  # None = auto-detect
            output_format=task_input.output_format,
            options=task_input.options,
        )

        # Upload parsed content
        parsed_uri = await upload_parsed_content(
            document.file_source, task_input.document_id, parse_result.content, parse_result.format.value
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
                pages_data.append(
                    DocumentPages(
                        document_id=task_input.document_id,
                        page_number=page.page_number,
                        content=page.content,
                        tables=[t.model_dump() for t in (page.tables or [])],
                        images=[i for i in (page.images or [])],
                        metadata=page.metadata or {},
                    )
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
            f"format={parse_result.format.value}, pages={page_count}"
        )

        return {
            "parsed_uri": parsed_uri,
            "page_count": page_count,
            "has_pages": has_pages,
            "format": parse_result.format.value,
            "engine_used": parse_result.engine_used,
        }

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
class DocumentChunkTask(ActivityTaskTemplate):
    """Chunk parsed document content"""

    async def execute(self) -> dict[str, Any]:
        # Validate and parse input
        task_input = DocumentChunkTaskInput(**self.input)
        document = await get_document_for_workflow(task_input.document_id)

        # Read parsed content
        provider = await FileSourceFactory.create(document.file_source)
        content_bytes = await provider.get_file(document.parsed_uri)
        content = content_bytes.decode("utf-8")

        db_pages = await DocumentPages.filter(document_id=document.id)

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

        # Detect format from uri
        ext = Path(document.parsed_uri).suffix
        output_format = OutputFormat.MARKDOWN if ext == ".md" else OutputFormat.TEXT

        # Create ParseResult
        parse_result = ParseResult(
            content=content, format=output_format, pages=pages, page_count=len(pages), engine_used=""
        )

        # Get chunking config
        chunk_config = task_input.get_chunk_config()

        # Chunk content with configured strategy
        chunker = TextChunker()
        chunks = await chunker.chunk(
            parse_result, strategy=task_input.strategy, config=chunk_config if chunk_config else None
        )

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
                )
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
class IndexChunkTask(ActivityTaskTemplate):
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

        # Generate embeddings for all chunks
        texts = [chunk.content for chunk in chunks]
        embeddings = await embedding_model.embed_batch(texts)

        # Create index helper
        helper = CollectionIndexModelHelper(document.collection)

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
                )
            )

        # Prepare dense index documents
        dense_docs = []
        for chunk, emb in zip(chunks, embeddings):
            dense_docs.append(
                helper.dense_model(
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
                )
            )

        # Bulk insert with configured batch size
        await helper.sparse_model.bulk_insert(
            sparse_docs, batch_size=task_input.batch_size, concurrent_batches=task_input.concurrent_batches
        )
        await helper.dense_model.bulk_insert(
            dense_docs, batch_size=task_input.batch_size, concurrent_batches=task_input.concurrent_batches
        )

        logger.info(
            f"Indexed {len(chunks)} chunks for document {task_input.document_id} "
            f"(batch_size={task_input.batch_size}, "
            f"concurrent_batches={task_input.concurrent_batches})"
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

            # Delete from both indices
            filter_clause = FilterClause(equals={"file_id": task_input.document_id})
            await helper.sparse_model.delete_by_query(filter_clause)
            await helper.dense_model.delete_by_query(filter_clause)

            logger.info(f"Cleaned up indexed chunks for document {task_input.document_id}")
        except Exception as e:
            logger.error(f"Failed to cleanup indexed chunks: {e}")

        await super()._handle_exception(exception)


@activity_task(prefix="workflow_document")  # type: ignore
class GenerateTagsTask(ActivityTaskTemplate):
    """Generate tags from parsed document (placeholder)"""

    async def execute(self) -> dict[str, Any]:
        task_input = GenerateTagsTaskInput(**self.input)
        logger.info(
            f"Generating tags for document {task_input.document_id} (placeholder), "
        )

        return {"tags": [], "message": "Placeholder - tag generation not implemented"}


@activity_task(prefix="workflow_document")  # type: ignore
class GenerateFAQTask(ActivityTaskTemplate):
    """Generate FAQ from parsed document (placeholder)"""

    async def execute(self) -> dict[str, Any]:
        task_input = GenerateFAQTaskInput(**self.input)
        logger.info(
            f"Generating FAQ for document {task_input.document_id} (placeholder), "
            f"max_faq={task_input.max_faq}, style={task_input.llm_model_config_id}"
        )

        return {"faq_count": 0, "message": "Placeholder - FAQ generation not implemented"}
