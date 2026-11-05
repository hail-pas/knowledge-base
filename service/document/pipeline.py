import os
import tempfile
from pathlib import Path
from uuid import UUID
from typing import Any

from loguru import logger

from ext.document_parser import DocumentParser
from ext.document_parser.core.parse_result import OutputFormat, PageResult, ParseResult
from ext.ext_tortoise.models.knowledge_base import Document, DocumentChunk, DocumentPages
from ext.text_chunker import TextChunker
from ext.ext_tortoise.main import ConnectionNameEnum
from tortoise.transactions import in_transaction
from service.document.storage import DocumentStorage


class DocumentPipeline:
    def __init__(self, document: Document):
        self.document = document
        self.storage = DocumentStorage(document)

    async def parse(
        self,
        *,
        workflow_uid: UUID,
        engine: str | None = None,
        output_format: OutputFormat = OutputFormat.AUTO,
        options: dict[str, Any] | None = None,
    ) -> dict:
        temp_path = None
        committed = False
        new_uri = None
        old_uri = self.document.parsed_uri
        try:
            if self.document.uri.lower().startswith(("http://", "https://")):
                source_path = self.document.uri
            else:
                fd, temp_path = tempfile.mkstemp(suffix=self.document.extension)
                os.close(fd)
                await self.storage.download_source(temp_path)
                source_path = temp_path

            result = await DocumentParser().parse(
                file_path=source_path, engine=engine, output_format=output_format, options=options
            )
            new_uri = await self.storage.upload_parsed(result.content, result.format.value)
            records = []
            if result.pages:
                for page in result.pages:
                    metadata = {
                        **(page.metadata or {}),
                        **(result.metadata or {}),
                        **(result.parse_metadata or {}),
                        "engine_used": result.engine_used,
                    }
                    content = page.content
                    if len(result.pages) == 1 and len(content) > 65535:
                        content = ""
                    records.append(DocumentPages(
                        document_id=self.document.id,
                        page_number=page.page_number,
                        content=content,
                        tables=[table.model_dump() for table in page.tables],
                        images=list(page.images),
                        metadata=metadata,
                    ))
            async with in_transaction(connection_name=ConnectionNameEnum.knowledge_base.value) as conn:
                current = await Document.filter(
                    id=self.document.id,
                    current_workflow_uid=workflow_uid,
                ).using_db(conn).select_for_update().first()
                if current is None:
                    raise RuntimeError("Workflow no longer owns this document")
                self.document.parsed_uri = new_uri
                await self.document.save(using_db=conn, update_fields=["parsed_uri"])
                await DocumentPages.filter(document_id=self.document.id).using_db(conn).delete()
                if records:
                    await DocumentPages.bulk_create(records, using_db=conn)
            committed = True

            if old_uri and old_uri != new_uri:
                try:
                    await self.storage.delete_managed_uri(old_uri)
                except Exception:
                    logger.exception("Failed to remove previous parsed artifact for document {}", self.document.id)

            logger.info("Document parsed: {}, engine={}, pages={}", self.document.file_name,
                        result.engine_used, len(result.pages))
            return {
                "parsed_uri": new_uri,
                "page_count": len(result.pages),
                "has_pages": bool(result.pages),
                "format": result.format.value,
                "engine_used": result.engine_used,
            }
        except Exception:
            if not committed and new_uri:
                try:
                    await self.storage.delete_managed_uri(new_uri)
                except Exception:
                    logger.exception("Failed to rollback parsed artifact for document {}", self.document.id)
            if not committed:
                self.document.parsed_uri = old_uri
            raise
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass
                except OSError:
                    logger.warning("Failed to remove temporary source file {}", temp_path)

    async def chunk(
        self,
        *,
        workflow_uid: UUID,
        strategy: str = "auto",
        config: dict | None = None,
    ) -> dict:
        content = await self.storage.read_parsed()
        db_pages = await DocumentPages.filter(document_id=self.document.id, deleted_at=0).order_by("page_number")
        if not db_pages:
            pages = [PageResult(page_number=1, content=content)]
        else:
            pages = [
                PageResult(page_number=p.page_number, content=p.content, metadata=p.metadata, tables=p.tables, images=p.images)
                for p in db_pages
            ]
            if len(db_pages) == 1 and not db_pages[0].content:
                pages[0].content = content
            else:
                content = DocumentPages.pages_to_content(db_pages)

        suffix = Path(self.document.parsed_uri or "").suffix
        result = ParseResult(
            content=content,
            format=OutputFormat.MARKDOWN if suffix == ".md" else OutputFormat.TEXT,
            pages=pages,
            page_count=len(pages),
            engine_used="",
        )
        chunks = await TextChunker().chunk(result, strategy=strategy, config=config or None)
        records = [
            DocumentChunk(
                document_id=self.document.id,
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
            for chunk in chunks
        ]
        async with in_transaction(connection_name=ConnectionNameEnum.knowledge_base.value) as conn:
            current = await Document.filter(
                id=self.document.id,
                current_workflow_uid=workflow_uid,
            ).using_db(conn).select_for_update().first()
            if current is None:
                raise RuntimeError("Workflow no longer owns this document")
            await DocumentChunk.filter(document_id=self.document.id).using_db(conn).delete()
            if records:
                await DocumentChunk.bulk_create(records, using_db=conn)
        saved = await DocumentChunk.filter(document_id=self.document.id, deleted_at=0).only("id").order_by("id")
        return {"chunk_count": len(saved), "chunk_ids": [chunk.id for chunk in saved], "strategy_used": strategy}
