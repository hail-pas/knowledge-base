from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from loguru import logger

import ext.embedding.providers  # noqa: F401
from ext.embedding import EmbeddingModelFactory
from ext.indexing.models import CollectionIndexModelHelper
from ext.indexing.types import DenseSearchClause, FilterClause, SparseSearchClause
from ext.ext_tortoise.models.knowledge_base import Collection, Document, DocumentChunk
from service.chat.domain.schema import ChatRequestContext, ProgressLevelEnum, RetrievalBlock, StrictModel
from service.chat.runtime.session import ChatSessionContext
from service.document.schema import DocumentChunkList, DocumentList

ActionCancellationChecker = Callable[[asyncio.Event], Awaitable[None]]
ProgressCallback = Callable[..., Awaitable[None]]


class KnowledgeRetrievalExecutionResult(StrictModel):
    retrievals: list[RetrievalBlock] = []
    requested_collection_ids: list[int] = []
    searched_collection_ids: list[int] = []
    missing_collection_ids: list[int] = []
    inaccessible_collection_ids: list[int] = []
    failed_collection_ids: list[int] = []


class KnowledgeRetrievalService:
    def can_access_collection(self, collection: Collection, *, context: ChatRequestContext) -> bool:
        if context.is_staff:
            return True
        if context.account_id is None:
            return False
        return bool(
            collection.is_public
            or collection.user_id is None
            or collection.user_id == context.account_id,
        )

    def build_document_list(self, document: Document | None) -> DocumentList | None:
        return DocumentList.model_validate(document) if document is not None else None

    def build_document_chunk_list(self, chunk: DocumentChunk | None) -> DocumentChunkList | None:
        return DocumentChunkList.model_validate(chunk) if chunk is not None else None

    async def retrieve(
        self,
        *,
        query: str,
        collection_ids: list[int],
        top_k: int,
        session_context: ChatSessionContext,
        cancel_event: asyncio.Event | None = None,
        ensure_not_canceled: ActionCancellationChecker | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> KnowledgeRetrievalExecutionResult:
        results: list[RetrievalBlock] = []
        searched_collection_ids: list[int] = []
        missing_collection_ids: list[int] = []
        inaccessible_collection_ids: list[int] = []
        failed_collection_ids: list[int] = []
        for collection_id in collection_ids:
            if cancel_event is not None and ensure_not_canceled is not None:
                await ensure_not_canceled(cancel_event)
            if progress_callback is not None:
                await progress_callback(
                    "collection_start",
                    f"开始检索 collection {collection_id}",
                    data={"collection_id": collection_id},
                )
            collection = await Collection.get_or_none(id=collection_id, deleted_at=0).prefetch_related(
                "embedding_model_config",
            )
            if not collection:
                missing_collection_ids.append(collection_id)
                if progress_callback is not None:
                    await progress_callback(
                        "collection_missing",
                        f"未找到 collection {collection_id}",
                        level=ProgressLevelEnum.warning,
                        data={"collection_id": collection_id},
                    )
                continue
            if not self.can_access_collection(collection, context=session_context.request_context):
                inaccessible_collection_ids.append(collection_id)
                if progress_callback is not None:
                    await progress_callback(
                        "collection_inaccessible",
                        f"无权访问 collection {collection_id}",
                        level=ProgressLevelEnum.warning,
                        data={"collection_id": collection_id},
                    )
                continue

            helper = CollectionIndexModelHelper(collection)
            filter_clause = FilterClause(equals={"collection_id": collection.id})
            try:
                if collection.embedding_model_config:
                    embedding_model = await EmbeddingModelFactory.create(collection.embedding_model_config)
                    vector = (await embedding_model.embed_batch([query]))[0]
                    dense_model = await helper.get_dense_model()
                    index_results = await dense_model.search(
                        query_clause=DenseSearchClause(vector=vector, top_k=top_k),
                        filter_clause=filter_clause,
                        limit=top_k,
                    )
                else:
                    index_results = await helper.sparse_model.search(
                        query_clause=SparseSearchClause(query_text=query, field_name="content", top_k=top_k),
                        filter_clause=filter_clause,
                        limit=top_k,
                    )
            except Exception:
                logger.exception("Knowledge retrieval failed: collection_id={}, query={!r}", collection.id, query[:200])
                failed_collection_ids.append(collection.id)
                if progress_callback is not None:
                    await progress_callback(
                        "collection_failed",
                        f"collection {collection.id} 检索失败",
                        level=ProgressLevelEnum.error,
                        data={"collection_id": collection.id},
                    )
                continue

            searched_collection_ids.append(collection.id)
            if progress_callback is not None:
                await progress_callback(
                    "collection_completed",
                    f"collection {collection.id} 检索完成",
                    data={"collection_id": collection.id, "hit_count": len(index_results)},
                )

            chunk_ids = [
                getattr(index_item, "db_chunk_id", None)
                for index_item, _ in index_results
                if getattr(index_item, "db_chunk_id", None)
            ]
            document_ids = [
                getattr(index_item, "file_id", None)
                for index_item, _ in index_results
                if getattr(index_item, "file_id", None)
            ]
            chunk_map = (
                {chunk.id: chunk for chunk in await DocumentChunk.filter(id__in=chunk_ids, deleted_at=0)}
                if chunk_ids
                else {}
            )
            document_map = (
                {document.id: document for document in await Document.filter(id__in=document_ids, deleted_at=0)}
                if document_ids
                else {}
            )

            for index_item, score in index_results:
                chunk_id = getattr(index_item, "db_chunk_id", None)
                document_id = getattr(index_item, "file_id", None)
                chunk = chunk_map.get(chunk_id) if isinstance(chunk_id, int) else None
                document = document_map.get(document_id) if isinstance(document_id, int) else None
                snippet = getattr(index_item, "content", None) or getattr(index_item, "answer", None) or ""
                source_id = f"collection:{collection.id}:chunk:{getattr(index_item, 'db_chunk_id', index_item.id)}"
                results.append(
                    RetrievalBlock(
                        source_id=source_id,
                        collection_id=collection.id,  # type: ignore[arg-type]
                        document_id=document.id if document else None,
                        score=float(score),
                        snippet=snippet[:1200],
                        document=self.build_document_list(document),
                        chunk=self.build_document_chunk_list(chunk),
                        metadata={"collection_name": collection.name},
                    ),
                )

        return KnowledgeRetrievalExecutionResult(
            retrievals=results[:top_k],
            requested_collection_ids=list(collection_ids),
            searched_collection_ids=searched_collection_ids,
            missing_collection_ids=missing_collection_ids,
            inaccessible_collection_ids=inaccessible_collection_ids,
            failed_collection_ids=failed_collection_ids,
        )
