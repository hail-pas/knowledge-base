from uuid import UUID

from ext.embedding import EmbeddingModelFactory
from ext.ext_tortoise.models.knowledge_base import Collection, Document, DocumentChunk
from ext.indexing.models import CollectionIndexModelHelper
from ext.indexing.types import FilterClause


class DocumentIndexer:
    def __init__(self, collection: Collection):
        self.collection = collection
        self.helper = CollectionIndexModelHelper(collection)

    async def delete_documents(self, documents: list[Document]) -> None:
        if not documents:
            return
        if any(document.collection_id != self.collection.id for document in documents):
            raise ValueError("All documents must belong to the configured collection")
        await self._delete_by_ids([document.id for document in documents], include_faq=True)

    async def delete_document(self, document_id: int) -> None:
        await self._delete_by_ids([document_id])

    async def _delete_by_ids(self, document_ids: list[int], *, include_faq: bool = False) -> None:
        if not document_ids:
            return
        dense_model = await self.helper.get_dense_model()
        clause = FilterClause(in_list={"file_id": document_ids})
        await self.helper.sparse_model.delete_by_query(clause)
        await dense_model.delete_by_query(clause)
        if include_faq:
            faq_model = await self.helper.get_faq_model()
            await faq_model.delete_by_query(clause)

    async def _ensure_current_workflow(self, document_id: int, workflow_uid: UUID) -> None:
        if not await Document.filter(id=document_id, current_workflow_uid=workflow_uid).exists():
            raise RuntimeError("Workflow no longer owns this document")

    async def build(
        self,
        document: Document,
        *,
        workflow_uid: UUID,
        batch_size: int = 20,
        concurrent_batches: int = 2,
    ) -> dict:
        await self._ensure_current_workflow(document.id, workflow_uid)
        chunks = await DocumentChunk.filter(document_id=document.id, deleted_at=0).order_by("id")
        if not chunks:
            await self.delete_document(document.id)
            return {"indexed_chunks": 0, "sparse_index_count": 0, "dense_index_count": 0}

        config = self.collection.embedding_model_config
        if config is None:
            raise ValueError(f"Collection {self.collection.id} has no embedding model configured")
        model = await EmbeddingModelFactory.create(config)
        embeddings = []
        for offset in range(0, len(chunks), batch_size):
            batch = [chunk.content for chunk in chunks[offset:offset + batch_size]]
            embeddings.extend(await model.embed_batch(batch))

        if len(embeddings) != len(chunks):
            raise ValueError(f"Embedding count mismatch: expected {len(chunks)}, got {len(embeddings)}")
        if any(len(vector) != config.dimension for vector in embeddings):
            raise ValueError(f"Embedding dimension mismatch: expected {config.dimension}")

        dense_model = await self.helper.get_dense_model()
        await self._ensure_current_workflow(document.id, workflow_uid)
        await self.delete_document(document.id)
        sparse_docs, dense_docs = [], []
        for chunk, vector in zip(chunks, embeddings, strict=True):
            fields = dict(
                db_chunk_id=chunk.id,
                collection_id=document.collection_id,
                file_id=document.id,
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
            sparse_docs.append(self.helper.sparse_model(**fields))
            dense_docs.append(dense_model(**fields, dense_vector=vector))

        try:
            await self.helper.sparse_model.bulk_insert(
                sparse_docs, batch_size=batch_size, concurrent_batches=concurrent_batches
            )
            await dense_model.bulk_insert(
                dense_docs, batch_size=batch_size, concurrent_batches=concurrent_batches
            )
        except Exception:
            if await Document.filter(
                id=document.id,
                current_workflow_uid=workflow_uid,
            ).exists():
                await self.delete_document(document.id)
            raise
        return {
            "indexed_chunks": len(chunks),
            "sparse_index_count": len(sparse_docs),
            "dense_index_count": len(dense_docs),
        }
