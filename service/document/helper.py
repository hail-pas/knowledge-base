from typing import List
from typing_extensions import Collection
from ext.ext_tortoise.models.knowledge_base import Document, DocumentChunk, DocumentPages, DocumentGeneratedFaq
from service.collection.helper import CollectionService
from ext.file_source.factory import FileSourceFactory


class DocumentService(CollectionService):

    async def delete(self, documents: List[Document]):
        assert all([self.collection.id == doc.collection_id for doc in documents]) # type: ignore

        document_ids = [doc.id for doc in documents]
        await Document.filter(id__in=document_ids).delete()
        await DocumentChunk.filter(document_id__in=document_ids).delete()
        await DocumentPages.filter(document_id__in=document_ids).delete()
        await DocumentGeneratedFaq.filter(document_id__in=document_ids).delete()

        await self.collection_index_helper.delete_by_documents(documents)
        await self.delete_filesource_related(documents)

    async def delete_filesource_related(self, documents: List[Document]):
        # TODO: gather async tasks

        assert all([self.collection.id == doc.collection_id for doc in documents]) # type: ignore

        for d in documents:
            fs = await FileSourceFactory.create(await d.file_source)
            await fs.delete_file(d.parsed_uri)
            await fs.delete_file(d.uri)
            pages = await DocumentPages.filter(document_id=d.id)
            for page in pages:
                if page.images:
                    for image in page.images:
                        await fs.delete_file(image)
