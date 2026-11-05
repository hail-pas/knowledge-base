from loguru import logger
from uuid import uuid4

from ext.document_parser.core.parse_result import OutputFormat
from ext.ext_tortoise.models.knowledge_base import Document, DocumentChunk, DocumentPages
from ext.file_source import FileSourceFactory
from service.file_source.security import build_file_upload_uri, validate_public_source_uri


class DocumentStorage:
    def __init__(self, document: Document):
        self.document = document

    async def provider(self):
        return await FileSourceFactory.create(self.document.file_source)

    async def download_source(self, destination: str) -> None:
        validate_public_source_uri(self.document.file_source, self.document.collection, self.document.uri)
        await (await self.provider()).download_to_local(self.document.uri, destination)

    async def upload_parsed(self, content: str, content_format: str) -> str:
        file_source = self.document.file_source
        collection = self.document.collection
        ext = ".md" if content_format == OutputFormat.MARKDOWN.value else ".txt"
        uri = build_file_upload_uri(file_source, collection, f"{self.document.id}/parsed/{uuid4().hex}{ext}")
        content_type = "text/markdown" if ext == ".md" else "text/plain"
        metadata = await (await self.provider()).upload_file(uri, content.encode("utf-8"), content_type)
        logger.info(f"Uploaded parsed content to {metadata.uri}")
        return metadata.uri

    async def delete_managed_uri(self, uri: str) -> bool:
        validate_public_source_uri(self.document.file_source, self.document.collection, uri)
        return await (await self.provider()).delete_file(uri)

    async def read_parsed(self) -> str:
        if not self.document.parsed_uri:
            raise ValueError(f"Document {self.document.id} has no parsed content")
        validate_public_source_uri(self.document.file_source, self.document.collection, self.document.parsed_uri)
        content = await (await self.provider()).get_file(self.document.parsed_uri)
        return content.decode("utf-8")

    @staticmethod
    async def cleanup_pages(document_id: int) -> int:
        return await DocumentPages.filter(document_id=document_id).delete()

    @staticmethod
    async def cleanup_chunks(document_id: int) -> int:
        return await DocumentChunk.filter(document_id=document_id).delete()
