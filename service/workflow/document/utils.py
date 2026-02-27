from loguru import logger
from ext.document_parser.core.parse_result import OutputFormat
from ext.file_source import FileSourceFactory
from ext.ext_tortoise.models.knowledge_base import (
    Document,
    DocumentPages,
    DocumentChunk,
    FileSource,
)


async def get_document_for_workflow(document_id: int) -> Document:
    """
    Fetch document with collection and file_source prefetched

    Args:
        document_id: Document primary key

    Returns:
        Document instance with related objects prefetched
    """
    return await Document.get(id=document_id).prefetch_related("collection__embedding_model_config", "file_source")


def get_parsed_uri_extension(content_format: str) -> str:
    """
    Returns .md or .txt based on format

    Args:
        content_format: OutputFormat value

    Returns:
        File extension (.md or .txt)
    """
    return ".md" if content_format == OutputFormat.MARKDOWN else ".txt"


async def upload_parsed_content(file_source: FileSource, document_id: int, content: str, content_format: str) -> str:
    """
    Upload parsed content to file source

    Args:
        file_source: FileSource model instance
        document_id: Document ID
        content: Parsed content string
        content_format: OutputFormat value

    Returns:
        URI of uploaded file
    """
    provider = await FileSourceFactory.create(file_source)
    ext = get_parsed_uri_extension(content_format)
    uri = f"{document_id}/parsed{ext}"

    content_bytes = content.encode("utf-8")
    content_type = "text/markdown" if ext == ".md" else "text/plain"

    metadata = await provider.upload_file(uri, content_bytes, content_type)
    logger.info(f"Uploaded parsed content to {metadata.uri}")
    return metadata.uri


async def delete_parsed_content(file_source: FileSource, uri: str) -> bool:
    """
    Delete uploaded parsed content for rollback

    Args:
        file_source: FileSource model instance
        uri: URI of content to delete

    Returns:
        True if deleted successfully
    """
    try:
        provider = await FileSourceFactory.create(file_source)
        result = await provider.delete_file(uri)
        logger.info(f"Deleted parsed content at {uri}")
        return result
    except Exception as e:
        logger.warning(f"Failed to delete parsed content at {uri}: {e}")
        return False


async def cleanup_document_pages(document_id: int) -> int:
    """
    Delete all DocumentPages records for document

    Args:
        document_id: Document ID

    Returns:
        Number of records deleted
    """
    count = await DocumentPages.filter(document_id=document_id).delete()
    logger.info(f"Cleaned up {count} document pages for document {document_id}")
    return count


async def cleanup_document_chunks(document_id: int) -> int:
    """
    Delete all DocumentChunk records for document

    Args:
        document_id: Document ID

    Returns:
        Number of records deleted
    """
    count = await DocumentChunk.filter(document_id=document_id).delete()
    logger.info(f"Cleaned up {count} document chunks for document {document_id}")
    return count
