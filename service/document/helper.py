import time
import uuid
import hashlib
import mimetypes
from typing import List
from pathlib import Path
from urllib.parse import unquote, urlparse

from loguru import logger

from core.exception import ApiException
from ext.ext_tortoise.enums import DocumentStatusEnum
from ext.file_source.factory import FileSourceFactory
from service.collection.helper import CollectionService
from ext.ext_tortoise.models.knowledge_base import (
    Document,
    FileSource,
    DocumentChunk,
    DocumentPages,
    DocumentGeneratedFaq,
)


class DocumentService(CollectionService):
    async def delete(self, documents: list[Document]) -> None:
        assert all(self.collection.id == doc.collection_id for doc in documents)  # type: ignore

        document_ids = [doc.id for doc in documents]
        await Document.filter(id__in=document_ids).delete()
        await DocumentChunk.filter(document_id__in=document_ids).delete()
        await DocumentPages.filter(document_id__in=document_ids).delete()
        await DocumentGeneratedFaq.filter(document_id__in=document_ids).delete()

        await self.collection_index_helper.delete_by_documents(documents)
        await self.delete_filesource_related(documents)

    async def delete_filesource_related(self, documents: list[Document]) -> None:
        # NOTE: File source cleanup still runs sequentially until batch deletion is implemented.

        assert all(self.collection.id == doc.collection_id for doc in documents)  # type: ignore

        for d in documents:
            fs = await FileSourceFactory.create(await d.file_source)
            await fs.delete_file(d.parsed_uri)
            await fs.delete_file(d.uri)
            pages = await DocumentPages.filter(document_id=d.id)
            for page in pages:
                if page.images:
                    for image in page.images:
                        await fs.delete_file(image)

    async def create_from_upload(
        self,
        file_content: bytes,
        file_name: str,
        display_name: str | None,
        file_source: FileSource,
        config_flag: int,
    ) -> Document:
        """
        通过上传文件创建文档

        Args:
            file_content: 文件内容
            file_name: 原始文件名
            display_name: 显示名称（可选）
            file_source: 文件源配置

        Returns:
            创建的 Document 实例
        """
        # 1. 计算文件属性
        file_size = len(file_content)
        extension = Path(file_name).suffix.lower()
        if not extension:
            raise ApiException("上传的文件异常，无法获取文件扩展名")

        # 2. 生成唯一 URI
        unique_filename = f"{uuid.uuid4()}_{file_name}"
        uri = f"{self.collection.id}/{unique_filename}"

        # 3. 计算 MD5 作为 source_version_key
        md5_hash = hashlib.md5(file_content).hexdigest()

        # 4. 上传文件到 file_source
        provider = await FileSourceFactory.create(file_source)
        content_type = mimetypes.guess_type(file_name)[0] or "application/octet-stream"

        metadata = await provider.upload_file(uri, file_content, content_type)
        logger.info(f"Uploaded file to {metadata.uri}, size={file_size}, md5={md5_hash}")

        # 5. 创建 Document 记录
        document = await Document.create(
            collection_id=self.collection.id,
            file_source_id=file_source.id,
            uri=metadata.uri,
            parsed_uri=None,
            file_name=file_name,
            display_name=display_name or file_name,
            extension=extension,
            file_size=file_size,
            source_last_modified=None,
            source_version_key=md5_hash,
            is_deleted_in_source=False,
            source_meta=None,
            short_summary=None,
            long_summary=None,
            status=DocumentStatusEnum.pending.value,
            current_workflow_uid=None,
            config_flag=config_flag,
            workflow_template=self.collection.workflow_template,
        )

        logger.info(f"Created document {document.id} from upload: {file_name}")
        return document

    async def create_from_uri(
        self,
        uri: str,
        display_name: str | None,
        file_source: FileSource,
        config_flag: int,
    ) -> Document:
        """
        通过 URI 创建文档

        Args:
            uri: 文件 URI
            display_name: 显示名称（可选）
            file_source: 文件源配置

        Returns:
            创建的 Document 实例
        """
        is_http_url = uri.startswith(("http://", "https://"))

        if is_http_url:
            # HTTP URL: 跳过 metadata 获取，使用默认值
            file_name = self._extract_filename_from_url(uri)
            extension = "html"
            file_size = None
            source_last_modified = None
            source_version_key = int(time.time())
            source_meta = None
            logger.info(f"Creating document from HTTP URL (metadata skipped): {uri}")
        else:
            # 非 HTTP URL: 从 file_source 获取 metadata
            provider = await FileSourceFactory.create(file_source)
            metadata = await provider.get_file_metadata(uri)

            file_name = metadata.file_name
            extension = Path(file_name).suffix.lower()
            file_size = metadata.file_size
            source_last_modified = metadata.last_modified
            source_version_key = metadata.etag or uri
            source_meta = metadata.extra
            logger.info(f"Creating document from URI with metadata: {uri}, size={file_size}, etag={metadata.etag}")

        # 创建 Document 记录
        document = await Document.create(
            collection_id=self.collection.id,
            file_source_id=file_source.id,
            uri=uri,
            parsed_uri=None,
            file_name=file_name,
            display_name=display_name or file_name,
            extension=extension,
            file_size=file_size,
            source_last_modified=source_last_modified,
            source_version_key=source_version_key,
            is_deleted_in_source=False,
            source_meta=source_meta,
            short_summary=None,
            long_summary=None,
            status=DocumentStatusEnum.pending.value,
            current_workflow_uid=None,
            config_flag=config_flag,
            workflow_template=self.collection.workflow_template,
        )

        logger.info(f"Created document {document.id} from URI: {uri}")
        return document

    def _extract_filename_from_url(self, url: str) -> str:
        """从 URL 中提取文件名"""

        parsed = urlparse(url)
        path = unquote(parsed.path)
        filename = Path(path).name

        if not filename:
            filename = "index.html"

        return url
