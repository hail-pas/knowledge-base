import time
import uuid
import hashlib
import mimetypes
from pathlib import Path
from urllib.parse import unquote, urlparse

from loguru import logger

from core.exception import ApiException
from tortoise.transactions import in_transaction
from ext.ext_tortoise.main import ConnectionNameEnum
from ext.workflow.manager import WorkflowManager
from ext.workflow import schedule_workflow
from ext.ext_tortoise.enums import DocumentStatusEnum
from ext.file_source.factory import FileSourceFactory
from service.file_source.security import build_file_upload_uri, validate_public_source_uri
from service.document.indexing import DocumentIndexer
from ext.ext_tortoise.models.knowledge_base import (
    Document,
    FileSource,
    DocumentChunk,
    DocumentPages,
    DocumentGeneratedFaq,
    Collection,
)


class DocumentService:
    def __init__(self, collection: Collection) -> None:
        self.collection = collection

    async def start_processing(self, document: Document) -> str:
        async with in_transaction(connection_name=ConnectionNameEnum.knowledge_base.value) as conn:
            locked = await Document.filter(
                id=document.id,
                collection_id=self.collection.id,
            ).using_db(conn).select_for_update().first()
            if locked is None:
                raise ApiException("Document不存在")
            if locked.current_workflow_uid:
                raise ApiException("文档已关联工作流，不能重复启动", code=409)
            if locked.status != DocumentStatusEnum.pending.value:
                raise ApiException("当前文档状态不能启动处理", code=409)

            workflow = await WorkflowManager.create_workflow(
                config=locked.workflow_template,
                initial_inputs={"document_id": locked.id},
            )
            locked.current_workflow_uid = workflow.uid
            await locked.save(using_db=conn, update_fields=["current_workflow_uid"])

        document.current_workflow_uid = workflow.uid
        return await schedule_workflow(workflow_uid=str(workflow.uid), execute_mode="celery")

    async def reprocess(self, document: Document) -> str:
        async with in_transaction(connection_name=ConnectionNameEnum.knowledge_base.value) as conn:
            locked = await Document.filter(
                id=document.id,
                collection_id=self.collection.id,
            ).using_db(conn).select_for_update().first()
            if locked is None:
                raise ApiException("Document不存在")
            if locked.status not in {DocumentStatusEnum.success.value, DocumentStatusEnum.failure.value}:
                raise ApiException("当前状态不支持重新处理")

            old_workflow_uid = locked.current_workflow_uid
            workflow = await WorkflowManager.create_workflow(
                config=locked.workflow_template,
                initial_inputs={"document_id": locked.id},
            )
            locked.status = DocumentStatusEnum.pending
            locked.current_workflow_uid = workflow.uid
            await locked.save(using_db=conn, update_fields=["status", "current_workflow_uid"])
            if old_workflow_uid:
                await WorkflowManager.delete_workflow(old_workflow_uid, using_db=conn)

        return await schedule_workflow(workflow_uid=str(workflow.uid), execute_mode="celery")

    async def rechunk(self, document: Document) -> str:
        async with in_transaction(connection_name=ConnectionNameEnum.knowledge_base.value) as conn:
            locked = await Document.filter(
                id=document.id,
                collection_id=self.collection.id,
            ).using_db(conn).select_for_update().first()
            if locked is None:
                raise ApiException("Document不存在")
            if locked.status not in {DocumentStatusEnum.success.value, DocumentStatusEnum.failure.value}:
                raise ApiException("当前状态不支持")

            workflow_uid = locked.current_workflow_uid
            if not workflow_uid:
                raise ApiException("Document 缺少可重新切块的工作流", code=409)

            reset_count = await WorkflowManager.reset_activities_from_task(
                workflow_uid,
                "workflow_document.DocumentChunkTask",
                using_db=conn,
            )
            if not reset_count:
                raise ApiException("当前工作流没有 DocumentChunkTask，无法重新切块", code=409)

            locked.status = DocumentStatusEnum.pending
            await locked.save(using_db=conn, update_fields=["status"])

        return await schedule_workflow(workflow_uid=str(workflow_uid), execute_mode="celery")

    async def delete(self, documents: list[Document]) -> None:
        assert all(self.collection.id == doc.collection_id for doc in documents)  # type: ignore

        document_ids = [doc.id for doc in documents]
        await DocumentIndexer(self.collection).delete_documents(documents)
        await self.delete_filesource_related(documents)

        async with in_transaction(connection_name=ConnectionNameEnum.knowledge_base.value) as conn:
            await DocumentChunk.filter(document_id__in=document_ids).using_db(conn).delete()
            await DocumentPages.filter(document_id__in=document_ids).using_db(conn).delete()
            await DocumentGeneratedFaq.filter(document_id__in=document_ids).using_db(conn).delete()
            await Document.filter(id__in=document_ids).using_db(conn).delete()

    async def delete_filesource_related(self, documents: list[Document]) -> None:
        """Delete Knowledge Base managed artifacts without deleting imported source files."""
        assert all(self.collection.id == doc.collection_id for doc in documents)  # type: ignore

        document_ids = [doc.id for doc in documents]
        pages = await DocumentPages.filter(document_id__in=document_ids, deleted_at=0)
        images_by_document: dict[int, list[str]] = {}
        for page in pages:
            for image in page.images or []:
                if isinstance(image, str):
                    image_uri = image
                elif isinstance(image, dict):
                    image_uri = image.get("uri") or image.get("key")
                else:
                    continue
                if isinstance(image_uri, str) and image_uri:
                    images_by_document.setdefault(page.document_id, []).append(image_uri)

        for document in documents:
            image_uris = images_by_document.get(document.id, [])
            managed_upload = (document.source_meta or {}).get("_knowledge_base", {}).get("managed_upload", False)
            if not document.parsed_uri and not managed_upload and not image_uris:
                continue

            file_source = await document.file_source
            validate_public_source_uri(file_source, self.collection, document.uri)
            if document.parsed_uri:
                validate_public_source_uri(file_source, self.collection, document.parsed_uri)
            for image_uri in image_uris:
                validate_public_source_uri(file_source, self.collection, image_uri)

            provider = await FileSourceFactory.create(file_source)
            if document.parsed_uri:
                await provider.delete_file(document.parsed_uri)
            if managed_upload:
                await provider.delete_file(document.uri)
            for image_uri in image_uris:
                await provider.delete_file(image_uri)

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
        # Strip client-supplied path components before constructing a storage key.
        file_name = file_name.replace("\\", "/").rsplit("/", 1)[-1].strip()
        if file_name in {"", ".", ".."} or "\x00" in file_name:
            raise ApiException("上传文件名无效")

        # 1. 计算文件属性
        file_size = len(file_content)
        extension = Path(file_name).suffix.lower()
        if not extension:
            raise ApiException("上传的文件异常，无法获取文件扩展名")

        # 2. 生成唯一 URI
        unique_filename = f"{uuid.uuid4()}_{file_name}"
        uri = build_file_upload_uri(file_source, self.collection, unique_filename)

        # 3. Use a collision-resistant fingerprint for source version tracking.
        content_hash = hashlib.sha256(file_content).hexdigest()

        # 4. 上传文件到 file_source
        provider = await FileSourceFactory.create(file_source)
        content_type = mimetypes.guess_type(file_name)[0] or "application/octet-stream"

        metadata = await provider.upload_file(uri, file_content, content_type)
        logger.info(f"Uploaded file to {metadata.uri}, size={file_size}, sha256={content_hash}")

        # 5. 创建 Document 记录
        try:
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
                source_version_key=content_hash,
                is_deleted_in_source=False,
                source_meta={"_knowledge_base": {"managed_upload": True}},
                short_summary=None,
                long_summary=None,
                status=DocumentStatusEnum.pending.value,
                current_workflow_uid=None,
                config_flag=config_flag,
                workflow_template=self.collection.workflow_template,
            )
        except BaseException:
            try:
                await provider.delete_file(metadata.uri)
            except Exception:
                logger.exception("Failed to rollback uploaded source file {}", metadata.uri)
            raise

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
        validate_public_source_uri(file_source, self.collection, uri)
        is_http_url = uri.lower().startswith(("http://", "https://"))

        if is_http_url:
            # HTTP URL: 跳过 metadata 获取，使用默认值
            file_name = self._extract_filename_from_url(uri)
            extension = Path(file_name).suffix.lower() or ".html"
            file_size = None
            source_last_modified = None
            source_version_key = int(time.time())
            source_meta = None
            logger.info(f"Creating document from HTTP URL (metadata skipped): {self._safe_url_for_log(uri)}")
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
            logger.info(f"Creating document from URI with metadata: file_name={file_name}, size={file_size}")

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

        logger.info(f"Created document {document.id} from URI")
        return document

    @staticmethod
    def _safe_url_for_log(url: str) -> str:
        """Redact query parameters and fragments that may contain access tokens."""
        return urlparse(url)._replace(query="", fragment="").geturl()

    def _extract_filename_from_url(self, url: str) -> str:
        """从 URL 中提取文件名"""
        path = unquote(urlparse(url).path)
        filename = Path(path).name
        return filename if filename not in {"", ".", ".."} else "index.html"
