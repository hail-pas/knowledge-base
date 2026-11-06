import mimetypes
from typing import Literal
from urllib.parse import quote

from fastapi import File, Form, Depends, Request, APIRouter, UploadFile
from pydantic import BaseModel
from loguru import logger
from fastapi.responses import StreamingResponse
from tortoise.queryset import QuerySet
from tortoise.expressions import Q

from core.types import ApiException
from core.schema import CRUDPager
from core.response import Resp, PageData
from service.depend import CurrentUser, api_permission_check
from ext.file_source import FileSourceFactory
from ext.ext_tortoise.curd import (
    DeleteResp,
    list_view,
    create_obj,
    update_obj,
    detail_view,
    pagination_factory,
)
from ext.ext_tortoise.enums import DocumentStatusEnum
from service.document.service import DocumentService
from service.document.indexing import DocumentIndexer
from service.file_source.security import validate_public_source_uri
from service.document.schema import (
    DocumentList,
    DocumentDetail,
    DocumentUpdate,
    DocumentPageList,
    DocumentChunkList,
    DocumentChunkCreate,
    DocumentChunkUpdate,
    DocumentGeneratedFaqList,
    DocumentGeneratedFaqCreate,
    DocumentGeneratedFaqUpdate,
)
from ext.workflow.validator import WorkflowTemplateValidator
from ext.ext_tortoise.models.knowledge_base import (
    Document,
    Collection,
    FileSource,
    DocumentChunk,
    DocumentPages,
    DocumentGeneratedFaq,
)

router = APIRouter(dependencies=[Depends(api_permission_check)])

MAX_UPLOAD_SIZE = 100 * 1024 * 1024


def get_document_queryset(request: Request, user: CurrentUser) -> QuerySet[Document]:
    """
    Permission-based queryset:
    - Staff: see all
    - Non-staff: see own collections' documents
    """
    queryset = Document.filter(deleted_at=0)

    queryset = queryset.filter(collection__user_id=user.id)

    return queryset


async def get_document_or_raise(request: Request, pk: int) -> Document:
    user: CurrentUser = request.scope["user"]
    queryset = get_document_queryset(request, user)
    obj = await queryset.prefetch_related("file_source", "collection").get_or_none(pk=pk)
    if not obj:
        raise ApiException("Document不存在")
    validate_public_source_uri(obj.file_source, obj.collection, obj.uri)
    if obj.parsed_uri:
        validate_public_source_uri(obj.file_source, obj.collection, obj.parsed_uri)
    return obj


@router.post("", summary="创建Document")
async def create_document(
    request: Request,
    collection_id: int = Form(..., description="关联集合ID"),
    file_source_id: int = Form(..., description="文件源ID"),
    file: UploadFile | None = File(None, description="上传的文件"),
    uri: str | None = Form(None, description="文件URI"),
    display_name: str | None = Form(None, description="显示名称"),
    config_flag: int = Form(0, description=""),
) -> Resp[DocumentList]:
    """
    创建文档，支持两种方式：
    1. 文件上传：提供 file 参数
    2. URI 指定：提供 uri 参数

    注意：
    - file 和 uri 二选一
    - display_name 未指定则使用原始文件名
    - HTTP URI 跳过 metadata 获取
    - 创建后自动启动文档处理 workflow
    """
    user: CurrentUser = request.scope["user"]

    # 1. 验证 collection
    collection = await Collection.get_or_none(id=collection_id, deleted_at=0).prefetch_related(
        "embedding_model_config",
    )
    if not collection:
        raise ApiException("Collection不存在")

    # 权限检查
    if collection.user_id != user.id:
        raise ApiException("无权访问该Collection")

    # 2. 验证 file_source
    file_source = await FileSource.filter(
        Q(user_id=None) | Q(user_id=user.id),
        id=file_source_id,
        is_enabled=True,
        deleted_at=0,
    ).get_or_none()
    if not file_source:
        raise ApiException("文件源不存在或未启用")

    # 3. 验证 file 和 uri 二选一
    if (file is None) == (uri is None):
        raise ApiException("必须提供 file 或 uri 之一，且不能同时提供")

    if file and not file.filename:
        raise ApiException("上传的文件异常，无法获取文件名")

    # 4. 创建文档
    service = DocumentService(collection)

    if file is not None:
        # 文件上传方式
        file_content = await file.read(MAX_UPLOAD_SIZE + 1)
        if len(file_content) > MAX_UPLOAD_SIZE:
            raise ApiException("上传文件不能超过 100 MiB", code=413)
        file_name = file.filename or ""

        document = await service.create_from_upload(
            file_content=file_content,
            file_name=file_name,  # type: ignore
            display_name=display_name,
            file_source=file_source,
            config_flag=config_flag,
        )
    else:
        # URI 方式
        document = await service.create_from_uri(
            uri=uri,  # type: ignore
            display_name=display_name,
            file_source=file_source,
            config_flag=config_flag,
        )

    await service.start_processing(document)

    return Resp(data=DocumentList.model_validate(document))


@router.delete("/{pk}", summary="删除Document")
async def delete_document(request: Request, pk: int) -> Resp[DeleteResp]:
    """
    删除文档
    仅允许删除状态为 success/failed 的文档
    """
    user: CurrentUser = request.scope["user"]
    queryset = get_document_queryset(request, user)

    obj = await queryset.prefetch_related("collection__embedding_model_config").get_or_none(pk=pk)
    if not obj:
        raise ApiException("Document不存在")

    # 验证状态
    if obj.status not in [DocumentStatusEnum.success.value, DocumentStatusEnum.failure.value]:
        raise ApiException("当前状态不支持删除")

    # 获取 collection 并删除
    collection = obj.collection
    service = DocumentService(collection)
    await service.delete([obj])

    return Resp(data=DeleteResp(deleted=1))


@router.put("/{pk}", summary="更新Document")
async def update_document(
    request: Request,
    pk: int,
    schema: DocumentUpdate,
) -> Resp:
    user: CurrentUser = request.scope["user"]
    queryset = get_document_queryset(request, user)

    obj = await queryset.get_or_none(pk=pk)
    if not obj:
        raise ApiException("Document不存在")

    update_data = schema.model_dump(exclude_unset=True)

    if "workflow_template" in update_data:
        # 验证 workflow_template 格式
        try:
            WorkflowTemplateValidator.validate(update_data["workflow_template"])
        except ValueError as e:
            raise ApiException(str(e)) from e

    await update_obj(obj, queryset, update_data)

    return Resp()


@router.get("", summary="Document列表")
async def list_documents(
    request: Request,
    collection_id: int | None = None,
    status: str | None = None,
    pager: CRUDPager = pagination_factory(
        db_model=Document,
        search_fields={"file_name", "display_name"},
        order_fields={"id", "created_at", "status"},
        list_schema=DocumentList,
        max_limit=100,
    ),
) -> Resp[PageData[DocumentList]]:
    user: CurrentUser = request.scope["user"]
    queryset = get_document_queryset(request, user)

    # 应用过滤
    if collection_id is not None:
        queryset = queryset.filter(collection_id=collection_id)
    if status is not None:
        queryset = queryset.filter(status=status)

    # 构造 filter 对象
    class FilterSchema(BaseModel):
        pass

    filter_obj = FilterSchema()

    return await list_view(queryset, filter_obj, pager)


@router.get("/{pk}", summary="Document详情")
async def get_document_detail(request: Request, pk: int) -> Resp[DocumentDetail]:
    user: CurrentUser = request.scope["user"]
    queryset = get_document_queryset(request, user)
    return await detail_view(queryset, pk, DocumentDetail)


@router.post("/{pk}/re-process", summary="Document 重新处理")
async def re_process_document(request: Request, pk: int) -> Resp[dict[Literal["workflow_id"], str]]:
    user: CurrentUser = request.scope["user"]
    document = await get_document_queryset(request, user).get_or_none(pk=pk)
    if not document:
        raise ApiException("Document不存在")
    collection = await document.collection
    workflow_id = await DocumentService(collection).reprocess(document)
    return Resp(data={"workflow_id": workflow_id})


@router.post("/{pk}/re-chunk", summary="Document 重新切块")
async def re_chunk_document(request: Request, pk: int) -> Resp[dict[Literal["workflow_id"], str]]:
    user: CurrentUser = request.scope["user"]
    document = await get_document_queryset(request, user).get_or_none(pk=pk)
    if not document:
        raise ApiException("Document不存在")
    collection = await document.collection
    workflow_id = await DocumentService(collection).rechunk(document)
    return Resp(data={"workflow_id": workflow_id})


@router.get(
    "/{pk}/stream",
    summary="Document文件流",
    response_class=StreamingResponse,
)
async def get_document_stream(request: Request, pk: int) -> StreamingResponse:
    document = await get_document_or_raise(request, pk)
    provider = await FileSourceFactory.create(document.file_source)

    media_type = mimetypes.guess_type(document.file_name)[0] or "application/octet-stream"
    filename = quote(document.file_name)

    return StreamingResponse(
        content=provider.get_file_stream(document.uri),
        media_type=media_type,
        headers={"Content-Disposition": f"inline; filename*=UTF-8''{filename}"},
    )


@router.get("/{pk}/pages", summary="DocumentPages列表")
async def list_document_pages(
    request: Request,
    pk: int,
    page_number: int | None = None,
) -> Resp[list[DocumentPageList]]:
    document = await get_document_or_raise(request, pk)

    pages_queryset = DocumentPages.filter(document_id=document.id, deleted_at=0).order_by("page_number")
    if page_number is not None:
        pages_queryset = pages_queryset.filter(page_number=page_number)
    pages = await pages_queryset

    if document.parsed_uri:
        for page in pages:
            if not page.content and page.page_number == 1:
                provider = await FileSourceFactory.create(document.file_source)
                content_bytes = await provider.get_file(document.parsed_uri)
                page.content = content_bytes.decode("utf-8", errors="ignore")

    return Resp(data=[DocumentPageList.model_validate(page) for page in pages])


@router.get("/{pk}/chunks", summary="DocumentChunk列表")
async def list_document_chunks(
    request: Request,
    pk: int,
    page_number: int | None = None,
) -> Resp[list[DocumentChunkList]]:
    document = await get_document_or_raise(request, pk)

    chunks_queryset = DocumentChunk.filter(document_id=document.id, deleted_at=0).order_by("-id")
    if page_number is not None:
        chunks_queryset = chunks_queryset.filter(min_page__lte=page_number, max_page__gte=page_number)
    chunks = await chunks_queryset

    return Resp(data=[DocumentChunkList.model_validate(chunk) for chunk in chunks])


@router.post("/{pk}/chunks", summary="新增DocumentChunk")
async def create_document_chunk(request: Request, pk: int, schema: DocumentChunkCreate) -> Resp[DocumentChunkList]:
    document = await get_document_or_raise(request, pk)

    if DocumentStatusEnum(document.status) not in {DocumentStatusEnum.success, DocumentStatusEnum.failure}:
        raise ApiException("仅允许对已完成或失败的文档进行手动切块操作", code=409)

    await document.collection.fetch_related("embedding_model_config", "indexing_backend_config")
    min_page = min(schema.pages)
    max_page = max(schema.pages)
    obj = await create_obj(
        DocumentChunk,
        {
            "document_id": document.id,
            "content": schema.content,
            "pages": schema.pages,
            "min_page": min_page,
            "max_page": max_page,
            "start": schema.start,
            "end": schema.end,
            "overlap_start": schema.overlap_start,
            "overlap_end": schema.overlap_end,
            "metadata": schema.metadata,
            "manual_add": schema.manual_add,
        },
    )
    try:
        await DocumentIndexer(document.collection).index_chunks(document, [obj])
    except Exception:
        await obj.delete()
        raise

    return Resp(data=DocumentChunkList.model_validate(obj))


@router.put("/{pk}/chunks/{chunk_id}", summary="修改DocumentChunk")
async def update_document_chunk(
    request: Request,
    pk: int,
    chunk_id: int,
    schema: DocumentChunkUpdate,
) -> Resp[DocumentChunkList]:
    document = await get_document_or_raise(request, pk)
    if DocumentStatusEnum(document.status) not in {DocumentStatusEnum.success, DocumentStatusEnum.failure}:
        raise ApiException("仅允许对已完成或失败的文档进行手动切块操作", code=409)

    chunk = await DocumentChunk.filter(
        id=chunk_id,
        document_id=document.id,
        deleted_at=0,
    ).get_or_none()
    if chunk is None:
        raise ApiException("DocumentChunk不存在", code=404)

    await document.collection.fetch_related("embedding_model_config", "indexing_backend_config")
    old_values = {
        "content": chunk.content,
        "pages": chunk.pages,
        "min_page": chunk.min_page,
        "max_page": chunk.max_page,
        "start": chunk.start,
        "end": chunk.end,
        "overlap_start": chunk.overlap_start,
        "overlap_end": chunk.overlap_end,
        "metadata": chunk.metadata,
        "manual_add": chunk.manual_add,
    }
    updates = schema.model_dump(exclude_unset=True)
    if not updates:
        return Resp(data=DocumentChunkList.model_validate(chunk))

    new_values = {**old_values, **updates}
    if "pages" in updates:
        if not updates["pages"]:
            raise ApiException("pages 不能为空", code=422)
        new_values["min_page"] = min(new_values["pages"])
        new_values["max_page"] = max(new_values["pages"])

    indexer = DocumentIndexer(document.collection)
    await indexer.delete_chunks(document.id, [chunk.id])
    try:
        for field, value in new_values.items():
            setattr(chunk, field, value)
        await chunk.save()
        await indexer.index_chunks(document, [chunk])
    except Exception:
        for field, value in old_values.items():
            setattr(chunk, field, value)
        await chunk.save()
        try:
            await indexer.delete_chunks(document.id, [chunk.id])
            await indexer.index_chunks(document, [chunk])
        except Exception:
            logger.exception("Failed to restore chunk index after update rollback: document={}, chunk={}",
                             document.id, chunk.id)
        raise

    return Resp(data=DocumentChunkList.model_validate(chunk))


@router.delete("/{pk}/chunks/{chunk_id}", summary="删除DocumentChunk")
async def delete_document_chunk(request: Request, pk: int, chunk_id: int) -> Resp[DeleteResp]:
    document = await get_document_or_raise(request, pk)
    if DocumentStatusEnum(document.status) not in {DocumentStatusEnum.success, DocumentStatusEnum.failure}:
        raise ApiException("仅允许对已完成或失败的文档进行手动切块操作", code=409)

    chunk = await DocumentChunk.filter(
        id=chunk_id,
        document_id=document.id,
        deleted_at=0,
    ).get_or_none()
    if chunk is None:
        return Resp(data=DeleteResp(deleted=0))

    await document.collection.fetch_related("embedding_model_config", "indexing_backend_config")
    await DocumentIndexer(document.collection).delete_chunks(document.id, [chunk.id])
    deleted = await DocumentChunk.filter(
        id=chunk.id,
        document_id=document.id,
        deleted_at=0,
    ).delete()
    return Resp(data=DeleteResp(deleted=deleted))


@router.get("/{pk}/faqs", summary="DocumentGeneratedFaq列表")
async def list_document_generated_faqs(
    request: Request,
    pk: int,
) -> Resp[list[DocumentGeneratedFaqList]]:
    document = await get_document_or_raise(request, pk)

    faqs_queryset = DocumentGeneratedFaq.filter(document_id=document.id, deleted_at=0).order_by("-id")
    faqs = await faqs_queryset

    return Resp(data=[DocumentGeneratedFaqList.model_validate(faq) for faq in faqs])


@router.post("/{pk}/faqs", summary="新增DocumentGeneratedFaq")
async def create_document_generated_faq(
    request: Request,
    pk: int,
    schema: DocumentGeneratedFaqCreate,
) -> Resp[DocumentGeneratedFaqList]:
    document = await get_document_or_raise(request, pk)
    obj = await create_obj(
        DocumentGeneratedFaq,
        {
            "document_id": document.id,
            "content": schema.content,
            "question": schema.question,
            "answer": schema.answer,
            "manual_add": schema.manual_add,
            "enabled": schema.enabled,
        },
    )
    return Resp(data=DocumentGeneratedFaqList.model_validate(obj))


@router.put("/{pk}/faqs/{faq_id}", summary="修改DocumentGeneratedFaq")
async def update_document_generated_faq(
    request: Request,
    pk: int,
    faq_id: int,
    schema: DocumentGeneratedFaqUpdate,
) -> Resp:
    await get_document_or_raise(request, pk)
    queryset = DocumentGeneratedFaq.filter(document_id=pk, deleted_at=0)
    obj = await queryset.get_or_none(id=faq_id)
    if not obj:
        raise ApiException("DocumentGeneratedFaq不存在")
    await update_obj(obj, queryset, schema.model_dump(exclude_unset=True))
    return Resp()


@router.delete("/{pk}/faqs/{faq_id}", summary="删除DocumentGeneratedFaq")
async def delete_document_generated_faq(request: Request, pk: int, faq_id: int) -> Resp[DeleteResp]:
    await get_document_or_raise(request, pk)
    deleted = await DocumentGeneratedFaq.filter(id=faq_id, document_id=pk, deleted_at=0).delete()
    return Resp(data=DeleteResp(deleted=deleted))
