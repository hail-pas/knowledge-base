from typing import Annotated
from datetime import datetime

from fastapi import Depends, Request, APIRouter, UploadFile, File, Form
from tortoise.queryset import QuerySet

from core.types import ApiException
from ext.ext_tortoise.enums import DocumentStatusEnum
from core.schema import CRUDPager
from core.response import Resp, PageData
from ext.ext_tortoise.curd import (
    list_view,
    create_obj,
    delete_view,
    update_view,
    pagination_factory,
    obj_prefetch_fields,
)
from ext.ext_tortoise.models.knowledge_base import (
    Document,
    FileSource,
)
from ext.file_source.factory import FileSourceAdapterFactory
from ext.file_source.exceptions import FileSourceNotFoundError

from api.service.document.schema import (
    DocumentList,
    DocumentDetail,
    DocumentCreateByUri,
    DocumentUpdate,
    DocumentFilterSchema,
)

router = APIRouter()


def get_document_queryset(request: Request) -> QuerySet[Document]:
    filter_ = {"deleted_at": 0}
    return Document.filter(**filter_)


@router.get(
    "",
    description=f"{Document.Meta.table_description}列表",
    summary=f"{Document.Meta.table_description}列表",
)
async def get_document_list(
    request: Request,
    filter_: Annotated[DocumentFilterSchema, Depends(DocumentFilterSchema.as_query)],  # type: ignore
    pager: CRUDPager = pagination_factory(
        db_model=Document,
        list_schema=DocumentList,
        search_fields={"file_name", "display_name"},
        order_fields={
            "created_at",
            "file_name",
            "display_name",
        },
        max_limit=1000,
    ),
) -> Resp[PageData[DocumentList]]:
    return await list_view(get_document_queryset(request), filter_, pager)


@router.get(
    "/{pk}",
    description=f"获取{Document.Meta.table_description}详情",
    summary=f"获取{Document.Meta.table_description}详情",
)
async def get_document_detail(request: Request, pk: int) -> Resp[DocumentDetail]:
    queryset = get_document_queryset(request)
    obj: Document | None = await queryset.get_or_none(
        **{queryset.model._meta.pk_attr: pk},
    )
    if not obj:
        raise ApiException("对象不存在")
    obj = await obj_prefetch_fields(obj, DocumentDetail)  # type: ignore
    data = DocumentDetail.model_validate(obj)
    return Resp(data=data)


@router.post(
    "/upload",
    description=f"通过上传文件创建{Document.Meta.table_description}",
    summary=f"通过上传文件创建{Document.Meta.table_description}",
)
async def create_document_by_upload(
    request: Request,
    file: UploadFile = File(..., description="上传的文件"),
    collection_id: int = Form(..., description="关联集合ID"),
    display_name: str | None = Form(None, description="显示名称，默认使用文件名"),
    file_source_id: int | None = Form(None, description="关联文件源ID（可选，不传则使用默认）"),
    status: str = Form(DocumentStatusEnum.pending.value, description="文件状态"),
) -> Resp:
    """通过上传文件创建文档

    用户直接上传文件，系统会：
    1. 读取文件信息（名称、大小、扩展名）
    2. 如果未指定 file_source_id，使用默认文件源
    3. 上传文件到指定的文件源
    4. 创建文档记录
    """
    # 获取文件源
    file_source: FileSource | None = None

    if file_source_id is not None:
        file_source = await FileSource.get_or_none(
            id=file_source_id,
            deleted_at=0,
            is_enabled=True,
        )
        if not file_source:
            raise ApiException("指定的文件源不存在或未启用")
    else:
        # 获取默认文件源
        file_source = await FileSource.filter(
            is_default=True,
            is_enabled=True,
            deleted_at=0,
        ).first()
        if not file_source:
            raise ApiException("未指定文件源且没有可用的默认文件源")

    # 读取文件内容
    file_content = await file.read()
    file_size = len(file_content)

    # 提取文件信息
    file_name = file.filename
    if not file_name:
        raise ApiException("文件名不能为空")
    display_name = display_name or file_name
    extension = file_name.rsplit(".", 1)[-1] if "." in file_name else ""
    # 获取文件源适配器并上传文件
    try:
        adapter = await FileSourceAdapterFactory.create(file_source)

        # 生成URI（使用时间戳和文件名组合）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        uri = f"{timestamp}_{file_name}"

        # 上传文件
        await adapter.upload_file(uri, file_content)

        # 获取文件元数据（如果支持）
        try:
            file_item = await adapter.get_file_meta(uri)
            source_last_modified = file_item.last_modified.isoformat()
            source_version_key = None  # 可能需要从元数据中提取
        except NotImplementedError:
            source_last_modified = None
            source_version_key = None

    except Exception as e:
        raise ApiException(f"文件上传失败: {str(e)}")

    # 准备文档数据
    document_data = {
        "collection_id": collection_id,
        "file_source_id": file_source.id,
        "uri": uri,
        "file_name": file_name,
        "display_name": display_name,
        "extension": extension,
        "file_size": file_size,
        "source_last_modified": source_last_modified,
        "source_version_key": source_version_key,
        "short_summary": "",
        "long_summary": "",
        "status": status,
    }

    await create_obj(Document, document_data)
    return Resp()



@router.post(
    "",
    description=f"通过URI创建{Document.Meta.table_description}",
    summary=f"通过URI创建{Document.Meta.table_description}",
)
async def create_document_by_uri(request: Request, schema: DocumentCreateByUri) -> Resp:
    """通过URI创建文档

    用户提供 file_source_id 和 uri，系统会：
    1. 验证文件源是否存在且已启用
    2. 验证文件在文件源中是否存在
    3. 获取文件元数据（如果支持）
    4. 创建文档记录
    """
    # 获取文件源
    file_source = await FileSource.get_or_none(
        id=schema.file_source_id,
        deleted_at=0,
        is_enabled=True,
    )
    if not file_source:
        raise ApiException("指定的文件源不存在或未启用")

    # 获取文件源适配器并验证文件存在
    try:
        adapter = await FileSourceAdapterFactory.create(file_source)

        # 尝试获取文件元数据来验证文件存在
        try:
            file_item = await adapter.get_file_meta(schema.uri)
            file_size = file_item.size
            source_last_modified = file_item.last_modified.isoformat()
            source_meta = file_item.metadata
        except NotImplementedError:
            # 如果不支持获取元数据，尝试获取文件内容的一部分来验证
            try:
                content = await adapter.get_file(schema.uri)
                file_size = len(content)
                source_last_modified = None
                source_meta = None
            except Exception:
                raise FileSourceNotFoundError(f"文件在文件源中不存在: {schema.uri}")

    except FileSourceNotFoundError:
        raise ApiException(f"文件在文件源中不存在: {schema.uri}")
    except Exception as e:
        raise ApiException(f"访问文件源失败: {str(e)}")

    # 提取文件信息
    file_name = schema.file_name
    display_name = schema.display_name or file_name
    extension = (file_name.rsplit(".", 1)[-1] if "." in file_name else "")

    # 准备文档数据
    document_data = {
        "collection_id": schema.collection_id,
        "file_source_id": file_source.id,
        "uri": schema.uri,
        "file_name": file_name,
        "display_name": display_name,
        "extension": extension,
        "file_size": file_size,
        "source_last_modified": schema.source_last_modified or source_last_modified,
        "source_version_key": schema.source_version_key,
        "short_summary": "",
        "long_summary": "",
        "source_meta": schema.source_meta or source_meta,
        "status": DocumentStatusEnum.pending,
    }

    await create_obj(Document, document_data)
    return Resp()


@router.put(
    "/{pk}",
    description=f"更新{Document.Meta.table_description}",
    summary=f"更新{Document.Meta.table_description}",
)
async def update_document(request: Request, pk: int, schema: DocumentUpdate) -> Resp:
    return await update_view(get_document_queryset(request), pk, schema)  # type: ignore


@router.delete(
    "/{pk}",
    description=f"删除{Document.Meta.table_description}",
    summary=f"删除{Document.Meta.table_description}",
)
async def delete_document(request: Request, pk: int) -> Resp:
    return await delete_view(pk, get_document_queryset(request))
