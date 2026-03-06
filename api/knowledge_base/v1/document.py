from typing import Annotated

from pydantic import BaseModel
from fastapi import APIRouter, Request, Depends, Form, File, UploadFile, Body
from tortoise.queryset import QuerySet
from tortoise.expressions import Q

from service.depend import api_permission_check
from core.schema import CRUDPager
from core.response import Resp, PageData
from core.types import ApiException
from ext.ext_tortoise.curd import (
    create_obj,
    list_view,
    detail_view,
    pagination_factory,
    update_obj,
    DeleteResp,
)
from ext.ext_tortoise.models.knowledge_base import Document, Collection, FileSource
from ext.ext_tortoise.models.user_center import Account
from ext.ext_tortoise.enums import DocumentStatusEnum

from service.document.schema import (
    DocumentList,
    DocumentDetail,
)
from service.document.helper import DocumentService
from service.collection.helper import WorkflowTemplateValidator

router = APIRouter(dependencies=[Depends(api_permission_check)])


def get_document_queryset(request: Request, user: Account) -> QuerySet[Document]:
    """
    Permission-based queryset:
    - Staff: see all
    - Non-staff: see own collections' documents
    """
    queryset = Document.filter(deleted_at=0)

    if not user.is_staff:
        queryset = queryset.filter(collection__user_id=user.id)

    return queryset


@router.post("", summary="创建Document")
async def create_document(
    request: Request,
    collection_id: int = Form(..., description="关联集合ID"),
    file_source_id: int = Form(..., description="文件源ID"),
    file: UploadFile | None = File(None, description="上传的文件"),
    uri: str | None = Form(None, description="文件URI"),
    display_name: str | None = Form(None, description="显示名称"),
    config_flag: int = Form(0, description="")
) -> Resp[DocumentList]:
    """
    创建文档，支持两种方式：
    1. 文件上传：提供 file 参数
    2. URI 指定：提供 uri 参数

    注意：
    - file 和 uri 二选一
    - display_name 未指定则使用原始文件名
    - HTTP URI 跳过 metadata 获取
    - 创建后状态为 pending，不自动触发 workflow
    """
    user: Account = request.scope["user"]

    # 1. 验证 collection
    collection = await Collection.get_or_none(id=collection_id, deleted_at=0).prefetch_related("embedding_model_config")
    if not collection:
        raise ApiException("Collection不存在")

    # 权限检查
    if not user.is_staff and collection.user_id != user.id:
        raise ApiException("无权访问该Collection")

    # 2. 验证 file_source
    file_source = await FileSource.get_or_none(id=file_source_id, is_enabled=True, deleted_at=0)
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
        file_content = await file.read()
        file_name = file.filename

        document = await service.create_from_upload(
            file_content=file_content,
            file_name=file_name,  # type: ignore
            display_name=display_name,
            file_source=file_source,
            config_flag=config_flag
        )
    else:
        # URI 方式
        document = await service.create_from_uri(
            uri=uri,  # type: ignore
            display_name=display_name,
            file_source=file_source,
            config_flag=config_flag
        )

    return Resp(data=DocumentList.model_validate(document))


@router.delete("/{pk}", summary="删除Document")
async def delete_document(request: Request, pk: int) -> Resp[DeleteResp]:
    """
    删除文档
    仅允许删除状态为 success/failed 的文档
    """
    user: Account = request.scope["user"]
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


@router.patch("/{pk}/workflow-template", summary="更新Document工作流模板")
async def update_document_workflow_template(
    request: Request, pk: int, workflow_template: dict = Body(..., description="工作流模板配置")
) -> Resp:
    """
    更新 document 的 workflow_template
    不触发 workflow 执行

    注意：
    - 不验证文档状态
    - 仅更新 workflow_template 字段
    - 使用 WorkflowTemplateValidator 验证格式
    """
    user: Account = request.scope["user"]
    queryset = get_document_queryset(request, user)

    obj = await queryset.get_or_none(pk=pk)
    if not obj:
        raise ApiException("Document不存在")

    if obj.workflow_template == workflow_template:
        return Resp()

    # 验证 workflow_template 格式
    try:
        WorkflowTemplateValidator.validate(workflow_template)
    except ValueError as e:
        raise ApiException(str(e))

    # 更新
    obj.workflow_template = workflow_template
    await obj.save()

    # TODO: 触发 workflow

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
    user: Account = request.scope["user"]
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
    user: Account = request.scope["user"]
    queryset = get_document_queryset(request, user)
    return await detail_view(queryset, pk, DocumentDetail)
