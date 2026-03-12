from typing import Annotated

from fastapi import Depends, Request, APIRouter
from tortoise.queryset import QuerySet
from tortoise.expressions import Q

from core.types import ApiException
from core.schema import CRUDPager
from core.response import Resp, PageData
from service.depend import api_permission_check
from ext.ext_tortoise.curd import (
    DeleteResp,
    list_view,
    create_obj,
    update_obj,
    detail_view,
    pagination_factory,
)
from service.document.helper import DocumentService
from service.collection.helper import CollectionService
from service.collection.schema import (
    CollectionList,
    CollectionCreate,
    CollectionDetail,
    CollectionUpdate,
    CollectionFilterSchema,
)
from ext.ext_tortoise.models.user_center import Account
from ext.ext_tortoise.models.knowledge_base import (
    Document,
    Collection,
    EmbeddingModelConfig,
)

router = APIRouter(dependencies=[Depends(api_permission_check)])


def get_collection_queryset(request: Request, user: Account) -> QuerySet[Collection]:
    """
    Permission-based queryset:
    - Staff: see all
    - Non-staff: see own + public
    """
    queryset = Collection.filter(deleted_at=0)

    if not user.is_staff:
        queryset = queryset.filter(Q(user_id=None) | Q(user_id=user.id) | Q(is_public=True))

    return queryset


@router.post("", summary="创建Collection")
async def create_collection(request: Request, schema: CollectionCreate) -> Resp[CollectionList]:
    schema_dict = schema.model_dump(exclude_unset=True)

    obj: Collection = await create_obj(Collection, schema_dict)  # type: ignore

    await obj.fetch_related("embedding_model_config")

    return Resp(data=CollectionList.model_validate(obj))


@router.put("/{pk}", summary="更新Collection")
async def update_collection(request: Request, pk: int, schema: CollectionUpdate) -> Resp:
    user: Account = request.scope["user"]
    queryset = get_collection_queryset(request, user)

    obj = await queryset.get_or_none(pk=pk)
    if not obj:
        raise ApiException("Collection不存在")

    update_data = schema.model_dump(exclude_unset=True)

    if "embedding_model_config_id" in update_data:
        new_config_id = update_data["embedding_model_config_id"]
        new_config = await EmbeddingModelConfig.get_or_none(id=new_config_id, deleted_at=0)
        if not new_config:
            raise ApiException("Embedding模型配置不存在")

        del update_data["embedding_model_config_id"]
        old_config_id = obj.embedding_model_config_id  # type: ignore

        if update_data:
            await update_obj(obj, queryset, update_data)

        service = CollectionService(obj)
        try:
            await service.switch_embedding_model(new_config)
        except Exception as e:
            await queryset.filter(pk=pk).update(embedding_model_config_id=old_config_id)
            raise ApiException(f"切换Embedding模型失败: {str(e)}") from e
    else:
        await update_obj(obj, queryset, update_data)

    return Resp()


@router.get("", summary="Collection列表")
async def list_collections(
    request: Request,
    filter_: Annotated[CollectionFilterSchema, Depends(CollectionFilterSchema.as_query)],  # type: ignore
    pager: CRUDPager = pagination_factory(
        db_model=Collection,
        search_fields={"name"},
        order_fields={"id", "created_at"},
        list_schema=CollectionList,
        max_limit=100,
    ),
) -> Resp[PageData[CollectionList]]:
    user: Account = request.scope["user"]
    queryset = get_collection_queryset(request, user)
    return await list_view(queryset, filter_, pager)


@router.get("/{pk}", summary="Collection详情")
async def get_collection_detail(request: Request, pk: int) -> Resp[CollectionDetail]:
    user: Account = request.scope["user"]
    queryset = get_collection_queryset(request, user)
    return await detail_view(queryset, pk, CollectionDetail)


@router.delete("/{pk}", summary="删除Collection")
async def delete_collection(request: Request, pk: int) -> Resp[DeleteResp]:
    user: Account = request.scope["user"]
    queryset = get_collection_queryset(request, user)

    obj = await queryset.prefetch_related("embedding_model_config").get_or_none(pk=pk)
    if not obj:
        raise ApiException("Collection不存在")

    if obj.is_external:
        deleted = await queryset.filter(pk=pk).delete()
        return Resp(data=DeleteResp(deleted=deleted))

    service = CollectionService(obj)
    can_delete = await service.can_delete_collection()
    if not can_delete:
        raise ApiException("Collection下有未完成的文档，无法删除")

    documents = await Document.filter(collection_id=pk, deleted_at=0)
    document_ids = [doc.id for doc in documents]

    if document_ids:
        doc_service = DocumentService(obj)
        await doc_service.delete(documents)

    await service.collection_index_helper.delete_by_collection()

    deleted = await queryset.filter(pk=pk).delete()

    return Resp(data=DeleteResp(deleted=deleted))
