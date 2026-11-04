from typing import Annotated

from fastapi import Depends, Query, Request, APIRouter
from tortoise.queryset import QuerySet
from tortoise.expressions import Q
from tortoise.functions import Count

from core.types import ApiException
from core.schema import CRUDPager
from core.response import Resp, PageData
from service.depend import CurrentUser, api_permission_check
from ext.ext_tortoise.curd import (
    DeleteResp,
    list_view,
    create_obj,
    update_obj,
    pagination_factory,
)
from service.document.helper import DocumentService
from service.collection.helper import CollectionService
from util.secret import encrypt_secret
from service.collection.schema import (
    CollectionList,
    CollectionCreate,
    CollectionDetail,
    CollectionUpdate,
    CollectionFilterSchema,
)
from ext.ext_tortoise.models.knowledge_base import (
    Document,
    Collection,
    EmbeddingModelConfig,
)

router = APIRouter(dependencies=[Depends(api_permission_check)])


async def _enrich_collections_with_doc_count(data: list[CollectionList]) -> list[CollectionList]:
    if not data:
        return data

    ids = [item.id for item in data]  # type: ignore
    if not ids:
        return data
    counts = (
        await Document.filter(
            collection_id__in=ids,
            deleted_at=0,
        )
        .annotate(count=Count("id"))
        .group_by("collection_id")
        .values("collection_id", "count")
    )
    count_map = {row["collection_id"]: row["count"] for row in counts}
    for d in data:
        d.document_count = count_map.get(d.id, 0)  # type: ignore
    return data


def get_collection_queryset(
    request: Request,
    user: CurrentUser,
    *,
    writable: bool = False,
) -> QuerySet[Collection]:
    """Return collections visible to this identity.

    Public and legacy ownerless collections are readable; mutations are restricted
    to the collection owner.
    """
    queryset = Collection.filter(deleted_at=0)
    if writable:
        return queryset.filter(user_id=user.id)
    return queryset.filter(Q(user_id=None) | Q(user_id=user.id) | Q(is_public=True))


@router.post("", summary="创建Collection")
async def create_collection(request: Request, schema: CollectionCreate) -> Resp[CollectionList]:
    user: CurrentUser = request.scope["user"]
    schema_dict = schema.model_dump(exclude_unset=True)
    schema_dict["user_id"] = user.id
    external_config = schema_dict.get("external_config")
    if external_config and external_config.get("authorization"):
        external_config["authorization"] = encrypt_secret(
            external_config["authorization"],
            purpose="collection.external_config.authorization",
        )

    obj: Collection = await create_obj(Collection, schema_dict)  # type: ignore

    await obj.fetch_related("embedding_model_config")

    return Resp(data=CollectionList.model_validate(obj))


@router.put("/{pk}", summary="更新Collection")
async def update_collection(request: Request, pk: int, schema: CollectionUpdate) -> Resp:
    user: CurrentUser = request.scope["user"]
    queryset = get_collection_queryset(request, user, writable=True)

    obj = await queryset.get_or_none(pk=pk)
    if not obj:
        raise ApiException("Collection不存在")

    update_data = schema.model_dump(exclude_unset=True)
    external_config = update_data.get("external_config")
    if external_config and external_config.get("authorization"):
        external_config["authorization"] = encrypt_secret(
            external_config["authorization"],
            purpose="collection.external_config.authorization",
        )

    if obj.is_external:
        if "workflow_template" in update_data:
            raise ApiException("外部知识库不能配置 workflow_template")
        if "external_config" in update_data and not update_data["external_config"]:
            raise ApiException("外部知识库必须保留 external_config")
    elif "external_config" in update_data and update_data["external_config"] is not None:
        raise ApiException("普通知识库不能设置 external_config")

    if "embedding_model_config_id" in update_data:
        new_config_id = update_data["embedding_model_config_id"]
        new_config = await EmbeddingModelConfig.get_or_none(
            id=new_config_id,
            deleted_at=0,
            is_enabled=True,
        )
        if not new_config:
            raise ApiException("Embedding模型配置不存在或已停用")

        if new_config_id != obj.embedding_model_config_id and await Document.filter(
            collection_id=pk,
            deleted_at=0,
        ).exists():
            raise ApiException(
                "非空集合暂不支持切换 Embedding 模型，请新建集合并重新导入文档",
                code=409,
            )

    await update_obj(obj, queryset, update_data)

    return Resp()


@router.get("", summary="Collection列表")
async def list_collections(
    request: Request,
    filter_: Annotated[CollectionFilterSchema, Query()],
    pager: CRUDPager = pagination_factory(
        db_model=Collection,
        search_fields={"name"},
        order_fields={"id", "created_at"},
        list_schema=CollectionList,
        max_limit=100,
    ),
) -> Resp[PageData[CollectionList]]:
    user: CurrentUser = request.scope["user"]
    queryset = get_collection_queryset(request, user)
    return await list_view(queryset, filter_, pager, post_process=_enrich_collections_with_doc_count)


@router.get("/{pk}", summary="Collection详情")
async def get_collection_detail(request: Request, pk: int) -> Resp[CollectionDetail]:
    user: CurrentUser = request.scope["user"]
    queryset = get_collection_queryset(request, user)
    obj = await queryset.prefetch_related("embedding_model_config").get_or_none(pk=pk)
    if not obj:
        raise ApiException("Collection不存在")

    data = CollectionDetail.model_validate(obj)
    if obj.user_id != user.id:
        data.workflow_template = {}
        data.embedding_model_config = None
    return Resp(data=data)


@router.delete("/{pk}", summary="删除Collection")
async def delete_collection(request: Request, pk: int) -> Resp[DeleteResp]:
    user: CurrentUser = request.scope["user"]
    queryset = get_collection_queryset(request, user, writable=True)

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
