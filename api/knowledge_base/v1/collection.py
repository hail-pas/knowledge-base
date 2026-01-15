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
    Collection,
    Document,
    FileSource,
)
from ext.file_source.factory import FileSourceAdapterFactory
from ext.file_source.exceptions import FileSourceNotFoundError

from api.service.collection.schema import (
    CollectionList,
    CollectionCreate,
    CollectionDetail,
    CollectionUpdate,
    CollectionFilterSchema
)

router = APIRouter()


# ============ Collection CRUD ============


def get_collection_queryset(request: Request) -> QuerySet[Collection]:
    filter_ = {"deleted_at": 0}
    return Collection.filter(**filter_)


@router.get(
    "",
    description=f"{Collection.Meta.table_description}列表",
    summary=f"{Collection.Meta.table_description}列表",
)
async def get_collection_list(
    request: Request,
    filter_: Annotated[CollectionFilterSchema, Depends(CollectionFilterSchema.as_query)],  # type: ignore
    pager: CRUDPager = pagination_factory(
        db_model=Collection,
        list_schema=CollectionList,
        search_fields={"name", "description"},
        order_fields={
            "created_at",
            "name",
        },
        max_limit=1000,
    ),
) -> Resp[PageData[CollectionList]]:
    return await list_view(get_collection_queryset(request), filter_, pager)


@router.get(
    "/{pk}",
    description=f"获取{Collection.Meta.table_description}详情",
    summary=f"获取{Collection.Meta.table_description}详情",
)
async def get_collection_detail(request: Request, pk: int) -> Resp[CollectionDetail]:
    queryset = get_collection_queryset(request)
    obj: Collection | None = await queryset.get_or_none(
        **{queryset.model._meta.pk_attr: pk},
    )
    if not obj:
        raise ApiException("对象不存在")
    obj = await obj_prefetch_fields(obj, CollectionDetail)  # type: ignore
    data = CollectionDetail.model_validate(obj)
    return Resp(data=data)


@router.post(
    "",
    description=f"创建{Collection.Meta.table_description}",
    summary=f"创建{Collection.Meta.table_description}",
)
async def create_collection(request: Request, schema: CollectionCreate) -> Resp:
    await create_obj(Collection, schema.model_dump(exclude_unset=True))
    return Resp()


@router.put(
    "/{pk}",
    description=f"更新{Collection.Meta.table_description}",
    summary=f"更新{Collection.Meta.table_description}",
)
async def update_collection(request: Request, pk: int, schema: CollectionUpdate) -> Resp:
    return await update_view(get_collection_queryset(request), pk, schema)  # type: ignore


@router.delete(
    "/{pk}",
    description=f"删除{Collection.Meta.table_description}",
    summary=f"删除{Collection.Meta.table_description}",
)
async def delete_collection(request: Request, pk: int) -> Resp:
    return await delete_view(pk, get_collection_queryset(request))
