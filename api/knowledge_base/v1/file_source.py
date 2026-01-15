from typing import Annotated
from datetime import datetime

from fastapi import Depends, Request, APIRouter
from tortoise.queryset import QuerySet

from core.types import ApiException
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
    FileSource,
)
from ext.file_source.factory import FileSourceAdapterFactory
from ext.file_source.exceptions import FileSourceNotFoundError

from api.service.file_source.schema import (
    FileSourceList,
    FileSourceDetail,
    FileSourceCreate,
    FileSourceUpdate,
    FileSourceFilterSchema

)

router = APIRouter()

# ============ FileSource CRUD ============


def get_file_source_queryset(request: Request) -> QuerySet[FileSource]:
    filter_ = {"deleted_at": 0}
    return FileSource.filter(**filter_)


@router.get(
    "",
    description=f"{FileSource.Meta.table_description}列表",
    summary=f"{FileSource.Meta.table_description}列表",
)
async def get_file_source_list(
    request: Request,
    filter_: Annotated[FileSourceFilterSchema, Depends(FileSourceFilterSchema.as_query)],  # type: ignore
    pager: CRUDPager = pagination_factory(
        db_model=FileSource,
        list_schema=FileSourceList,
        search_fields={"name", "description"},
        order_fields={
            "created_at",
            "name",
        },
        max_limit=1000,
    ),
) -> Resp[PageData[FileSourceList]]:
    return await list_view(get_file_source_queryset(request), filter_, pager)


@router.get(
    "/{pk}",
    description=f"获取{FileSource.Meta.table_description}详情",
    summary=f"获取{FileSource.Meta.table_description}详情",
)
async def get_file_source_detail(request: Request, pk: int) -> Resp[FileSourceDetail]:
    queryset = get_file_source_queryset(request)
    obj: FileSource | None = await queryset.get_or_none(
        **{queryset.model._meta.pk_attr: pk},
    )
    if not obj:
        raise ApiException("对象不存在")
    obj = await obj_prefetch_fields(obj, FileSourceDetail)  # type: ignore
    data = FileSourceDetail.model_validate(obj)
    return Resp(data=data)


@router.post(
    "",
    description=f"创建{FileSource.Meta.table_description}",
    summary=f"创建{FileSource.Meta.table_description}",
)
async def create_file_source(request: Request, schema: FileSourceCreate) -> Resp:
    await create_obj(FileSource, schema.model_dump(exclude_unset=True))
    return Resp()


@router.put(
    "/{pk}",
    description=f"更新{FileSource.Meta.table_description}",
    summary=f"更新{FileSource.Meta.table_description}",
)
async def update_file_source(request: Request, pk: int, schema: FileSourceUpdate) -> Resp:
    return await update_view(get_file_source_queryset(request), pk, schema)  # type: ignore


@router.delete(
    "/{pk}",
    description=f"删除{FileSource.Meta.table_description}",
    summary=f"删除{FileSource.Meta.table_description}",
)
async def delete_file_source(request: Request, pk: int) -> Resp:
    return await delete_view(pk, get_file_source_queryset(request))
