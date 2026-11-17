"""Public CRUD helpers.

Kept as the integration point used by API modules; implementation lives in
focused crud/ and query/ modules.
"""

import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import TypeVar

from pydantic import BaseModel
from tortoise.models import Model
from tortoise.queryset import QuerySet
from tortoise.contrib.pydantic.base import PydanticModel

from core.response import Resp, PageData
from core.types import ApiException
from ext.ext_tortoise.crud import (
    DeleteResp,
    batch_delete_view,
    create_obj,
    delete_view,
    update_obj,
)
from ext.ext_tortoise.query.pagination import get_all_obj, pagination_factory

PydanticModelType = TypeVar("PydanticModelType", bound=PydanticModel)
ModelType = TypeVar("ModelType", bound=Model)


async def obj_prefetch_fields(obj: Model, schema: type[PydanticModelType]) -> Model:
    fields_by_connection = defaultdict(list)
    for field_name in obj._meta.fetch_fields.intersection(schema.model_fields):
        related_model = obj._meta.fields_map[field_name].related_model
        fields_by_connection[related_model._meta.db].append(field_name)
    for connection, fields in fields_by_connection.items():
        await obj.fetch_related(*fields, using_db=connection)
    return obj


async def list_view(
    queryset: QuerySet,
    filter: BaseModel,
    pager,
    *,
    post_process: Callable[[list], Awaitable[list | None]] | None = None,
) -> Resp:
    data, total = await get_all_obj(
        queryset=queryset.distinct(),
        pagination=pager,
        **filter.model_dump(exclude_unset=True, exclude_none=True),
    )
    if post_process:
        result = await post_process(data)
        if result is not None:
            data = result
    return Resp(data=PageData.create(items=data, total_count=total, pager=pager))


async def detail_view(
    queryset: QuerySet,
    pk: str | int | uuid.UUID,
    resp_schema: type[PydanticModelType],
) -> Resp:
    obj = await queryset.get_or_none(**{queryset.model._meta.pk_attr: pk})
    if not obj:
        raise ApiException("对象不存在")
    await obj_prefetch_fields(obj, resp_schema)
    return Resp(data=resp_schema.model_validate(obj))


async def update_view(queryset: QuerySet[ModelType], id: str | uuid.UUID | int, schema: PydanticModel) -> Resp:
    data = schema.model_dump(exclude_unset=True)
    if not data:
        return Resp()
    obj = await queryset.get_or_none(**{queryset.model._meta.pk_attr: id})
    if not obj:
        raise ApiException("对象不存在")
    await update_obj(obj, queryset, data)
    return Resp()


__all__ = [
    "DeleteResp",
    "batch_delete_view",
    "create_obj",
    "delete_view",
    "detail_view",
    "get_all_obj",
    "list_view",
    "obj_prefetch_fields",
    "pagination_factory",
    "update_obj",
    "update_view",
]
