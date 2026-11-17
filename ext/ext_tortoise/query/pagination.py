"""Reusable pagination and list-query helpers."""

from functools import lru_cache
from typing import TypeVar
from collections.abc import Callable

from fastapi import Body, Depends, Query
from pydantic import create_model
from tortoise.expressions import Q
from tortoise.models import Model
from tortoise.queryset import QuerySet
from tortoise.contrib.pydantic.base import PydanticModel

from core.schema import CRUDPager, paginate

ModelType = TypeVar("ModelType", bound=Model)


def pagination_factory(
    db_model: type[ModelType],
    search_fields: set[str],
    order_fields: set[str],
    list_schema: type[PydanticModel],
    max_limit: int | None = None,
    param_type: type[Query] | type[Body] = Query,
) -> CRUDPager:
    return Depends(paginate(db_model, search_fields, order_fields, list_schema, max_limit, param_type))


@lru_cache(maxsize=256)
def _selected_schema(schema: type[PydanticModel], selected_fields: frozenset[str]) -> type[PydanticModel]:
    fields = {
        name: (field.annotation, field)
        for name, field in schema.model_fields.items()
        if name in selected_fields
    }
    return create_model(
        f"{schema.__name__}Selected",
        __config__=schema.model_config,
        __module__=schema.__module__,
        **fields,
    )


async def get_all_obj(
    queryset: QuerySet[ModelType],
    pagination: CRUDPager,
    *args: Q,
    **kwargs,
) -> tuple[list, int]:
    queryset = queryset.filter(*args, **kwargs)
    if pagination.search and pagination.available_search_fields:
        conditions = [
            Q(**{f"{field}__icontains": pagination.search})
            for field in pagination.available_search_fields
        ]
        queryset = queryset.filter(Q(*conditions, join_type=Q.OR))

    if pagination.order_by:
        queryset = queryset.order_by(*pagination.order_by)

    schema = pagination.list_schema
    if pagination.selected_fields:
        schema = _selected_schema(schema, frozenset(pagination.selected_fields))

    total = await queryset.count()
    data = await schema.from_queryset(
        queryset.offset(pagination.offset).limit(pagination.limit)
    )
    return data, total
