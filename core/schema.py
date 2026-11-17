from collections.abc import Callable

from fastapi import Body, Query
from pydantic import BaseModel, PositiveInt, conint

from core.types import ApiException


class Pager(BaseModel):
    limit: PositiveInt = 10
    offset: conint(ge=0) = 0  # type: ignore


class CRUDPager(Pager):
    order_by: set[str] = set()
    search: str | None = None
    selected_fields: set[str] | None = None
    available_search_fields: set[str] | None = None
    list_schema: type[BaseModel]


class IdsSchema(BaseModel):
    ids: set[str]


def pure_get_pager(
    page: PositiveInt = Query(default=1, examples=[1], description="第几页"),
    size: PositiveInt = Query(default=10, examples=[10], description="每页数量"),
) -> Pager:
    return Pager(limit=size, offset=(page - 1) * size)


def paginate(
    model,
    search_fields: set[str],
    order_fields: set[str],
    list_schema: type[BaseModel],
    max_limit: int | None,
    param_type: type[Query] | type[Body] = Query,  # type: ignore
) -> Callable:
    def get_pager(
        page: PositiveInt = param_type(default=1, examples=[1], description="第几页"),
        size: PositiveInt = param_type(default=10, examples=[10], description="每页数量"),
        search: str | None = param_type(
            None,
            description="搜索关键字."
            + (f" 匹配字段: {', '.join(sorted(search_fields))}" if search_fields else "无可匹配的字段"),
        ),
        order_by: set[str] = param_type(
            default=set(),
            description=(
                "排序字段. 升序保持原字段名, 降序增加前缀-."
                + (f" 可选字段: {', '.join(sorted(order_fields))}" if order_fields else " 无可排序字段")
            ),
        ),
        selected_fields: set[str] = param_type(
            default=set(),
            description=f"指定返回字段. 可选字段: {', '.join(list_schema.model_fields)}",
        ),
    ):
        if max_limit is not None:
            size = min(size, max_limit)
        if page <= 0:
            raise ApiException("页码必须大于0")
        if size <= 0:
            raise ApiException("每页数量必须大于0")

        normalized_order: set[str] = set()
        for value in order_by:
            field = value[1:] if value.startswith("-") else value
            if not field or field not in order_fields:
                raise ApiException("排序字段不存在")
            normalized_order.add(value)

        if selected_fields:
            available_fields = set(list_schema.model_fields)
            invalid_fields = selected_fields.difference(available_fields)
            if invalid_fields:
                raise ApiException(f"返回字段不存在: {', '.join(sorted(invalid_fields))}")
            selected_fields = set(selected_fields)
            if "id" in available_fields:
                selected_fields.add("id")

        return CRUDPager(
            limit=size,
            offset=(page - 1) * size,
            order_by=normalized_order,
            search=search,
            selected_fields=selected_fields,
            available_search_fields=search_fields,
            list_schema=list_schema,
        )

    return get_pager
