from typing import Callable

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
    # available_sort_fields: set[str] | None = None
    # available_search_fields: set[str] | None = None


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
) -> Callable[[PositiveInt, PositiveInt, str, set[str], set[str] | None], CRUDPager]:
    def get_pager(
        page: PositiveInt = param_type(default=1, examples=[1], description="第几页"),
        size: PositiveInt = param_type(default=10, examples=[10], description="每页数量"),
        search: str = param_type(
            None,
            description="搜索关键字."
            + (f" 匹配字段: {', '.join(search_fields)}" if search_fields else "无可匹配的字段"),  # ruff: noqa: E501
        ),
        order_by: set[str] = param_type(
            default=set(),
            # examples=["-id"],
            description=(
                "排序字段. 升序保持原字段名, 降序增加前缀-."
                + (f" 可选字段: {', '.join(order_fields)}" if order_fields else " 无可排序字段")  # ruff: noqa: E501
            ),
        ),
        selected_fields: set[str] = param_type(
            default=set(),
            description=f"指定返回字段. 可选字段: {', '.join(list_schema.model_fields.keys())}",
        ),
    ) -> CRUDPager:
        if max_limit is not None:
            size = min(size, max_limit)
        for field in order_by:
            if field.startswith("-"):
                field = field[1:]  # noqa

            if hasattr(model, "model_fields"):
                available_order_fields = model.model_fields.keys()
            else:
                available_order_fields = model._meta.db_fields

            if field not in available_order_fields:
                raise ApiException(
                    "排序字段不存在",
                )
        if selected_fields:
            selected_fields.add("id")

        if page <= 0:
            raise ApiException(
                "页码必须大于0",
            )
        if size <= 0:
            raise ApiException(
                "每页数量必须大于0",
            )
        return CRUDPager(
            limit=size,
            offset=(page - 1) * size,
            order_by=set(
                filter(lambda i: i.split("-")[-1] in order_fields, order_by),
            ),
            search=search,
            selected_fields=selected_fields,
            available_search_fields=search_fields,
            list_schema=list_schema,
        )

    return get_pager  # type: ignore
