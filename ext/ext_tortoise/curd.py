import re
import uuid
from typing import TypeVar
from collections import defaultdict

from fastapi import Body, Query, Depends
from pydantic import BaseModel
from tortoise.models import Model
from tortoise.queryset import QuerySet
from tortoise.exceptions import IntegrityError
from tortoise.expressions import Q
from tortoise.contrib.pydantic.base import PydanticModel

from core.types import ApiException
from core.schema import CRUDPager, paginate
from core.response import Resp, PageData

unique_error_msg_key_regex = re.compile(r"'(.*?)'")


ModelType = TypeVar("ModelType", bound=Model)

PydanticModelType = TypeVar("PydanticModelType", bound=PydanticModel)


def pagination_factory(
    db_model: type[ModelType],
    search_fields: set[str],
    order_fields: set[str],
    list_schema: type[PydanticModel],
    max_limit: int | None = None,
    param_type: type[Query] | type[Body] = Query,  # type: ignore
) -> CRUDPager:
    return Depends(paginate(db_model, search_fields, order_fields, list_schema, max_limit, param_type))  # type: ignore


async def get_all_obj(
    queryset: QuerySet[ModelType],  # type: ignore
    pagination: CRUDPager,
    *args: Q,
    **kwargs: dict,
) -> tuple[list, int]:  # type: ignore
    queryset = queryset.filter(*args).filter(**kwargs).order_by(*pagination.order_by)

    search = pagination.search
    if search and pagination.available_search_fields:
        sub_q_exps = []
        for search_field in pagination.available_search_fields:
            sub_q_exps.append(
                Q(**{f"{search_field}__icontains": search}),
            )
        q_expression = Q(*sub_q_exps, join_type=Q.OR)
        queryset = queryset.filter(q_expression)

    list_schema = pagination.list_schema
    if pagination.selected_fields:
        list_schema = create_sub_fields_model(  # type: ignore
            pagination.list_schema,
            pagination.selected_fields,
        )

    data = await list_schema.from_queryset(  # type: ignore
        queryset.offset(pagination.offset).limit(pagination.limit),
    )
    total = await queryset.count()
    return data, total


async def obj_prefetch_fields(obj: Model, schema: type[PydanticModelType]) -> Model:
    db_model = obj.__class__
    _db2fields = defaultdict(list)
    for f in db_model._meta.fetch_fields.intersection(set(schema.model_fields.keys())):
        _db2fields[db_model._meta.fields_map[f].related_model._meta.db].append(f)  # type: ignore

    for db, fetch_fields in _db2fields.items():
        await obj.fetch_related(
            *fetch_fields,
            using_db=db,
        )
    return obj


async def kwargs_clean(
    data: dict,
    model: type[ModelType],
) -> tuple[dict, dict]:
    fields_map = model._meta.fields_map
    fk_fields = [f"{i}_id" for i in model._meta.fk_fields]
    m2m_fields = model._meta.m2m_fields

    simple_data = {}
    m2m_fields_data: dict = defaultdict(list)

    for key in data:
        if key not in fields_map:
            continue
        if key in fk_fields:
            if data[key]:
                field = fields_map[key.split("_id")[0]]
                obj = await field.related_model.get_or_none(  # type: ignore
                    **{field.to_field: data[key]},  # type: ignore
                )
                if not obj:
                    raise ApiException(
                        f"ID为{data[key]}的{field.description}不存在",
                    )
            simple_data[key] = data[key]
            continue

        if key in m2m_fields:
            if data[key] is None:
                m2m_fields_data[key] = None  # type: ignore
                continue
            m2m_fields_data[key] = []
            field = fields_map[key]
            model = field.related_model  # type: ignore
            for related_id in data[key]:
                if isinstance(related_id, Model):
                    m2m_fields_data[key].append(obj) # type: ignore
                    continue
                obj = await model.get_or_none(
                    **{model._meta.pk_attr: related_id},
                )
                if not obj:
                    raise ApiException(
                        f"id为{related_id}的{model._meta.table_description}不存在",
                    )
                m2m_fields_data[key].append(obj)
            continue

        simple_data[key] = data[key]

    return simple_data, m2m_fields_data


async def create_obj(
    db_model: type[ModelType],
    data: dict,
) -> Model:
    data, m2m_data = await kwargs_clean(
        data,
        db_model,
    )

    try:
        obj = await db_model.create(**data)
    except IntegrityError as e:
        # 安全地提取错误消息
        msg = str(e)
        if len(e.args) > 0 and hasattr(e.args[0], 'args') and len(e.args[0].args) > 1:
            msg = e.args[0].args[1]

        if "Duplicate" in msg:
            msg_keys = unique_error_msg_key_regex.findall(msg)
            if (
                msg_keys
                and hasattr(db_model.Meta, "unique_error_messages")
                and db_model.Meta.unique_error_messages.get(msg_keys[-1])  # type: ignore
            ):
                msg = db_model.Meta.unique_error_messages.get(msg_keys[-1])  # type: ignore
            else:
                msg = f"{db_model._meta.table_description}已存在"
        raise ApiException(message=msg) from e

    for k, v in m2m_data.items():
        if v:
            await getattr(obj, k).add(*v)

    return obj


async def update_obj(
    obj: Model,
    queryset: QuerySet[ModelType],
    data: dict,
) -> Model:
    if not data:
        return obj

    db_model = obj.__class__
    data, m2m_data = await kwargs_clean(
        data,
        db_model,
    )

    if data:
        try:
            await queryset.filter(
                **{
                    db_model._meta.pk_attr: getattr(
                        obj,
                        db_model._meta.pk_attr,
                    ),
                },
            ).update(**data)
        except IntegrityError as e:
            msg = e.args[0].args[1]
            if "Duplicate" in msg:
                msg_keys = unique_error_msg_key_regex.findall(msg)
                if (
                    msg_keys
                    and hasattr(db_model.Meta, "unique_error_messages")
                    and db_model.Meta.unique_error_messages.get(msg_keys[-1])  # type: ignore
                ):
                    msg = db_model.Meta.unique_error_messages.get(msg_keys[-1])  # type: ignore
                else:
                    msg = f"{db_model._meta.table_description}已存在"
            raise ApiException(message=msg) from e
    for k, v in m2m_data.items():
        if v is None:
            continue
        await getattr(obj, k).clear()
        if not v:
            continue
        await getattr(obj, k).add(*v)
    await obj.refresh_from_db()
    return obj  # type: ignore


async def list_view(
    queryset: QuerySet,
    filter: BaseModel,
    pager: CRUDPager,
) -> Resp:
    if pager.selected_fields:
        queryset = queryset.only(*pager.selected_fields)
    data, total = await get_all_obj(
        queryset=queryset.distinct(),
        pagination=pager,
        **filter.model_dump(exclude_unset=True, exclude_none=True),
    )
    return Resp(
        data=PageData.create(records=data, total_count=total, pager=pager),  # type: ignore
    )


async def detail_view(queryset: QuerySet, pk: str | int | uuid.UUID, resp_schema: type[PydanticModelType]) -> Resp:
    obj = await queryset.get_or_none(
        **{queryset.model._meta.pk_attr: pk},
    )
    if not obj:
        raise ApiException("对象不存在")
    obj = await obj_prefetch_fields(obj, resp_schema)
    data = resp_schema.model_validate(obj)
    return Resp(
        data=data,  # type: ignore
    )


async def update_view(
    queryset: QuerySet[ModelType],
    id: str | uuid.UUID | int,
    schema: PydanticModel,
) -> Resp:
    data = schema.model_dump(exclude_unset=True)
    if not data:
        return Resp()

    obj = await queryset.get_or_none(
        **{queryset.model._meta.pk_attr: id},
    )
    if not obj:
        raise ApiException("对象不存在")

    await update_obj(obj, queryset, data)
    return Resp()


class DeleteResp(BaseModel):
    deleted: int


async def delete_view(id: str | uuid.UUID | int, queryset: QuerySet[ModelType]) -> Resp[DeleteResp]:
    db_model = queryset.model
    db_model_label = db_model._meta.table_description
    if hasattr(db_model, "delete_by_ids"):
        r = await db_model.delete_by_ids([id])  # type: ignore
    else:
        r = await queryset.filter(
            **{db_model._meta.pk_attr: id},
        ).delete()
    if r < 1:
        return Resp.fail(message=f"{db_model_label}不存在或已被删除")
    return Resp(data=DeleteResp(deleted=r))


async def batch_delete_view(
    queryset: QuerySet[ModelType],
    ids: set[str | uuid.UUID | int],
) -> Resp[DeleteResp]:
    db_model = queryset.model
    db_model_label = db_model._meta.table_description
    if hasattr(db_model, "delete_by_ids"):
        r = await db_model.delete_by_ids(ids)  # type: ignore
    else:
        r = await queryset.filter(
            id__in=ids,
        ).delete()
    if r < 1:
        return Resp.fail(message=f"{db_model_label}不存在或已被删除")
    return Resp(data=DeleteResp(deleted=r))
