"""Delete operations with explicit soft-delete and physical-delete semantics."""

import datetime
import uuid

from pydantic import BaseModel
from tortoise.models import Model
from tortoise.queryset import QuerySet

from core.response import Resp
from core.types import ApiException


class DeleteResp(BaseModel):
    deleted: int


def _primary_key_filter(model: type[Model], ids: list | set | tuple) -> dict:
    return {f"{model._meta.pk_attr}__in": ids}


async def delete_view(id: str | uuid.UUID | int, queryset: QuerySet) -> Resp[DeleteResp]:
    model = queryset.model
    if hasattr(model, "delete_by_ids"):
        deleted = await model.delete_by_ids([id])
    else:
        deleted = await queryset.filter(**{model._meta.pk_attr: id}).delete()
    if deleted < 1:
        return Resp.fail(message=f"{model._meta.table_description}不存在或已被删除")
    return Resp(data=DeleteResp(deleted=deleted))


async def batch_delete_view(queryset: QuerySet, ids: set[str | uuid.UUID | int]) -> Resp[DeleteResp]:
    if not ids:
        return Resp(data=DeleteResp(deleted=0))
    model = queryset.model
    if hasattr(model, "delete_by_ids"):
        deleted = await model.delete_by_ids(ids)
    else:
        deleted = await queryset.filter(**_primary_key_filter(model, ids)).delete()
    if deleted < 1:
        return Resp.fail(message=f"{model._meta.table_description}不存在或已被删除")
    return Resp(data=DeleteResp(deleted=deleted))
