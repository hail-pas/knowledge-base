"""Shared input normalization for Tortoise model writes."""

from collections.abc import Mapping
from typing import Any

from tortoise.models import Model

from core.types import ApiException


async def normalize_write_data(data: Mapping[str, Any], model: type[Model]) -> tuple[dict, dict]:
    """Split scalar/FK values from M2M values and validate M2M IDs in bulk.

    Foreign-key existence is deliberately left to PostgreSQL constraints: checking
    each FK before writing adds a round trip and still cannot prevent races.
    """
    fields_map = model._meta.fields_map
    fk_id_fields = {f"{name}_id" for name in model._meta.fk_fields}
    m2m_fields = model._meta.m2m_fields
    scalar_data: dict[str, Any] = {}
    relations: dict[str, list[Model] | None] = {}

    unknown = set(data).difference(fields_map)
    if unknown:
        raise ApiException(f"模型字段不存在: {', '.join(sorted(unknown))}")

    for name, value in data.items():
        if name in m2m_fields:
            if value is None:
                relations[name] = None
                continue
            if not isinstance(value, (list, tuple, set)):
                raise ApiException(f"{name}必须为ID列表")

            related_model = fields_map[name].related_model
            objects_by_id: dict[Any, Model] = {}
            requested_ids = []
            for item in value:
                if isinstance(item, Model):
                    if not isinstance(item, related_model):
                        raise ApiException(f"{name}包含不匹配的关联对象")
                    objects_by_id[item.pk] = item
                else:
                    requested_ids.append(item)

            missing_ids = list(dict.fromkeys(item for item in requested_ids if item not in objects_by_id))
            if missing_ids:
                fetched = await related_model.filter(
                    **{f"{related_model._meta.pk_attr}__in": missing_ids}
                )
                objects_by_id.update({item.pk: item for item in fetched})

            ordered_ids = list(dict.fromkeys(
                item.pk if isinstance(item, Model) else item for item in value
            ))
            missing = [item for item in ordered_ids if item not in objects_by_id]
            if missing:
                raise ApiException(f"关联数据不存在: {', '.join(map(str, missing))}")
            relations[name] = [objects_by_id[item] for item in ordered_ids]
        elif name in fk_id_fields or name in fields_map:
            scalar_data[name] = value

    return scalar_data, relations
