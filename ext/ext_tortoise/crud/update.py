"""Update operations."""

from tortoise.exceptions import IntegrityError
from tortoise.models import Model
from tortoise.queryset import QuerySet
from tortoise.transactions import in_transaction

from ext.ext_tortoise.crud.common import normalize_write_data
from ext.ext_tortoise.crud.exceptions import translate_integrity_error


async def update_obj(obj: Model, queryset: QuerySet, data: dict) -> Model:
    if not data:
        return obj

    model = type(obj)
    scalar_data, relations = await normalize_write_data(data, model)
    try:
        async with in_transaction(connection_name=model._meta.default_connection) as connection:
            if scalar_data:
                obj.update_from_dict(scalar_data)
                await obj.save(using_db=connection, update_fields=list(scalar_data))
            for field_name, related_objects in relations.items():
                if related_objects is None:
                    continue
                relation = getattr(obj, field_name)
                await relation.clear(using_db=connection)
                if related_objects:
                    await relation.add(*related_objects, using_db=connection)
    except IntegrityError as error:
        raise translate_integrity_error(error, model) from error

    await obj.refresh_from_db()
    return obj
