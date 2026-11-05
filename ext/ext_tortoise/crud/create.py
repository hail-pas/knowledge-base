"""Create operations."""

from tortoise.exceptions import IntegrityError
from tortoise.models import Model

from ext.ext_tortoise.crud.common import normalize_write_data
from ext.ext_tortoise.crud.exceptions import translate_integrity_error


async def create_obj(db_model: type[Model], data: dict) -> Model:
    scalar_data, relations = await normalize_write_data(data, db_model)
    try:
        obj = await db_model.create(**scalar_data)
        for field_name, related_objects in relations.items():
            if related_objects:
                await getattr(obj, field_name).add(*related_objects)
    except IntegrityError as error:
        raise translate_integrity_error(error, db_model) from error
    return obj
