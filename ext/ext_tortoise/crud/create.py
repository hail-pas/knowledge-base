"""Create operations."""

from tortoise.exceptions import IntegrityError
from tortoise.models import Model
from tortoise.transactions import in_transaction

from ext.ext_tortoise.crud.common import normalize_write_data
from ext.ext_tortoise.crud.exceptions import translate_integrity_error


async def create_obj(db_model: type[Model], data: dict) -> Model:
    scalar_data, relations = await normalize_write_data(data, db_model)
    try:
        async with in_transaction(connection_name=db_model._meta.default_connection) as connection:
            obj = await db_model.create(using_db=connection, **scalar_data)
            for field_name, related_objects in relations.items():
                if related_objects:
                    await getattr(obj, field_name).add(*related_objects, using_db=connection)
    except IntegrityError as error:
        raise translate_integrity_error(error, db_model) from error
    return obj
