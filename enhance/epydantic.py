import inspect
from typing import Literal
from collections.abc import Callable

import pydantic
from fastapi import Body, Form, Query
from pydantic import BaseModel
from fastapi.params import _Unset

from util.general import filter_dict


def optional(*fields: str) -> Callable[[type[pydantic.BaseModel]], type[pydantic.BaseModel]]:
    def dec(cls: type[pydantic.BaseModel]) -> type[pydantic.BaseModel]:
        new_fields = {}
        for field in fields or cls.model_fields.keys():
            if field in cls.model_fields:
                field_info = cls.model_fields[field]
                field_info.default = None
                new_fields[field] = (field_info.annotation | None, field_info)  # type: ignore
            else:
                raise ValueError(f"Field {field} not found in model {cls.__name__}")

        return pydantic.create_model(cls.__name__, __base__=cls, **new_fields)  # type: ignore

    return dec


def create_sub_fields_model(
    base_model: type[BaseModel],
    fields: set[str],
) -> type[BaseModel]:
    model_fields = {}

    for field_name, field in base_model.model_fields.items():
        if field_name in fields:
            model_fields[field_name] = (field.annotation, field)

    sub_model = pydantic.create_model(f"{base_model.__name__}Subset", **model_fields, __base__=BaseModel)  # type: ignore

    for k, v in base_model.model_config.items():
        sub_model.model_config[k] = v

    pydantic_methods = dir(pydantic.BaseModel)
    for name in dir(base_model):
        if name.startswith("_") or name in pydantic_methods:
            continue
        attr = getattr(base_model, name)
        if callable(attr) and not isinstance(attr, property):
            setattr(sub_model, name, attr)

    return sub_model


def create_parameter_from_field_info(
    type_: Literal["query", "form", "body"],
    field_name: str,
    field_info: pydantic.fields.FieldInfo,
) -> inspect.Parameter:
    match type_:
        case "query":
            fastapi_parameter_cls = Query
        case "form":
            fastapi_parameter_cls = Form
        case "body":
            fastapi_parameter_cls = Body
        case _:
            raise ValueError(f"Invalid type: {type_}")

    attribute_set = field_info._attributes_set

    return inspect.Parameter(
        field_info.alias or field_name,
        inspect.Parameter.POSITIONAL_ONLY,
        default=fastapi_parameter_cls(  # type: ignore
            default=field_info.default,
            default_factory=field_info.default_factory,  # type: ignore
            media_type="application/x-www-form-urlencoded" if type_ != "body" else "application/json",
            alias=field_info.alias,
            alias_priority=field_info.alias_priority,
            validation_alias=field_info.validation_alias,  # type: ignore
            serialization_alias=field_info.serialization_alias,
            title=field_info.title,
            description=field_info.description,
            gt=attribute_set.get("gt"),  # type: ignore
            ge=attribute_set.get("ge"),  # type: ignore
            lt=attribute_set.get("lt"),  # type: ignore
            le=attribute_set.get("le"),  # type: ignore
            min_length=attribute_set.get("min_length"),  # type: ignore
            max_length=attribute_set.get("max_length"),  # type: ignore
            pattern=attribute_set.get("pattern"),  # type: ignore
            multiple_of=attribute_set.get("multiple_of") or _Unset,  # type: ignore
            allow_inf_nan=attribute_set.get("allow_inf_nan") or _Unset,  # type: ignore
            max_digits=attribute_set.get("max_digits") or _Unset,  # type: ignore
            decimal_places=attribute_set.get("decimal_places") or _Unset,  # type: ignore
            example=field_info.examples,
            deprecated=field_info.deprecated,
            json_schema_extra=field_info.json_schema_extra,  # type: ignore
            # min_length=field_info.metadata[0].min_length,
        ),
        annotation=field_info.annotation,
    )


def as_query(cls: type[BaseModel]) -> type[BaseModel]:
    new_parameters = []

    for field_name, model_field in cls.model_fields.items():
        new_parameters.append(create_parameter_from_field_info("query", field_name, model_field))

    async def as_query_func(**data) -> pydantic.BaseModel:
        data = filter_dict(data, lambda _, v: v is not None)
        return cls(**data)

    sig = inspect.signature(as_query_func)
    sig = sig.replace(parameters=new_parameters)
    as_query_func.__signature__ = sig  # type: ignore
    cls.as_query = as_query_func  # type: ignore
    return cls


def as_form(cls: type[pydantic.BaseModel]) -> type[pydantic.BaseModel]:
    new_parameters = []

    for field_name, field_info in cls.model_fields.items():
        new_parameters.append(create_parameter_from_field_info("form", field_name, field_info))

    async def as_form_func(**data) -> pydantic.BaseModel:
        data = filter_dict(data, lambda _, v: v is not None)
        return cls(**data)

    sig = inspect.signature(as_form_func)
    sig = sig.replace(parameters=new_parameters)
    as_form_func.__signature__ = sig  # type: ignore
    cls.as_form = as_form_func  # type: ignore
    return cls
