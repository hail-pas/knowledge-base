from typing import Any

from fastapi import routing
from pydantic import ValidationError
from fastapi.types import IncEx
from fastapi._compat import ModelField, _normalize_errors, _regenerate_error_with_loc
from fastapi.routing import _prepare_response_content
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import ResponseValidationError
from pymysql.converters import escape_item, escape_bytes_prefixed
from aiomysql.connection import Connection
from tortoise.expressions import RawSQL
from starlette.concurrency import run_in_threadpool

from core.response import Resp


def validate(
    self: ModelField,
    value: Any,  # ruff: noqa: ANN401
    values: dict[str, Any] = {},  # noqa: B006
    *,
    loc: tuple[int | str, ...] = (),
) -> tuple[Any, list[dict[str, Any]] | None]:
    try:
        return (
            self._type_adapter.validate_python(value, from_attributes=True),
            None,
        )
    except ValidationError as exc:
        errors = exc.errors(include_url=False)
        annotation = self.field_info.annotation
        if hasattr(annotation, "model_fields"):
            fields = annotation.model_fields  # type: ignore
            for e in errors:
                t = ()
                for current_loc in e["loc"]:
                    t = (
                        # ruff: noqa: E501
                        t + ((fields[current_loc].description or fields[current_loc].title).split(";")[0],)  # type: ignore
                        if current_loc in fields and (fields[current_loc].description or fields[current_loc].title)
                        else t + (current_loc,)
                    )
                e["loc"] = t
        else:
            for e in errors:
                if self.field_info.title:
                    e["loc"] = (self.field_info.title,)

        return None, _regenerate_error_with_loc(errors=errors, loc_prefix=loc)


def escape(self: Connection, obj: Any) -> Any:
    # 处理 RawSQL
    if isinstance(obj, str):
        return "'" + self.escape_string(obj) + "'"
    if isinstance(obj, bytes):
        return escape_bytes_prefixed(obj)
    if isinstance(obj, RawSQL):
        return obj.sql
    return escape_item(obj, self._charset)


async def serialize_response(
    *,
    field: ModelField | None = None,
    response_content: Any,
    include: IncEx | None = None,
    exclude: IncEx | None = None,
    by_alias: bool = True,
    exclude_unset: bool = False,
    exclude_defaults: bool = False,
    exclude_none: bool = False,
    is_coroutine: bool = True,
) -> Any:
    if isinstance(response_content, Resp):
        # 兼容 Resp 和 PageResp
        value = response_content.model_dump(
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
        )
        return jsonable_encoder(
            value,
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            custom_encoder=response_content.model_config.get("json_encoders"),  # type: ignore
        )

    if field:
        errors = []
        if not hasattr(field, "serialize"):
            # pydantic v1
            response_content = _prepare_response_content(
                response_content,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
                exclude_none=exclude_none,
            )
        if is_coroutine:
            value, errors_ = field.validate(response_content, {}, loc=("response",))
        else:
            value, errors_ = await run_in_threadpool(field.validate, response_content, {}, loc=("response",))
        if isinstance(errors_, list):
            errors.extend(errors_)
        elif errors_:
            errors.append(errors_)
        if errors:
            raise ResponseValidationError(errors=_normalize_errors(errors), body=response_content)

        if hasattr(field, "serialize"):
            return field.serialize(
                value,
                include=include,
                exclude=exclude,
                by_alias=by_alias,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
                exclude_none=exclude_none,
            )

        return jsonable_encoder(
            value,
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
        )
    return jsonable_encoder(response_content)


def patch() -> None:
    # ValidationError loc 字段改为使用 title
    ModelField.validate = validate  # type: ignore
    Connection.escape = escape  # type: ignore
    routing.serialize_response = serialize_response  # type: ignore
