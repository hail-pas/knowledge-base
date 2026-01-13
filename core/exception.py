import traceback
from typing import Any, Callable

from fastapi import Request, WebSocket
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException

from core.types import ApiException, ValidationError
from config.main import local_configs
from core.response import Resp, AesResponse, ResponseCodeEnum
from config.default import EnvironmentEnum
from constant.validate import (
    ValidationErrorMsgTemplates,
    DirectValidateErrorMsgTemplates,
)


async def api_exception_handler(
    request: Request | WebSocket,
    exc: ApiException,
) -> AesResponse:
    return AesResponse(
        content=Resp(
            code=exc.code,
            message=exc.message,
            data=None,
        ).model_dump_json(),
        # 小于0的为200状态下的自定义code
        status_code=exc.code if exc.code > 0 else 200,
    )


async def unexpected_exception_handler(
    request: Request | WebSocket,
    exc: Exception,
) -> AesResponse | HTMLResponse:
    if local_configs.project.environment in [
        EnvironmentEnum.local,
        EnvironmentEnum.development,
        EnvironmentEnum.test,
    ]:
        return HTMLResponse(
            status_code=500,
            content=traceback.format_exc(),
            headers=local_configs.server.cors.headers,
        )

    return AesResponse(
        content=Resp(
            code=ResponseCodeEnum.internal_error.value,
            message=ResponseCodeEnum.internal_error.label,
            data=None,
        ).model_dump_json(),
        status_code=500,
        headers=local_configs.server.cors.headers,
    )


async def http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> AesResponse:
    """HttpException 状态码非 200 的错误
    :param request:
    :param exc:
    :return:
    """
    return AesResponse(
        content=Resp(
            code=exc.status_code,
            message=exc.detail,
            data=None,
        ).model_dump_json(),
        status_code=exc.status_code,
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> AesResponse:
    """参数校验错误."""

    error = exc.errors()[0]
    error_type = error["type"]
    ctx = error.get("ctx", {})

    field_name = error["loc"][0] if len(error["loc"]) == 1 else ".".join([str(i) for i in error["loc"][1:]])

    if error_type in DirectValidateErrorMsgTemplates:
        field_name, message = DirectValidateErrorMsgTemplates[error_type]
        message = message.format(**ctx)
    else:
        message = ValidationErrorMsgTemplates[error_type]
        message = message.format(**ctx)

    return AesResponse(
        content=Resp(
            code=ResponseCodeEnum.failed,
            message=f"{field_name}: {message}",
            data=exc.errors(),  # {"data": exc.body, "errors": error_list},
        ).model_dump_json(),
        status_code=422,
    )


def get_validation_text(exc: RequestValidationError, pyd: BaseModel) -> str:
    error = exc.errors()[0]
    error_type = error["type"]
    ctx = error.get("ctx", {})

    field_name = error["loc"][0] if len(error["loc"]) else ".".join([str(i) for i in error["loc"][1:]])

    if error_type in DirectValidateErrorMsgTemplates:
        field_name, message = DirectValidateErrorMsgTemplates[error_type]
        message = message.format(**ctx)
    else:
        message = ValidationErrorMsgTemplates[error_type]
        message = message.format(**ctx)
    field_info = pyd.model_fields.get(field_name)
    if hasattr(field_info, "description"):
        return f"{field_info.description}错误"  # type: ignore
        # return f"{field_info.description}({field_name}): {message}"
    return f"{field_name}错误"


async def custom_validation_error_handler(
    request: Request,
    exc: ValidationError,
) -> AesResponse:
    message = exc.error_message_template.format(**exc.ctx)
    if "field_name" in exc.ctx:
        message = f"{exc.ctx['field_name']}: {message}"

    return AesResponse(
        content=Resp(
            code=ResponseCodeEnum.failed,
            message=message,
        ).model_dump_json(),
        status_code=422,
    )


handler_roster: list[tuple[type[Exception], Callable[..., Any]]] = [
    (RequestValidationError, validation_exception_handler),
    # (PydanticValidationError, validation_exception_handler),
    (ValidationError, custom_validation_error_handler),
    # (TortoiseValidationError, tortoise_validation_error_handler),
    (ApiException, api_exception_handler),
    (HTTPException, http_exception_handler),
    (Exception, unexpected_exception_handler),
]
