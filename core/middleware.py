import time
import uuid

from loguru import logger
from starlette.types import Message
from starlette_context import context, request_cycle_context
from starlette.requests import Request, HTTPConnection
from starlette.responses import Response
from fastapi.middleware.gzip import GZipMiddleware
from starlette.datastructures import MutableHeaders
from starlette_context.errors import MiddleWareValidationError
from starlette.middleware.base import RequestResponseEndpoint
from starlette.middleware.cors import CORSMiddleware
from starlette_context.plugins import Plugin
from starlette_context.middleware import ContextMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from core.types import ContextKeyEnum, ResponseHeaderKeyEnum
from config.main import local_configs
from core.response import ResponseCodeEnum


# enrich response 会触发两次 http.response.start、http.response.body
class RequestIdPlugin(Plugin):
    """请求唯一标识"""

    key: str = ContextKeyEnum.request_id.value
    # is_internal: bool = False

    # def __init__(self, is_internal: bool = False) -> None:
    #     self.is_internal = is_internal

    async def process_request(
        self,
        request: Request | HTTPConnection,
    ) -> str:
        # if self.is_internal:
        request_id = request.headers.get(
            ResponseHeaderKeyEnum.request_id.value,
        )
        return request_id or str(uuid.uuid4())

    async def enrich_response(
        self,
        response: Response | Message,
    ) -> None:
        value = str(context.get(self.key))
        # for ContextMiddleware
        if isinstance(response, Response):
            response.headers[ResponseHeaderKeyEnum.request_id.value] = value
        # for ContextPureMiddleware
        else:
            if response["type"] == "http.response.start":
                headers = MutableHeaders(scope=response)
                headers.append(ResponseHeaderKeyEnum.request_id.value, value)


class RequestStartTimestampPlugin(Plugin):
    """请求开始时间"""

    key = ContextKeyEnum.request_start_timestamp.value

    async def process_request(
        self,
        request: Request | HTTPConnection,
    ) -> float:
        return time.time()


class RequestProcessInfoPlugin(Plugin):
    """请求、响应相关的日志"""

    key = ContextKeyEnum.process_time.value

    async def process_request(
        self,
        request: HTTPConnection,
    ) -> dict:
        return {
            "method": request.scope["method"],
            "uri": request.scope["path"],
            "client": request.scope.get("client", ("", ""))[0],
        }

    async def enrich_response(
        self,
        response: Response | Message,
    ) -> None:
        request_start_timestamp = context.get(RequestStartTimestampPlugin.key)
        if not request_start_timestamp:
            raise RuntimeError("Cannot evaluate process time")
        process_time = (time.time() - float(request_start_timestamp)) * 1000  # ms
        if isinstance(response, Response):
            response.headers[ResponseHeaderKeyEnum.process_time.value] = str(
                process_time,
            )
        else:
            if response["type"] == "http.response.start":
                headers = MutableHeaders(scope=response)
                headers.append(
                    ResponseHeaderKeyEnum.process_time.value,
                    str(process_time),
                )
        info_dict = context.get(self.key)
        info_dict["process_time"] = process_time  # type: ignore
        code = context.get(ContextKeyEnum.response_code.value)
        if code is not None and code != ResponseCodeEnum.success.value:
            data = context.get(ContextKeyEnum.response_data.value)
            info_dict["response_data"] = data  # type: ignore

        logger.info(info_dict)


class ContextMiddlewareWithTraceId(ContextMiddleware):
    """上下文中间件"""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        try:
            context = await self.set_context(request)
        except MiddleWareValidationError as e:
            error_response = e.error_response or self.error_response
            return error_response

        # create request-scoped context
        with (
            request_cycle_context(context),
            logger.contextualize(  # add this
                trace_id=context.get(RequestIdPlugin.key),
            ),
        ):
            # process rest of response stack
            response = await call_next(request)
            # gets back to middleware, process response with plugins
            for plugin in self.plugins:
                await plugin.enrich_response(response)
            # return response before resetting context
            # allowing further middlewares to still use the context
            return response


roster = [
    # >>>>> Middleware Func
    (
        ContextMiddlewareWithTraceId,
        {
            "plugins": [
                RequestIdPlugin(),
                RequestStartTimestampPlugin(),
                RequestProcessInfoPlugin(),
            ],
        },
    ),
    (GZipMiddleware, {"minimum_size": 1000}),
    # >>>>> Middleware Class
    (
        CORSMiddleware,
        {
            "allow_origins": local_configs.server.cors.allow_origins,
            "allow_credentials": local_configs.server.cors.allow_credentials,
            "allow_methods": local_configs.server.cors.allow_methods,
            "allow_headers": local_configs.server.cors.allow_headers,
            "expose_headers": local_configs.server.cors.expose_headers,
        },
    ),
    (
        TrustedHostMiddleware,
        {
            "allowed_hosts": local_configs.server.allow_hosts,
        },
    ),
]
