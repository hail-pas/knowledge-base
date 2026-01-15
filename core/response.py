from enum import unique
from math import ceil
from typing import Self, Generic, TypeVar
from datetime import datetime
from collections.abc import Sequence

import orjson
from pydantic import (
    Field,
    BaseModel,
    ConfigDict,
    ValidationInfo,
    field_validator,
    model_validator,
)
from fastapi.responses import ORJSONResponse
from starlette_context import context

from core.types import IntEnum, ContextKeyEnum
from core.schema import Pager, CRUDPager
from constant.format import DATETIME_FORMAT_STRING


@unique
class ResponseCodeEnum(IntEnum):
    """业务响应代码, 除了500之外都在200的前提下返回对应code."""

    # 唯一成功响应
    success = (0, "成功")

    # custom error code
    failed = (-1, "失败")

    # http error code
    internal_error = (500, "内部服务器错误")
    unauthorized = (401, "未授权访问")
    forbidden = (403, "禁止访问")
    request_limited = (429, "请求频率限制")


class AesResponse(ORJSONResponse):
    def render(self, content: dict) -> bytes:
        """AES加密响应体"""
        # if not get_settings().DEBUG:
        # content = AESUtil(local_configs.AES.SECRET).encrypt_data(
        #       orjson.dumps(content, option=orjson.OPT_NON_STR_KEYS).decode())
        if isinstance(content, str):
            dump_content = content.encode()

        else:
            dump_content = orjson.dumps(
                content,
                option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_PASSTHROUGH_DATETIME,
            )

        return dump_content


DataT = TypeVar("DataT")


# def validate_trace_id(v: str, info: ValidationInfo) -> str:
#     if not v:
#         v = str(context.get(ContextKeyEnum.request_id.value, ""))
#     return v


# TraceId = Annotated[str, PlainValidator(validate_trace_id)]


class Resp(BaseModel, Generic[DataT]):
    """响应Model."""

    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={datetime: lambda v: v.strftime(DATETIME_FORMAT_STRING)},
    )

    code: int = Field(
        default=ResponseCodeEnum.success,
        description=f"业务响应代码, {ResponseCodeEnum._dict}",  # type: ignore
    )
    response_time: datetime | None = Field(
        default_factory=datetime.now,
        description="响应时间",
    )
    message: str | None = Field(default=None, description="响应提示信息")
    data: DataT | None = Field(
        default=None,
        description="响应数据格式",
    )
    trace_id: str = Field(default="", description="请求唯一标识", validate_default=True)

    @field_validator("trace_id")
    @classmethod
    def set_trace_id(cls, value: str, info: ValidationInfo) -> str:
        if not value:
            value = str(context.get(ContextKeyEnum.request_id.value, ""))
        return value

    @model_validator(mode="after")
    def set_failed_response(self) -> Self:
        context[ContextKeyEnum.response_code.value] = self.code
        if self.code != ResponseCodeEnum.success:
            context[ContextKeyEnum.response_data.value] = {
                "code": self.code,
                "message": self.message,
                "data": self.data,
            }
        return self

    @classmethod
    def fail(
        cls,
        message: str,
        code: int = ResponseCodeEnum.failed.value,
    ) -> Self:
        return cls(code=code, message=message)


class SimpleSuccess(Resp):
    """简单响应成功."""


class PageInfo(BaseModel):
    """翻页相关信息."""

    total_page: int
    total_count: int
    size: int
    page: int


class PageData(BaseModel, Generic[DataT]):
    page_info: PageInfo
    items: Sequence[DataT]

    @classmethod
    def create(
        cls,
        items: Sequence[DataT],
        total_count: int,
        pager: Pager | CRUDPager,
        page_info: PageInfo | None = None,
    ) -> "PageData[DataT]":
        if page_info is None:
            page_info = generate_page_info(total_count, pager)
        return cls(page_info=page_info, items=items)


def generate_page_info(total_count: int, pager: Pager | CRUDPager) -> PageInfo:
    return PageInfo(
        total_page=ceil(total_count / pager.limit),
        total_count=total_count,
        size=pager.limit,
        page=pager.offset // pager.limit + 1,
    )
