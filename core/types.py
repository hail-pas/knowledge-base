# ruff: noqa
from enum import IntEnum as OriginIntEnum
from enum import StrEnum as OriginStrEnum
from enum import EnumMeta, unique


class ExtendedEnumMeta(EnumMeta):
    def __call__(cls, value, label: str = ""):  # type: ignore
        obj = super().__call__(value)  # type: ignore
        obj._value_ = value  # type: ignore
        if label:
            obj._label = label  # type: ignore
        else:
            obj._label = obj._dict[value]  # type: ignore
        return obj

    def __new__(metacls, cls, bases, classdict):  # type: ignore
        enum_class = super().__new__(metacls, cls, bases, classdict)
        enum_class._dict = {member.value: member.label for member in enum_class}  # type: ignore
        enum_class._help_text = ", ".join([f"{member.value}: {member.label}" for member in enum_class])  # type: ignore
        return enum_class


class StrEnum(OriginStrEnum, metaclass=ExtendedEnumMeta):
    _dict: dict[str, str]
    _help_text: str

    def __new__(cls, value, label: str = ""):  # type: ignore
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj._label = label  # type: ignore
        return obj

    @property
    def label(self):
        """The value of the Enum member."""
        return self._label  # type: ignore


class IntEnum(OriginIntEnum, metaclass=ExtendedEnumMeta):
    _dict: dict[int, str]
    _help_text: str

    def __new__(cls, value, label: str = ""):  # type: ignore
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj._label = label  # type: ignore
        return obj

    @property
    def label(self):
        """The value of the Enum member."""
        return self._label  # type: ignore


@unique
class RequestHeaderKeyEnum(StrEnum):
    """请求头key"""

    front_scene = ("X-Front-Scene", "请求的系统标识")
    front_version = ("X-Front-Version", "版本号")


@unique
class ContextKeyEnum(StrEnum):
    """上下文变量key."""

    # plugins
    request_id = ("request_id", "请求ID")
    request_start_timestamp = ("request_start_timestamp", "请求开始时间")
    request_body = ("request_body", "请求体")
    process_time = ("process_time", "请求处理时间/ms")

    # custom
    response_code = ("response_code", "响应code")
    response_data = ("response_data", "响应数据")  #  只记录code != 0 的


@unique
class ResponseHeaderKeyEnum(StrEnum):
    """响应头key"""

    request_id = ("X-Request-Id", "请求唯一ID")
    process_time = ("X-Process-Time", "请求处理时间")  # ms


class ApiException(Exception):
    """非 0 的业务错误."""

    code: int = -1
    message: str

    def __init__(
        self,
        message: str,
        code: int = -1,
    ) -> None:
        self.code = code
        self.message = message


class ValidationError(Exception):
    """自定义校验异常
    1. 覆盖tortoise Validation Error, 用于自定义提示语
    """

    error_type: str
    error_message_template: str
    ctx: dict  # value

    def __init__(
        self,
        error_type: str,
        error_message_template: str,
        ctx: dict,
    ) -> None:
        self.error_type = error_type
        self.error_message_template = error_message_template
        self.ctx = ctx

    def __str__(self) -> str:
        msg = self.error_message_template.format(**self.ctx)
        field_name = self.ctx.get("field_name")
        if field_name:
            msg = f"{field_name}: {msg}"
        return msg
