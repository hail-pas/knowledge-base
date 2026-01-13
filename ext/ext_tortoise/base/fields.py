import abc
import uuid
import datetime
import warnings
from typing import Any, Protocol
from zoneinfo import ZoneInfo
from urllib.parse import urlparse
from collections.abc import Callable

from tortoise import fields, timezone, validators
from tortoise.models import Model
from tortoise.timezone import get_use_tz, get_default_timezone
from tortoise.exceptions import ConfigurationError
from tortoise.expressions import RawSQL


class StorageMixin(Protocol):

    def get_full_path(
        self,
        path: str,
        expire: int | None = None,
    ) -> tuple[bool, str]: ...


class FileField(fields.CharField):
    """
    OSS文件字段
    """

    _file_storage: StorageMixin
    _expire: int | None
    _extensions: list[str] | None

    def __init__(
        self,
        max_length: int,
        storage: StorageMixin,
        extensions: list[str] | None = None,
        expire: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(max_length=max_length, **kwargs)
        self._file_storage = storage
        self._expire = expire
        self._extensions = extensions

    def to_db_value(self, value: str, instance: "FileField") -> str:  # type: ignore
        if not value:
            return ""
        if "." not in value:
            return value
        if value.startswith("http"):
            value = urlparse(value).path
        extension = value.split(".")[-1]
        if self._extensions and extension not in self._extensions:
            raise ValueError(
                f"extension not supported, required extension in {self._extensions}",
            )
        return value

    def to_python_value(self, value: str) -> str | None:
        if not value or value.startswith("http") or "." not in value:
            return value
        try:
            is_success, url_or_error = self._file_storage.get_full_path(
                value,
                self._expire,
            )
        except Exception as e:
            raise ValueError(
                f"Obtain file from storage {self._file_storage} failed with exception {e}",
            ) from e
        else:
            if is_success:
                return url_or_error  # type: ignore
            raise ValueError(url_or_error)


class TimestampField(fields.DatetimeField):
    """
    Big integer for datetime field. (64-bit signed)
    """

    read_only: bool
    timezone: ZoneInfo

    SQL_TYPE = "BIGINT"

    class _db_mysql:
        SQL_TYPE = "BIGINT"

    class _db_postgres:
        SQL_TYPE = "BIGINT"

    class _db_mssql:
        SQL_TYPE = "BIGINT"

    class _db_oracle:
        SQL_TYPE = "INT"

    def __init__(
        self,
        null: bool = True,
        default: str | None = None,
        read_only: bool = True,
        index: bool = False,
        description: str | None = None,
        validators: list[validators.Validator | Callable] | None = None,
        timezone: ZoneInfo = ZoneInfo("Asia/Shanghai"),
    ) -> None:
        super().__init__(
            null=null,
            default=default,
            description=description,
            index=index,
            validators=validators,
        )
        self.read_only = read_only
        self.timezone = timezone

    def to_db_value(self, value: int | datetime.datetime, instance: "TimestampField") -> str:  # type: ignore
        if value is None:
            return "0"
        if isinstance(value, datetime.datetime):
            value = int(value.timestamp())
        self.validate(value)
        return str(value)

    @property
    def constraints(self) -> dict:
        return {"readOnly": self.read_only}

    def to_python_value(
        self,
        value: str | int | None,
    ) -> datetime.datetime | None:
        if value is None or value in [0, "0"]:
            return None
        # if value == "0":
        #     # 区分 0 和 null
        #     return 0  # type: ignore

        return datetime.datetime.fromtimestamp(
            value,  # type: ignore
            self.timezone,
        )


class TimeField(fields.TimeField):
    def timedelta_to_time(self, td: datetime.timedelta) -> datetime.time:
        # 确保 timedelta 表示的总秒数不超过一天的秒数
        total_seconds = int(td.total_seconds()) % (24 * 3600)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        # 创建并返回一个 datetime.time 对象
        return datetime.time(hour=hours, minute=minutes, second=seconds)

    def to_python_value(
        self,
        value: Any,
    ) -> datetime.time | None:  # ruff: noqa: ANN401
        if isinstance(value, datetime.timedelta):
            value = self.timedelta_to_time(value)
        if value is not None:
            if isinstance(value, str):
                value = datetime.time.fromisoformat(value)
            if timezone.is_naive(value):
                value = value.replace(tzinfo=get_default_timezone())
        self.validate(value)
        return value  # type: ignore

    def to_db_value(
        self,
        value: datetime.time | None,  # type: ignore
        instance: "type[Model] | Model",
    ) -> datetime.time | datetime.timedelta | None:
        # Only do this if it is a Model instance, not class. Test for guaranteed instance var
        if hasattr(instance, "_saved_in_db") and (
            self.auto_now or (self.auto_now_add and getattr(instance, self.model_field_name) is None)
        ):
            now = timezone.now().time()
            setattr(instance, self.model_field_name, now)
            return now
        if value is not None and get_use_tz() and timezone.is_naive(value):
            warnings.warn(  # ruff: noqa: B028
                f"TimeField {self.model_field_name} received a naive time ({value})"
                " while time zone support is active.",
                RuntimeWarning,
            )
            value = value.replace(tzinfo=get_default_timezone())
        self.validate(value)
        return value


class BinaryUUIDField(fields.Field[uuid.UUID], uuid.UUID):
    SQL_TYPE = "BINARY(16)"

    class _db_postgres:
        SQL_TYPE = "UUID"

    def to_db_value(self, value: uuid.UUID | str | bytes | None, instance: type[Model] | Model) -> RawSQL | None:
        match value:
            case RawSQL():
                return value
            case uuid.UUID():
                return RawSQL(f"UUID_TO_BIN('{value}')")
            case str():
                return RawSQL(f"UUID_TO_BIN('{uuid.UUID(value)}')")
            case bytes():
                return RawSQL(f"UUID_TO_BIN('{uuid.UUID(bytes=value)}')")
            case None:
                return None
            case _:
                raise ConfigurationError("This field only accepts UUID values")

    def to_python_value(self, value: str | None) -> uuid.UUID | None:
        match value:
            case uuid.UUID():
                return value
            case str():
                return uuid.UUID(value)
            case bytes():
                return uuid.UUID(bytes=value)
            case None:
                return None
            case _:
                raise ConfigurationError("This field only accepts UUID values")
