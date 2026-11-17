import datetime
from collections.abc import Callable
from zoneinfo import ZoneInfo

from tortoise import fields, validators
from tortoise.models import Model


class TimestampField(fields.BigIntField):
    """Store epoch seconds in BIGINT while exposing timezone-aware datetimes."""

    read_only: bool
    timezone: ZoneInfo

    def __init__(
        self,
        null: bool = True,
        default: int | None = None,
        read_only: bool = True,
        index: bool = False,
        db_index: bool | None = None,
        description: str | None = None,
        validators: list[validators.Validator | Callable] | None = None,
        timezone: ZoneInfo = ZoneInfo("Asia/Shanghai"),
    ) -> None:
        super().__init__(
            null=null,
            default=default,
            description=description,
            index=index if db_index is None else db_index,
            validators=validators,
        )
        self.read_only = read_only
        self.timezone = timezone

    def to_db_value(
        self,
        value: int | datetime.datetime | None,
        instance: type[Model] | Model,
    ) -> int:
        if value is None:
            return 0
        if isinstance(value, datetime.datetime):
            value = int(value.timestamp())
        self.validate(value)
        return int(value)

    @property
    def constraints(self) -> dict:
        return {"readOnly": self.read_only}

    def to_python_value(self, value: str | int | None) -> datetime.datetime | None:
        if value is None or value == 0 or value == "0":
            return None
        return datetime.datetime.fromtimestamp(int(value), self.timezone)
