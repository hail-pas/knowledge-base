"""Database exception translation for API-facing CRUD operations."""

import re

from tortoise.exceptions import IntegrityError

from core.types import ApiException

_UNIQUE_KEY_PATTERN = re.compile(r"'([^']+)'")
_UNIQUE_MARKERS = ("duplicate", "unique constraint", "unique violation")


def translate_integrity_error(error: IntegrityError, model: type) -> ApiException:
    """Translate common integrity failures without exposing SQL or driver details."""
    message = str(error)
    original = error.args[0] if error.args else None
    if isinstance(original, BaseException):
        message = str(original)

    lowered = message.lower()
    if any(marker in lowered for marker in _UNIQUE_MARKERS):
        keys = _UNIQUE_KEY_PATTERN.findall(message)
        custom_messages = getattr(getattr(model, "Meta", None), "unique_error_messages", {})
        if keys and keys[-1] in custom_messages:
            return ApiException(message=custom_messages[keys[-1]])
        return ApiException(message=f"{model._meta.table_description}已存在")

    if "foreign key" in lowered or "violates foreign key constraint" in lowered:
        return ApiException(message="关联数据不存在或仍被其他数据引用")

    if "not-null" in lowered or "not null" in lowered:
        return ApiException(message="必填字段不能为空")

    return ApiException(message="数据完整性校验失败")
