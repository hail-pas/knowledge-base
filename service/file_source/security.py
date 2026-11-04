"""Security helpers for shared file-source credential handling and URI scoping."""

import posixpath
from string import Formatter
from urllib.parse import unquote, urlsplit

from core.types import ApiException

_PUBLIC_SOURCE_SCHEMES = {
    "s3": "s3",
    "minio": "minio",
    "aliyun_oss": "oss",
}
_ALLOWED_PREFIX_FIELDS = {"user_id", "collection_id"}


def validate_allowed_prefix_template(value: str | None) -> str | None:
    """Validate a relative object-key template scoped to a user or collection."""
    if value is None:
        return None
    if (
        not value.strip()
        or value.startswith("/")
        or "\\" in value
        or any(char in value for char in ("\x00", "?", "#", ":"))
    ):
        raise ValueError("allowed_prefix_template 必须是相对对象路径")

    fields: set[str] = set()
    try:
        for _, field_name, format_spec, conversion in Formatter().parse(value):
            if field_name is None:
                continue
            if field_name not in _ALLOWED_PREFIX_FIELDS or format_spec or conversion:
                raise ValueError("allowed_prefix_template 仅支持 {user_id} 和 {collection_id}")
            fields.add(field_name)
    except ValueError as exc:
        raise ValueError("allowed_prefix_template 格式无效") from exc

    if not fields:
        raise ValueError("allowed_prefix_template 必须包含 {user_id} 或 {collection_id}")

    literal = value.replace("{user_id}", "user").replace("{collection_id}", "collection")
    if any(part in {"", ".", ".."} for part in literal.rstrip("/").split("/")):
        raise ValueError("allowed_prefix_template 不允许空路径段或路径穿越")
    return value


def resolve_public_source_prefix(file_source, collection) -> str | None:
    """Resolve the tenant/collection prefix required for a shared file source."""
    if file_source.user_id is not None:
        return None

    source_type = getattr(file_source.type, "value", file_source.type)
    if source_type not in _PUBLIC_SOURCE_SCHEMES:
        raise ApiException("公共文件源仅支持配置了隔离前缀的对象存储")
    template = validate_allowed_prefix_template(file_source.allowed_prefix_template)
    if not template:
        raise ApiException("公共文件源必须配置 allowed_prefix_template")
    if collection.user_id is None:
        raise ApiException("公共文件源只能用于归属明确的知识库集合")

    rendered = template.format(user_id=str(collection.user_id), collection_id=str(collection.id))
    parts = rendered.rstrip("/").split("/")
    if not rendered or rendered.startswith("/") or any(part in {"", ".", ".."} for part in parts):
        raise ApiException("公共文件源隔离前缀无效")

    normalized = posixpath.normpath(rendered)
    if normalized in {"", ".", ".."} or normalized.startswith("../"):
        raise ApiException("公共文件源隔离前缀无效")
    return f"{normalized.rstrip('/')}/"


def build_file_upload_uri(file_source, collection, filename: str) -> str:
    """Build a storage key inside the collection's allowed namespace."""
    prefix = resolve_public_source_prefix(file_source, collection) or f"{collection.id}/"
    return f"{prefix}{filename}"


def validate_public_source_uri(file_source, collection, uri: str) -> None:
    """Reject shared-source URIs outside the caller's user/collection namespace."""
    if file_source.user_id is not None:
        return

    prefix = resolve_public_source_prefix(file_source, collection)
    source_type = getattr(file_source.type, "value", file_source.type)
    expected_scheme = _PUBLIC_SOURCE_SCHEMES[source_type]
    parsed = urlsplit(uri)
    if parsed.query or parsed.fragment:
        raise ApiException("公共文件源 URI 不允许携带 query 或 fragment")

    if parsed.scheme:
        if (
            parsed.scheme != expected_scheme
            or parsed.netloc != file_source.storage_location
            or parsed.query
            or parsed.fragment
        ):
            raise ApiException("公共文件源 URI 与存储桶配置不匹配")
        key = unquote(parsed.path.lstrip("/"))
    else:
        if "://" in uri:
            raise ApiException("公共文件源 URI 格式无效")
        key = unquote(uri)

    if not key or key.startswith("/") or "\\" in key or "\x00" in key:
        raise ApiException("公共文件源 URI 格式无效")
    parts = key.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ApiException("公共文件源 URI 不允许路径穿越")
    if posixpath.normpath(key) != key or not key.startswith(prefix):
        raise ApiException("无权访问公共文件源该路径，URI 必须位于当前用户或集合的隔离前缀内")
