"""Canonical parsing and validation for file-source URIs and object keys."""

from urllib.parse import unquote, urlsplit


_SCHEMES = {"s3", "minio", "oss"}


def parse_object_key(uri: str, *, scheme: str | None = None, bucket: str | None = None) -> str:
    """Accept a provider URI or a bare key; reject foreign buckets and unsafe keys."""
    if not isinstance(uri, str) or not uri or "\x00" in uri or "\\" in uri:
        raise ValueError("Invalid file URI or object key")

    parsed = urlsplit(uri)
    if parsed.scheme:
        if parsed.scheme not in _SCHEMES or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("Invalid object storage URI")
        if scheme is not None and parsed.scheme != scheme:
            raise ValueError("Object storage URI scheme does not match provider")
        if bucket is not None and parsed.netloc != bucket:
            raise ValueError("Object storage URI bucket does not match configured bucket")
        key = unquote(parsed.path.lstrip("/"))
    else:
        if "://" in uri or uri.startswith("/"):
            raise ValueError("Object key must be relative")
        key = unquote(uri)

    if not key or key.startswith("/") or "\\" in key or "\x00" in key:
        raise ValueError("Object key must be a non-empty relative path")
    if any(part in {"", ".", ".."} for part in key.split("/")):
        raise ValueError("Object key contains an invalid path segment")
    return key
