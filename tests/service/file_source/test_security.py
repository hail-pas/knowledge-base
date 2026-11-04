from types import SimpleNamespace

import pytest

from core.types import ApiException
from service.file_source.security import (
    build_file_upload_uri,
    resolve_public_source_prefix,
    validate_public_source_uri,
)


def _public_source(template: str | None):
    return SimpleNamespace(
        user_id=None,
        type=SimpleNamespace(value="s3"),
        storage_location="shared-bucket",
        allowed_prefix_template=template,
    )


def _collection():
    return SimpleNamespace(id=42, user_id="00000000-0000-0000-0000-000000000001")


def test_shared_source_prefix_is_scoped_to_collection_and_user():
    source = _public_source("users/{user_id}/collections/{collection_id}/")
    collection = _collection()

    assert resolve_public_source_prefix(source, collection) == (
        "users/00000000-0000-0000-0000-000000000001/collections/42/"
    )
    assert build_file_upload_uri(source, collection, "document.pdf").endswith(
        "/collections/42/document.pdf"
    )


def test_shared_source_rejects_uri_outside_allowed_prefix():
    source = _public_source("users/{user_id}/collections/{collection_id}/")
    collection = _collection()

    validate_public_source_uri(
        source,
        collection,
        "s3://shared-bucket/users/00000000-0000-0000-0000-000000000001/collections/42/file.pdf",
    )

    with pytest.raises(ApiException, match="隔离前缀"):
        validate_public_source_uri(
            source,
            collection,
            "s3://shared-bucket/users/00000000-0000-0000-0000-000000000002/collections/42/file.pdf",
        )


def test_shared_source_fails_closed_without_prefix_and_rejects_path_traversal():
    collection = _collection()
    with pytest.raises(ApiException, match="必须配置"):
        resolve_public_source_prefix(_public_source(None), collection)

    source = _public_source("users/{user_id}/collections/{collection_id}/")
    with pytest.raises(ApiException):
        validate_public_source_uri(
            source,
            collection,
            "s3://shared-bucket/users/00000000-0000-0000-0000-000000000001/collections/42/../other/file.pdf",
        )


def test_shared_source_rejects_query_fragment_and_malformed_uris():
    source = _public_source("users/{user_id}/collections/{collection_id}/")
    collection = _collection()

    for uri in (
        "s3://shared-bucket/users/00000000-0000-0000-0000-000000000001/collections/42/file.pdf?download=1",
        "s3://shared-bucket/users/00000000-0000-0000-0000-000000000001/collections/42/file.pdf#fragment",
        "s3://[malformed",
    ):
        with pytest.raises(ApiException):
            validate_public_source_uri(source, collection, uri)
