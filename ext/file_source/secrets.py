"""Credential transformation for file-source provider-specific options."""

import copy

from util.secret import decrypt_secret, encrypt_secret


def _transform_file_source_extra_config(source_type: str, extra_config: dict | None, *, decrypt: bool) -> dict:
    result = copy.deepcopy(extra_config or {})
    secret_field = {
        "s3": "session_token",
        "aliyun_oss": "private_key_content",
    }.get(source_type)
    if secret_field and result.get(secret_field):
        transform = decrypt_secret if decrypt else encrypt_secret
        result[secret_field] = transform(
            result[secret_field],
            purpose=f"file_source.{source_type}.extra_config.{secret_field}",
        )
    return result


def encrypt_file_source_extra_config(source_type: str, extra_config: dict | None) -> dict:
    return _transform_file_source_extra_config(source_type, extra_config, decrypt=False)


def decrypt_file_source_extra_config(source_type: str, extra_config: dict | None) -> dict:
    return _transform_file_source_extra_config(source_type, extra_config, decrypt=True)
