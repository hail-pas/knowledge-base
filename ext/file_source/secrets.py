"""Credential transformation for file-source provider-specific options."""

from util.secret import transform_extra_config_secrets


def _transform(source_type: str, extra_config: dict | None, *, decrypt: bool) -> dict:
    return transform_extra_config_secrets(
        extra_config,
        purpose=f"file_source.{source_type}.extra_config",
        decrypt=decrypt,
    )


def encrypt_file_source_extra_config(source_type: str, extra_config: dict | None) -> dict:
    return _transform(source_type, extra_config, decrypt=False)


def decrypt_file_source_extra_config(source_type: str, extra_config: dict | None) -> dict:
    return _transform(source_type, extra_config, decrypt=True)
