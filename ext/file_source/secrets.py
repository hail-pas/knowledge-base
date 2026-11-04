"""Credential transformation for file-source provider-specific options."""

import copy

from util.secret import decrypt_secret, encrypt_secret


def _transform_file_source_extra_config(source_type: str, extra_config: dict | None, *, decrypt: bool) -> dict:
    result = copy.deepcopy(extra_config or {})
    secret_field = {
        "s3": "session_token",
        "aliyun_oss": "private_key_content",
    }.get(source_type)
    transform = decrypt_secret if decrypt else encrypt_secret
    if secret_field and result.get(secret_field):
        result[secret_field] = transform(
            result[secret_field],
            purpose=f"file_source.{source_type}.extra_config.{secret_field}",
        )

    if source_type == "s3":
        provider_config = result.get("config")
        proxies = provider_config.get("proxies") if isinstance(provider_config, dict) else None
        if isinstance(proxies, dict):
            for protocol, proxy_url in proxies.items():
                if proxy_url:
                    proxies[protocol] = transform(
                        proxy_url,
                        purpose=f"file_source.s3.extra_config.config.proxies.{protocol.lower()}",
                    )
    return result


def encrypt_file_source_extra_config(source_type: str, extra_config: dict | None) -> dict:
    return _transform_file_source_extra_config(source_type, extra_config, decrypt=False)


def decrypt_file_source_extra_config(source_type: str, extra_config: dict | None) -> dict:
    return _transform_file_source_extra_config(source_type, extra_config, decrypt=True)
