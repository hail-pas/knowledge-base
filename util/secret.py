"""Authenticated encryption for credentials persisted by Knowledge Base."""

import base64
import copy
import os
import secrets

from Cryptodome.Cipher import AES

_PREFIX = "enc:v1:"
_ENV_NAME = "KB_SECRET_ENCRYPTION_KEY"
_NONCE_SIZE = 12
_TAG_SIZE = 16


def _encryption_key() -> bytes:
    value = os.getenv(_ENV_NAME)
    if not value:
        raise RuntimeError(f"{_ENV_NAME} must be set to a 32-byte hex key")

    if len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value):
        raise RuntimeError(f"{_ENV_NAME} must contain exactly 64 hexadecimal characters")

    try:
        key = bytes.fromhex(value)
    except ValueError:
        raise RuntimeError(f"{_ENV_NAME} must contain exactly 64 hexadecimal characters") from None

    if len(key) != 32:
        raise RuntimeError(f"{_ENV_NAME} must contain exactly 64 hexadecimal characters")
    return key


def validate_encryption_key() -> None:
    """Fail fast when the service encryption key is missing or malformed."""
    _encryption_key()


def encrypt_secret(value: str | None, *, purpose: str) -> str | None:
    """Encrypt a secret with AES-256-GCM and bind it to its storage purpose."""
    if value is None or value == "":
        return value
    if not isinstance(value, str):
        raise TypeError("Secret values must be strings")
    if not purpose:
        raise ValueError("An encryption purpose is required")

    if value.startswith(_PREFIX):
        decrypt_secret(value, purpose=purpose)
        return value

    nonce = secrets.token_bytes(_NONCE_SIZE)
    cipher = AES.new(_encryption_key(), AES.MODE_GCM, nonce=nonce, mac_len=_TAG_SIZE)
    cipher.update(purpose.encode("utf-8"))
    ciphertext, tag = cipher.encrypt_and_digest(value.encode("utf-8"))
    payload = base64.urlsafe_b64encode(nonce + tag + ciphertext).decode("ascii")
    return f"{_PREFIX}{payload}"


def decrypt_secret(value: str | None, *, purpose: str) -> str | None:
    """Decrypt a stored secret; plaintext values are rejected."""
    if value is None or value == "":
        return value
    if not isinstance(value, str) or not value.startswith(_PREFIX):
        raise ValueError("Stored secret is not encrypted")
    if not purpose:
        raise ValueError("A decryption purpose is required")

    try:
        payload = value[len(_PREFIX) :].encode("ascii")
        decoded = base64.b64decode(payload, altchars=b"-_", validate=True)
        if len(decoded) < _NONCE_SIZE + _TAG_SIZE:
            raise ValueError("Invalid encrypted secret")
        nonce = decoded[:_NONCE_SIZE]
        tag = decoded[_NONCE_SIZE : _NONCE_SIZE + _TAG_SIZE]
        ciphertext = decoded[_NONCE_SIZE + _TAG_SIZE :]
        cipher = AES.new(_encryption_key(), AES.MODE_GCM, nonce=nonce, mac_len=_TAG_SIZE)
        cipher.update(purpose.encode("utf-8"))
        return cipher.decrypt_and_verify(ciphertext, tag).decode("utf-8")
    except (ValueError, UnicodeError):
        raise ValueError("Unable to decrypt stored secret; verify the encryption key and purpose") from None


def transform_extra_config_secrets(extra_config: dict | None, *, purpose: str, decrypt: bool = False) -> dict:
    """Transform secret-like values recursively without changing the config shape."""
    transform = decrypt_secret if decrypt else encrypt_secret
    sensitive_parts = (
        "api_key",
        "apikey",
        "access_key",
        "accesskey",
        "client_secret",
        "credential",
        "authorization",
        "password",
        "secret",
        "token",
        "private_key",
        "privatekey",
    )
    sensitive_exact = {"auth", "signature", "sig", "x_api_key"}

    def visit(value, path: str, *, in_headers: bool = False, sensitive_context: bool = False):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                normalized_key = str(key).lower().replace("-", "_")
                child_path = f"{path}.{normalized_key}"
                is_sensitive = (
                    normalized_key in sensitive_exact
                    or any(part in normalized_key for part in sensitive_parts)
                )
                child_in_headers = in_headers or normalized_key == "headers"
                child_sensitive = sensitive_context or in_headers or is_sensitive

                if isinstance(item, str) and item and child_sensitive:
                    result[key] = transform(item, purpose=child_path)
                else:
                    result[key] = visit(
                        item,
                        child_path,
                        in_headers=child_in_headers,
                        sensitive_context=child_sensitive,
                    )
            return result

        if isinstance(value, list):
            return [
                visit(
                    item,
                    f"{path}.{index}",
                    in_headers=in_headers,
                    sensitive_context=sensitive_context,
                )
                for index, item in enumerate(value)
            ]

        if isinstance(value, str) and value.startswith(_PREFIX):
            return decrypt_secret(value, purpose=path) if decrypt else encrypt_secret(value, purpose=path)

        if isinstance(value, str) and value and sensitive_context:
            return transform(value, purpose=path)

        if isinstance(value, str) and value and "://" in value:
            from urllib.parse import parse_qsl, urlsplit

            try:
                parsed = urlsplit(value)
                query_has_secret = any(
                    key.lower().replace("-", "_") in sensitive_exact
                    or any(part in key.lower().replace("-", "_") for part in sensitive_parts)
                    for key, _ in parse_qsl(parsed.query, keep_blank_values=True)
                )
                if parsed.username or parsed.password or query_has_secret:
                    return transform(value, purpose=path)
            except ValueError:
                if "@" in value:
                    return transform(value, purpose=path)

        return value

    return visit(copy.deepcopy(extra_config or {}), purpose)
