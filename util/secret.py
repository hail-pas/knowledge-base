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
    except (ValueError, UnicodeDecodeError):
        raise ValueError("Unable to decrypt stored secret; verify the encryption key and purpose") from None


def transform_extra_config_secrets(extra_config: dict | None, *, purpose: str, decrypt: bool = False) -> dict:
    """Encrypt custom HTTP headers and credential-like query parameters."""
    result = copy.deepcopy(extra_config or {})
    transform = decrypt_secret if decrypt else encrypt_secret

    headers = result.get("headers")
    if isinstance(headers, dict):
        result["headers"] = {
            name: (
                transform(value, purpose=f"{purpose}.headers.{name.lower()}")
                if value
                else value
            )
            for name, value in headers.items()
        }

    query_params = result.get("query_params")
    if isinstance(query_params, dict):
        sensitive_names = ("key", "token", "secret", "auth", "password", "credential")
        for name, value in query_params.items():
            if value and any(part in name.lower() for part in sensitive_names):
                query_params[name] = transform(value, purpose=f"{purpose}.query_params.{name.lower()}")

    return result
