import pytest

from util.secret import decrypt_secret, encrypt_secret, transform_extra_config_secrets


def test_secret_round_trip_and_idempotent_encryption(monkeypatch):
    monkeypatch.setenv("KB_SECRET_ENCRYPTION_KEY", "ab" * 32)

    encrypted = encrypt_secret("provider-api-key", purpose="llm_model.api_key")

    assert encrypted is not None
    assert encrypted.startswith("enc:v1:")
    assert encrypted != "provider-api-key"
    assert decrypt_secret(encrypted, purpose="llm_model.api_key") == "provider-api-key"
    assert encrypt_secret(encrypted, purpose="llm_model.api_key") == encrypted


def test_secret_is_bound_to_its_storage_purpose(monkeypatch):
    monkeypatch.setenv("KB_SECRET_ENCRYPTION_KEY", "ab" * 32)
    encrypted = encrypt_secret("provider-api-key", purpose="llm_model.api_key")

    with pytest.raises(ValueError):
        decrypt_secret(encrypted, purpose="embedding_model.api_key")


def test_plaintext_and_missing_encryption_key_are_rejected(monkeypatch):
    with pytest.raises(ValueError, match="not encrypted"):
        decrypt_secret("plaintext-secret", purpose="llm_model.api_key")

    monkeypatch.delenv("KB_SECRET_ENCRYPTION_KEY", raising=False)
    with pytest.raises(RuntimeError, match="KB_SECRET_ENCRYPTION_KEY"):
        encrypt_secret("provider-api-key", purpose="llm_model.api_key")


def test_extra_config_header_and_auth_query_secrets_round_trip(monkeypatch):
    monkeypatch.setenv("KB_SECRET_ENCRYPTION_KEY", "cd" * 32)
    original = {
        "headers": {"Authorization": "Bearer header-secret", "X-Trace": "trace-id"},
        "query_params": {"api_key": "query-secret", "page": "2"},
        "config": {
            "proxies": {"https": "http://proxy-user:proxy-pass@proxy.example:8080"},
            "credentials": ["first-secret", "second-secret"],
        },
    }

    encrypted = transform_extra_config_secrets(original, purpose="embedding_model.extra_config")
    assert encrypted["headers"]["Authorization"].startswith("enc:v1:")
    assert encrypted["headers"]["X-Trace"].startswith("enc:v1:")
    assert encrypted["query_params"]["api_key"].startswith("enc:v1:")
    assert encrypted["query_params"]["page"] == "2"
    assert encrypted["config"]["proxies"]["https"].startswith("enc:v1:")
    assert encrypted["config"]["credentials"][0].startswith("enc:v1:")

    decrypted = transform_extra_config_secrets(
        encrypted,
        purpose="embedding_model.extra_config",
        decrypt=True,
    )
    assert decrypted == original
