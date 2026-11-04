import pytest

from util.secret import decrypt_secret, encrypt_secret


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
