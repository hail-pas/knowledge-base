import pytest
from fastapi.testclient import TestClient

from api.knowledge_base.factory import knowledge_api


@pytest.fixture(autouse=True)
def mock_pero_userinfo(monkeypatch):
    """Mock the external Pero identity boundary for Knowledge Base API tests."""

    async def fake_userinfo(access_token: str) -> dict[str, str]:
        assert access_token == "token"
        return {"sub": "00000000-0000-0000-0000-000000000001"}

    monkeypatch.setattr("service.depend._fetch_userinfo", fake_userinfo)


@pytest.fixture
def client():
    """Create a test client for knowledge_base API."""

    return TestClient(knowledge_api, headers={"Authorization": "Bearer token"})
