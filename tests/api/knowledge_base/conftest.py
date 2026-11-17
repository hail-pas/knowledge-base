import pytest
from fastapi.testclient import TestClient

from api.knowledge_base.factory import knowledge_api


@pytest.fixture(autouse=True)
def mock_pero_userinfo(monkeypatch):
    """Mock Pero token validation and grant test requests an explicit test permission."""

    async def fake_userinfo(access_token: str) -> dict[str, str]:
        assert access_token == "token"
        return {"sub": "00000000-0000-0000-0000-000000000001"}

    async def allow_all_permissions(user_id):
        return {"*"}

    monkeypatch.setattr("service.depend._fetch_userinfo", fake_userinfo)
    monkeypatch.setattr("service.depend._get_user_permission_codes", allow_all_permissions)

    async def allow_super_admin(user_id):
        return True

    monkeypatch.setattr("api.knowledge_base.v1.config.is_super_admin_user", allow_super_admin)


@pytest.fixture
def client():
    """Create a test client for Knowledge Base API."""

    return TestClient(knowledge_api, headers={"Authorization": "Bearer token"})
