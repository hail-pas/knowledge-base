import pytest
from fastapi.testclient import TestClient
from api.user_center.factory import user_center_api



@pytest.fixture
def client():
    """Create a test client for user_center API"""
    return TestClient(user_center_api, headers={"Authorization": "Bearer token"})
