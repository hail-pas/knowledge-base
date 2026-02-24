import pytest
from fastapi.testclient import TestClient
from util.encrypt import PasswordUtil
from api.user_center.factory import user_center_api
from ext.ext_tortoise.models.user_center import Account, Role



@pytest.fixture
def client():
    """Create a test client for user_center API"""
    return TestClient(user_center_api, headers={"Authorization": "Bearer token"})
