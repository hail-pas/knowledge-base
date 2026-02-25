import pytest
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from api.knowledge_base.factory import knowledge_api


@pytest.fixture
def client():
    """Create a test client for knowledge_base API"""
    return TestClient(knowledge_api, headers={"Authorization": "Bearer token"})
