import pytest
from httpx import AsyncClient, ASGITransport
from api.knowledge_base.factory import knowledge_api


@pytest.fixture(scope="session")
async def client():
    """FastAPI 测试客户端"""
    transport = ASGITransport(app=knowledge_api)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
