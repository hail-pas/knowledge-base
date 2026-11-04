import os

import pytest
from tortoise import Tortoise

from config.main import local_configs

os.environ.setdefault("KB_SECRET_ENCRYPTION_KEY", "00" * 32)


@pytest.fixture(scope="session", autouse=True)
async def setup_context():
    """Initialize Knowledge Base domain and authorization models."""

    database_url = os.getenv(
        "TEST_DATABASE_URL",
        "postgres://postgres:postgres@127.0.0.1:5432/knowledge_base_test",
    )
    config = {
        "connections": {
            "knowledge_base": database_url,
            "authorization": database_url,
        },
        "apps": {
            "knowledge_base": {
                "models": ["ext.ext_tortoise.models.knowledge_base"],
                "default_connection": "knowledge_base",
            },
            "authorization": {
                "models": ["ext.ext_tortoise.models.authorization"],
                "default_connection": "authorization",
            },
        },
        "use_tz": False,
        "timezone": "Asia/Shanghai",
    }

    await Tortoise.init(config=config)
    await Tortoise.generate_schemas()
    await local_configs.extensions.redis.register()
    await local_configs.extensions.httpx.register()

    yield

    await local_configs.extensions.httpx.unregister()
    await local_configs.extensions.redis.unregister()
    await Tortoise.close_connections()
