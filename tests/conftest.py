import pytest
from tortoise import Tortoise

from config.main import local_configs


@pytest.fixture(scope="session", autouse=True)
async def setup_context():
    """Initialize Knowledge Base domain and authorization models."""

    config = {
        "connections": {
            "knowledge_base": {
                "engine": "tortoise.backends.sqlite",
                "credentials": {"file_path": ":memory:"},
            },
            "authorization": {
                "engine": "tortoise.backends.sqlite",
                "credentials": {"file_path": ":memory:"},
            },
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
