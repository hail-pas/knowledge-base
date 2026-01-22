import asyncio
import pytest
from tortoise import Tortoise
from core.context import init_ctx, clear_ctx
from config.main import local_configs


@pytest.fixture(scope="session")
def event_loop():
    """Create a session-scoped event loop."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
async def setup_context():
    """Tortoise ORM fixture 使用 SQLite 内存数据库进行测试"""
    # 初始化 Tortoise ORM 配置
    config = {
        "connections": {
            "knowledge_base": {
                "engine": "tortoise.backends.sqlite",
                "credentials": {"file_path": ":memory:"},
            }
        },
        "apps": {
            "knowledge_base": {
                "models": [
                    "ext.ext_tortoise.models.knowledge_base",
                    # "aerich.models",
                ],
                "default_connection": "knowledge_base",
            }
        },
        "use_tz": False,
        "timezone": "Asia/Shanghai",
    }

    # 初始化数据库连接
    await Tortoise.init(config=config)

    # 生成所有表结构
    await Tortoise.generate_schemas()

    await local_configs.extensions.redis.register()
    await local_configs.extensions.httpx.register()

    yield

    await local_configs.extensions.httpx.unregister()
    await local_configs.extensions.redis.unregister()
    # 清理：关闭连接
    await Tortoise.close_connections()
