import pytest
from tortoise import Tortoise





@pytest.fixture(scope="session", autouse=True)
async def db():
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
        "use_tz": True,
        "timezone": "Asia/Shanghai",
    }

    # 初始化数据库连接
    await Tortoise.init(config=config)

    # 生成所有表结构
    await Tortoise.generate_schemas()

    yield

    # 清理：关闭连接
    await Tortoise.close_connections()
