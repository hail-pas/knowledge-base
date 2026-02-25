import asyncio
import pytest
from tortoise import Tortoise
from config.main import local_configs
from util.encrypt import PasswordUtil
from ext.ext_tortoise.models.user_center import Account, Role
from unittest.mock import patch, AsyncMock


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
            "user_center": {
                "engine": "tortoise.backends.sqlite",
                "credentials": {"file_path": ":memory:"},
            },
            "knowledge_base": {
                "engine": "tortoise.backends.sqlite",
                "credentials": {"file_path": ":memory:"},
            },
        },
        "apps": {
            "user_center": {
                "models": [
                    "ext.ext_tortoise.models.user_center",
                ],
                "default_connection": "user_center",
            },
            "knowledge_base": {
                "models": [
                    "ext.ext_tortoise.models.knowledge_base",
                    # "aerich.models",
                ],
                "default_connection": "knowledge_base",
            },
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


@pytest.fixture(scope="session", autouse=True)
async def setup_account(setup_context):
    role = await Role.create(label="super admin", remark="super admin role")

    account = await Account.create(
        username="test-admin",
        phone="18888888888",
        email="test_admin@example.com",
        password=PasswordUtil.get_password_hash("test-password"),
        is_active=True,
        is_staff=True,
        is_super_admin=True,
        role=role,
    )

    async def mock_validate_jwt_token(request, token):
        request.scope["user"] = account
        request.scope["scene"] = "test"
        request.scope["is_staff"] = account.is_staff
        request.scope["is_super_admin"] = account.is_super_admin
        return account

    with patch("service.depend._validate_jwt_token", new=mock_validate_jwt_token):
        yield account
