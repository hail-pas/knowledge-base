import enum
import datetime
from typing import override
from zoneinfo import ZoneInfo
from urllib.parse import unquote

from aerich import Command
from pydantic import PostgresDsn
from tortoise import Tortoise

from config.default import RegisterExtensionConfig


class ConnectionNameEnum(str, enum.Enum):
    """数据库连接名称"""

    knowledge_base = "knowledge_base"
    authorization = "authorization"


def _model_modules(connection: ConnectionNameEnum) -> list[str]:
    return [f"ext.ext_tortoise.models.{connection.value}"]


class TortoiseConfig(RegisterExtensionConfig):
    url: PostgresDsn

    @property
    def datetime_now(self) -> datetime.datetime:
        from config.main import local_configs

        return datetime.datetime.now(tz=ZoneInfo(local_configs.server.timezone))

    def config_dict(self) -> dict:
        # PostgresDsn is a MultiHostUrl in Pydantic v2; read connection details
        # from its first host entry rather than using single-host URL properties.
        from config.main import local_configs

        host = self.url.hosts()[0]
        return {
            "engine": "tortoise.backends.asyncpg",
            "credentials": {
                "host": host.get("host"),
                "port": host.get("port") or 5432,
                "user": unquote(host.get("username") or ""),
                "password": unquote(host.get("password") or ""),
                "database": unquote(self.url.path.lstrip("/")),
                "minsize": 1,
                "maxsize": 10,
                "server_settings": {"timezone": local_configs.server.timezone},
            },
        }

    @override
    async def register(self) -> None:
        from ext.ext_tortoise.migrate.env import VERSION_FILE_PATH

        for connection in ConnectionNameEnum:
            Tortoise.init_models(_model_modules(connection), connection.value)

        for connection in ConnectionNameEnum:
            command = Command(
                tortoise_config=gen_tortoise_config_dict(),
                app=connection.value,
                location=VERSION_FILE_PATH,
            )
            await command.init()
            await command.upgrade(run_in_transaction=True)
        await Tortoise.init(config=gen_tortoise_config_dict())

    @override
    async def unregister(self) -> None:
        await Tortoise.close_connections()


def gen_tortoise_config_dict() -> dict:
    from config.main import local_configs

    return {
        "connections": {
            connection.value: getattr(local_configs.extensions, f"rdb_{connection.value}").config_dict()
            for connection in ConnectionNameEnum
        },
        "apps": {
            connection.value: {
                "models": [*_model_modules(connection), "aerich.models"],
                "default_connection": connection.value,
            }
            for connection in ConnectionNameEnum
        },
        "timezone": local_configs.server.timezone,
        "use_tz": False,
    }
