import enum
import datetime
from typing import override
from zoneinfo import ZoneInfo
from urllib.parse import unquote

from aerich import Command
from pydantic import MySQLDsn
from tortoise import Tortoise

from config.default import RegisterExtensionConfig


class ConnectionNameEnum(str, enum.Enum):
    """数据库连接名称"""

    # default = "default"  # "默认连接"
    user_center = "user_center"  # "用户中心连接"
    knowledge_base = "knowledge_base"


class TortoiseConfig(RegisterExtensionConfig):
    url: MySQLDsn
    echo: bool = False

    # model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def datetime_now(self) -> datetime.datetime:
        from config.main import local_configs

        return datetime.datetime.now(
            tz=ZoneInfo(local_configs.server.timezone),
        )

    def config_dict(self) -> dict:
        return {
            # "engine": "tortoise.backends.sqlite",
            # "credentials": {"file_path": ":memory:"},
            "engine": "tortoise.backends.mysql",
            "credentials": {
                "host": self.url.host,
                "port": self.url.port,
                "user": self.url.username,
                "password": unquote(self.url.password) if self.url.password else "",
                "database": self.url.path.strip("/"),  # type: ignore
                "echo": self.echo,
                "minsize": 1,  # 连接池的最小连接数
                "maxsize": 10,  # 连接池的最大连接数
                "pool_recycle": 3600,  # 连接的最大存活时间（秒）
            },
        }

    @override
    async def register(self) -> None:
        from ext.ext_tortoise.migrate.env import VERSION_FILE_PATH

        for c in ConnectionNameEnum:
            Tortoise.init_models(
                [
                    f"ext.ext_tortoise.models.{c.value}",
                ],
                c.value,
            )

        for c in ConnectionNameEnum:
            command = Command(
                tortoise_config=gen_tortoise_config_dict(),
                app=c.value,
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
            k.value: getattr(local_configs.extensions, f"rdb_{k.value}").config_dict() for k in ConnectionNameEnum
        },
        "apps": {
            k.value: {
                "models": (
                    [
                        f"ext.ext_tortoise.models.{k.value}",
                        "aerich.models"
                    ]
                    # + [
                    #     "aerich.models",
                    # ]
                    # if k.value == ConnectionNameEnum.user_center.value
                    # else []
                ),
                "default_connection": k.value,
            }
            for k in ConnectionNameEnum
        },
        # "use_tz": True,   # Will Always Use UTC as Default Timezone
        "timezone": local_configs.server.timezone,
        # 'routers': ['path.router1', 'path.router2'],
    }
