import enum
from typing import override

from pydantic import BaseModel, PostgresDsn
from tortoise import Tortoise
from tortoise.backends.base.config_generator import expand_db_url
from tortoise.migrations.api import migrate

from config.default import RegisterExtensionConfig


class ConnectionNameEnum(str, enum.Enum):
    """数据库连接名称"""

    knowledge_base = "knowledge_base"
    authorization = "authorization"


def _model_modules(connection: ConnectionNameEnum) -> list[str]:
    return [f"ext.ext_tortoise.models.{connection.value}"]


class DatabaseConnectionConfig(BaseModel):
    """PostgreSQL connection settings for one Tortoise app."""

    url: PostgresDsn

    def config_dict(self, timezone: str) -> dict:
        config = expand_db_url(str(self.url))
        config["credentials"].update(
            minsize=1,
            maxsize=10,
            server_settings={"timezone": timezone},
        )
        return config


async def _legacy_schema_apps(config: dict) -> list[str]:
    """Detect databases created by Aerich before adopting native migrations."""
    await Tortoise.init(config=config, init_connections=False)

    legacy_apps: list[str] = []
    core_tables = {
        ConnectionNameEnum.knowledge_base.value: "collection",
        ConnectionNameEnum.authorization.value: "role",
    }
    for app_label, core_table in core_tables.items():
        connection = Tortoise.get_connection(app_label)
        rows = await connection.execute_query_dict(
            "SELECT "
            "to_regclass('public.tortoise_migrations') IS NOT NULL AS has_native_migrations, "
            f"to_regclass('public.{core_table}') IS NOT NULL AS has_legacy_schema"
        )
        if rows and rows[0]["has_legacy_schema"] and not rows[0]["has_native_migrations"]:
            legacy_apps.append(app_label)

    return legacy_apps


class TortoiseConfig(RegisterExtensionConfig):
    """Initialize both PostgreSQL apps through one extension lifecycle."""

    knowledge_base: DatabaseConnectionConfig
    authorization: DatabaseConnectionConfig

    @override
    async def register(self) -> None:
        config = gen_tortoise_config_dict()

        # Existing Aerich databases already contain the initial schema. Mark only
        # the native baseline as applied, then let native migrations reconcile
        # subsequent schema changes. Fresh databases run the baseline normally.
        for app_label in await _legacy_schema_apps(config):
            await migrate(
                config=config,
                app_labels=[app_label],
                target=f"{app_label}.0001_initial",
                fake=True,
            )

        await migrate(config=config)
        await Tortoise.init(config=config)

    @override
    async def unregister(self) -> None:
        await Tortoise.close_connections()


def gen_tortoise_config_dict() -> dict:
    from config.main import local_configs

    tortoise_config = local_configs.extensions.tortoise
    return {
        "connections": {
            connection.value: getattr(tortoise_config, connection.value).config_dict(
                local_configs.server.timezone
            )
            for connection in ConnectionNameEnum
        },
        "apps": {
            connection.value: {
                "models": _model_modules(connection),
                "default_connection": connection.value,
                "migrations": f"ext.ext_tortoise.migrate.{connection.value}",
            }
            for connection in ConnectionNameEnum
        },
        "timezone": local_configs.server.timezone,
        "use_tz": False,
    }
