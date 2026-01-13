from functools import lru_cache

from pydantic_settings import (
    BaseSettings,
    YamlConfigSettingsSource,
    PydanticBaseSettingsSource,
)

from ext.register import ExtensionRegistry
from config.default import BASE_DIR, ENVIRONMENT, ServerConfig, ProjectConfig


class LocalConfig(BaseSettings):
    """全部的配置信息."""

    server: ServerConfig
    project: ProjectConfig
    extensions: ExtensionRegistry

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (YamlConfigSettingsSource(settings_cls, f"{str(BASE_DIR)}/etc/{ENVIRONMENT.lower()}.yaml", "utf-8"),)


@lru_cache
def create_local_configs() -> LocalConfig:
    """create json file base setting object"""

    return LocalConfig()  # type: ignore


local_configs: LocalConfig = create_local_configs()  # type: ignore
