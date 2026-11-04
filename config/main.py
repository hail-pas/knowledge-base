from typing import Any
from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import (
    BaseSettings,
    YamlConfigSettingsSource,
    PydanticBaseSettingsSource,
)

from ext.register import ExtensionRegistry
from config.default import (
    BASE_DIR,
    ENVIRONMENT,
    EnvironmentEnum,
    ServerConfig,
    ProjectConfig,
    UserCenterIntegrationConfig,
)


class LocalConfig(BaseSettings):
    """全部的配置信息."""

    server: ServerConfig
    project: ProjectConfig
    extensions: ExtensionRegistry
    user_center: UserCenterIntegrationConfig = UserCenterIntegrationConfig()
    service: dict[str, Any] = {}

    @model_validator(mode="after")
    def validate_production_security(self) -> "LocalConfig":
        if self.project.environment == EnvironmentEnum.production:
            if self.user_center.base_url.scheme != "https":
                raise ValueError("Production user-center URL must use HTTPS")
            if "*" in self.server.allow_hosts:
                raise ValueError("Production must configure explicit allowed hosts")
            if self.server.cors.allow_credentials and "*" in self.server.cors.allow_origins:
                raise ValueError("Production credentialed CORS must use explicit allowed origins")
        return self

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
