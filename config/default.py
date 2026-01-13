import os
import abc
import enum
import multiprocessing
from typing import Self, Generic, TypeVar
from pathlib import Path

from pydantic import HttpUrl, BaseModel, model_validator

T = TypeVar("T")


class EnvironmentEnum(str, enum.Enum):
    local = "local"
    development = "development"
    test = "test"
    production = "production"


ENVIRONMENT = os.environ.get(
    "environment",  # noqa
    EnvironmentEnum.local.value,
)

BASE_DIR = Path(__file__).resolve().parent.parent


class ProfilingConfig(BaseModel):
    secret: str
    interval: float = 0.001


class ServerConfig(BaseModel):
    class CorsConfig(BaseModel):
        allow_origins: list[str] = ["*"]
        allow_credentials: bool = True
        allow_methods: list[str] = ["*"]
        allow_headers: list[str] = ["*"]
        expose_headers: list[str] = []

        @property
        def headers(self) -> dict:
            header = {
                "Access-Control-Allow-Origin": ",".join(self.allow_origins) if "*" not in self.allow_origins else "*",
                "Access-Control-Allow-Credentials": str(
                    self.allow_credentials,
                ).lower(),
                "Access-Control-Expose-Headers": (
                    ",".join(self.allow_headers) if "*" not in self.allow_headers else "*"
                ),
                "Access-Control-Allow-Methods": (
                    ",".join(self.allow_methods) if "*" not in self.allow_methods else "*"
                ),
            }
            if self.expose_headers:
                header["Access-Control-Expose-Headers"] = ", ".join(
                    self.expose_headers,
                )

            return header

    address: HttpUrl = HttpUrl("http://0.0.0.0:8000")
    cors: CorsConfig = CorsConfig()
    worker_number: int = multiprocessing.cpu_count() * int(os.getenv("WORKERS_PER_CORE", "2")) + 1
    profiling: ProfilingConfig | None = None
    allow_hosts: list = ["*"]
    static_path: str = "/static"
    docs_uri: str = "/docs"
    redoc_uri: str = "/redoc"
    openapi_uri: str = "/openapi.json"
    token_expire_seconds: int = 3600 * 24
    timezone: str = "Asia/Shanghai"

    class SwaggerServerConfig(BaseModel):
        url: HttpUrl
        description: str

    swagger_servers: list[SwaggerServerConfig] = []


class ProjectConfig(BaseModel):
    unique_code: str
    debug: bool = False
    environment: EnvironmentEnum = EnvironmentEnum(ENVIRONMENT)
    sentry_dsn: HttpUrl | None = None

    class SwaggerServerConfig(BaseModel):
        url: HttpUrl
        description: str

    swagger_servers: list[SwaggerServerConfig] = []

    @model_validator(mode="after")
    def check_debug_options(self) -> Self:
        assert not (
            self.debug and self.environment == EnvironmentEnum.production
        ), "Production cannot set with debug enabled"
        return self

    @property
    def base_dir(self) -> Path:
        return BASE_DIR


class ExtensionConfig(BaseModel): ...


class InstanceExtensionConfig(ExtensionConfig, Generic[T]):

    @property
    @abc.abstractmethod
    def instance(self) -> T: ...


class RegisterExtensionConfig(ExtensionConfig):

    @abc.abstractmethod
    async def register(self) -> None: ...

    @abc.abstractmethod
    async def unregister(self) -> None: ...
