from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from config.main import local_configs
from core.logger import LogLevelEnum, setup_loguru
from config.default import RegisterExtensionConfig
from enhance.monkey_patch import patch


async def init_ctx():
    # patch
    patch()
    # logger
    setup_loguru(LogLevelEnum.DEBUG if local_configs.project.debug else LogLevelEnum.INFO)
    # extensions
    for _, ext_conf in local_configs.extensions:  # type: ignore
        if isinstance(ext_conf, RegisterExtensionConfig):
            await ext_conf.register()


async def clear_ctx():
    for _, ext_conf in local_configs.extensions:  # type: ignore
        if isinstance(ext_conf, RegisterExtensionConfig):
            await ext_conf.unregister()


@asynccontextmanager
async def ctx() -> AsyncGenerator:
    await init_ctx()

    yield

    await clear_ctx()
