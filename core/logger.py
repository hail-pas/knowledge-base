from __future__ import annotations

import sys
import logging
import warnings
import traceback
from enum import Enum
from types import FrameType
from typing import Any, TextIO, cast
from itertools import chain
from collections.abc import Callable

import loguru
from loguru import logger
from gunicorn import glogging  # type: ignore

from core.types import IntEnum
from config.default import ENVIRONMENT, EnvironmentEnum


class LogLevelEnum(IntEnum):
    """日志级别"""

    CRITICAL = (logging.CRITICAL, "CRITICAL")
    # FATAL = ("50", "FATAL")
    ERROR = (logging.ERROR, "ERROR")
    # WARN = ("30", "WARN")
    WARNING = (logging.WARNING, "WARNING")
    INFO = (logging.INFO, "INFO")
    DEBUG = (logging.DEBUG, "DEBUG")
    NOTSET = (logging.NOTSET, "NOTSET")


class LoggerNameEnum(str, Enum):
    root = "root"
    fastaapi = "fastapi"
    tortoise = "tortoise"
    gunicorn_error = "gunicorn.error"
    gunicorn_asgi = "gunicorn.asgi"
    gunicorn_gunicorn = "gunicorn.access"
    uvicorn_error = "uvicorn.error"
    uvicorn_asgi = "uvicorn.asgi"
    uvicorn_access = "uvicorn.access"


IgnoredLoggerNames = [
    LoggerNameEnum.uvicorn_access.value,
]


class InterceptHandler(logging.Handler):
    """Logs to loguru from Python logging module"""

    def emit(self, record: logging.LogRecord) -> None:
        if record.name in IgnoredLoggerNames:
            return
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        if record.exc_info:
            # 保持日志一致性
            tb = traceback.extract_tb(sys.exc_info()[2])
            # 获取最后一个堆栈帧的文件名和行号
            file_name, line_num, func_name, _ = tb[-1]
            location = f"{file_name}:{func_name}:{line_num}"
            if ENVIRONMENT in [EnvironmentEnum.local.value]:
                print(traceback.format_exc())
                return

            logger.bind(location=location).critical(
                traceback.format_exc(),
            )
            return

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:  # noqa: WPS609
            frame = cast(FrameType, frame.f_back)
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level,
            record.getMessage(),
        )


def setup_loguru_logging_intercept(
    level: int = logging.DEBUG,
    modules: tuple = (),
) -> None:
    logging.basicConfig(handlers=[InterceptHandler()], level=level)  # noqa
    for logger_name in chain(("",), modules):
        mod_logger = logging.getLogger(logger_name)
        mod_logger.handlers = [InterceptHandler(level=level)]
        mod_logger.setLevel(level)
        # mod_logger.propagate = False


class GunicornLogger(glogging.Logger):
    def __init__(self, cfg: Any) -> None:  # ruff: noqa: ANN401
        super().__init__(cfg)
        LOGGING_MODULES = (
            LoggerNameEnum.gunicorn_error.value,
            LoggerNameEnum.gunicorn_asgi.value,
            LoggerNameEnum.gunicorn_gunicorn.value,
        )
        setup_loguru_logging_intercept(
            level=logging.getLevelName(LogLevelEnum.INFO.value),  # type: ignore
            modules=LOGGING_MODULES,
        )


def edit_record_and_gen_format(record: loguru.Record) -> str:
    extra = record.get("extra") or {}
    # record["message"] = {"message": record["message"], **extra}  # type: ignore
    if record["level"].no <= 10:
        # debug
        level_color = "white"
    elif record["level"].no <= 20:
        # info
        level_color = "blue"
    elif record["level"].no <= 30:
        # warning
        level_color = "yellow"
    elif record["level"].no <= 40:
        # error
        level_color = "red"
    else:
        # other
        level_color = "magenta"
    if ENVIRONMENT in [EnvironmentEnum.local.value]:
        format_s = (
            "<green>[{time:YYYY-MM-DD HH:mm:ss}]</green> | "
            + f"<{level_color}>"
            + "<bold>[{level}]</bold>"
            + f"</{level_color}>"
            + " | <fg 0,75,0><underline>{name}:{line}</underline> >> {function}</fg 0,75,0> | <cyan>{message}</cyan>\n"
        )
    else:
        format_s = "[{time:YYYY-MM-DD HH:mm:ss}] | [{level}] | {name}:{line} >> {function} | {message}\n"

    if extra:
            format_s += " | " + "{extra}"

    return format_s + "\n"


def setup_loguru(
    level: LogLevelEnum = LogLevelEnum.INFO,
) -> None:
    # loguru
    logger.remove()
    # logger.add(
    #     sink=os.path.join(
    #         BASE_DIR,
    #         f'{local_configs.PROJECT.LOG_DIR}{datetime_now().strftime("%Y-%m-%d")}'
    #          '-{local_configs.PROJECT.UNIQUE_CODE}-service.log',
    #     ),
    #     rotation="500 MB",  # 日志文件最大限制500mb
    #     retention="30 days",  # 最长保留30天
    #     format="{message}",  # 日志显示格式
    #     compression="zip",  # 压缩形式保存
    #     encoding="utf-8",  # 编码
    #     level=LOG_LEVEL,  # 日志级别
    #     enqueue=True,  # 默认是线程安全的，enqueue=True使得多进程安全
    #     serialize=True,
    #     backtrace=True,
    #     diagnose=True,
    # )
    logger.add(
        sink=sys.stdout,  # type: ignore
        format=edit_record_and_gen_format,  # 日志显示格式
        level=level,  # 日志级别
        enqueue=True,  # 默认是线程安全的，enqueue=True使得多进程安全
        serialize=False,
        backtrace=True,
        diagnose=True,
        colorize=None,
    )

    UVICORN_LOGGING_MODULES = (
        LoggerNameEnum.root.value,  # type: ignore
        LoggerNameEnum.uvicorn_error.value,
        LoggerNameEnum.uvicorn_asgi.value,
        LoggerNameEnum.uvicorn_access.value,
        LoggerNameEnum.fastaapi.value,  # type: ignore
    )

    setup_loguru_logging_intercept(
        level=logging.getLevelName(level),  # type: ignore
        modules=UVICORN_LOGGING_MODULES,
    )

    # Tortoise
    setup_loguru_logging_intercept(
        level=logging.getLevelName(level),  # type: ignore
        modules=[
            LoggerNameEnum.tortoise.value,  # type: ignore
        ],
    )

    # # Uvicorn access log gunicorn启动时不生效
    # setup_loguru_logging_intercept(
    #     level=logging.getLevelName(logging.ERROR), modules=[LoggerNameEnum.uvicorn_access.value,]
    # )

    # disable duplicate logging
    logging.getLogger(LoggerNameEnum.root.value).handlers.clear()  # type: ignore
    logging.getLogger("uvicorn").handlers.clear()
    logging.getLogger("gunicorn").handlers.clear()
    # capture warning
    logging.captureWarnings(True)
    showwarning_ = warnings.showwarning

    def showwarning(message, *args, **kwargs):
        logger.warning(message)
        showwarning_(message, *args, **kwargs)

    warnings.showwarning = showwarning
