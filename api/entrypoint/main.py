import os
import sys
import signal
import logging
import argparse
import importlib
from typing import Any

from core.api import ApiApplication

sys.path.append(".")  # 将当前目录加入到环境变量中

import asyncio  # noqa

import gunicorn.app.base  # type:ignore
from loguru import logger  # type:ignore

# from common.log import setup_loguru
from config.main import local_configs  # noqa

"""FastAPI"""


def handle_sigterm(signum, frame):
    logging.error(f"Worker (pid:{os.getpid()}) was sent SIGTERM!")
    sys.exit(0)


# Function to set up loguru and standard logging
def setup_logging():
    # Setup loguru
    logger.remove()  # Remove all loguru handlers

    # Reconfigure the signal handling for the worker
    signal.signal(signal.SIGTERM, handle_sigterm)


class FastApiApplication(gunicorn.app.base.BaseApplication):
    def __init__(self, app: ApiApplication, options: dict | None = None) -> None:
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self) -> None:
        config = {key: value for key, value in self.options.items() if key in self.cfg.settings and value is not None}  # type: ignore
        for key, value in config.items():
            self.cfg.set(key.lower(), value)  # type: ignore

    def load(self) -> ApiApplication:
        return self.application


def post_fork(server: Any, worker: Any) -> None:  # ruff: noqa
    # Important: The import of skywalking should be inside the post_fork function
    # if local_configs.PROJECT.SKYWALKINGT_SERVER:
    #     print({"level": "INFO", "message": "Skywalking agent started"})
    #     import os

    #     from skywalking import agent, config

    #     # append pid-suffix to instance name
    #     # This must be done to distinguish instances if you give your instance customized names
    #     # (highly recommended to identify workers)
    #     # Notice the -child(pid) part is required to tell the difference of each worker.

    #     config.init(
    #         agent_collector_backend_services="192.168.3.46:11800",
    #         agent_name=f"python:{local_configs.PROJECT.NAME}",
    #         agent_instance_name=agent_instance_name,
    #         plugin_fastapi_collect_http_params=True,
    #         agent_protocol="grpc",
    #     )

    #     agent.start()
    # setup_logging()
    ...


# Pre-fork hook to setup logging before workers are forked
def pre_fork(server, worker):
    ...
    # setup_logging()


def import_app(app_path: str) -> ApiApplication:
    if ":" not in app_path:
        raise ValueError("Invalid app_path format. It should be in the format'module:app'.")
    module_name, app_name = app_path.split(":")
    module = importlib.import_module(module_name)
    app = getattr(module, app_name)
    if not isinstance(app, ApiApplication):
        raise ValueError(f"The app object must be an instance of FastAPI. Got: {type(app)}")
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FastAPI application with Gunicorn.")
    parser.add_argument(
        "app_path",
        help="The FastAPI app to run, in the format 'module:app'. For example: 'apis.user_center.entrypoint.factory:user_center_api'",
    )
    args = parser.parse_args()

    app = import_app(args.app_path)

    # gunicorn core.factory:app
    # --workers 4
    # --worker-class uvicorn.workers.UvicornWorker
    # --timeout 180
    # --graceful-timeout 120
    # --max-requests 4096
    # --max-requests-jitter 512
    # --log-level debug
    # --logger-class core.loguru.GunicornLogger
    # --bind 0.0.0.0:80
    # import sys
    if hasattr(app, "before_server_start"):
        asyncio.get_event_loop().run_until_complete(app.before_server_start())

    options = {
        "bind": f"{local_configs.server.address.host}:{local_configs.server.address.port}",
        "workers": local_configs.server.worker_number,
        # "worker_class": "uvicorn.workers.UvicornWorker",
        "worker_class": "uvicorn.workers.UvicornH11Worker",
        "debug": local_configs.project.debug,
        "log_level": "debug" if local_configs.project.debug else "info",
        "max_requests": 4096,  # # 最大请求数之后重启worker，防止内存泄漏
        "max_requests_jitter": 512,  # 随机重启防止所有worker一起重启：randint(0, max_requests_jitter)
        "graceful_timeout": 120,
        "timeout": 180,
        # "logger_class": "common.log.GunicornLogger",
        # "config": "entrypoint.gunicorn_conf.py",
        # "post_fork": "entrypoint.main.post_fork",
    }

    FastApiApplication(app, options).run()
