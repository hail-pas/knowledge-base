from typing import Any
from collections.abc import Callable

from fastapi import FastAPI
from starlette.routing import Mount, Route, WebSocketRoute


def gte_all_uris(
    app: FastAPI,
    _filter: Callable[[Route | WebSocketRoute | Mount], bool] | None = None,
) -> list[dict[str, Any]]:
    """获取app下所有的URI

    Args:
        app (FastAPI): FastAPI App

    Returns:
        list[str]: URI 列表
    """
    uri_list = []
    paths = []

    def get_uri_list(_app: FastAPI | Mount, prefix: str = ""):
        for route in _app.routes:
            route_info = {
                "path": f"{prefix}{route.path}",  # type: ignore
                "name": getattr(route, "summary", None) or route.name,  # type: ignore
                "tags": getattr(route, "tags", []),
                "operation_id": getattr(route, "operation_id", None),  # type: ignore
            }
            if _filter and not _filter(route):  # type: ignore
                continue
            if isinstance(route, Route):
                if not route.methods:
                    continue
                for method in route.methods:
                    full_path = f"{method}:{route_info['path']}"
                    if method in ["HEAD", "OPTIONS"] or full_path in paths:
                        continue
                    uri_list.append(
                        {
                            "method": method,
                            **route_info,
                        },
                    )
                    paths.append(full_path)
            elif isinstance(route, WebSocketRoute):
                if f"{method}:{route_info['path']}" in paths:
                    continue
                uri_list.append(
                    {
                        "method": "ws",
                        **route_info,
                    },
                )
                paths.append(full_path)
            elif isinstance(route, Mount):
                get_uri_list(route, prefix=f"{prefix}{route.path}")

    get_uri_list(app)
    return uri_list
