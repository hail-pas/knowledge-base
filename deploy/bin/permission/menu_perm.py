import sys
import asyncio
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))  # type: ignore

from loguru import logger
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.routing import Mount, Route, WebSocketRoute

from core.api import ApiApplication
from util.route import gte_all_uris
from core.context import ctx
from service.depend import api_permission_check
from ext.ext_tortoise.enums import PermissionTypeEnum, TokenSceneTypeEnum
from api.entrypoint.main import import_app
from deploy.bin.permission.data import MenuAndPerm
from ext.ext_tortoise.models.authorization import Permission, Resource


def filter_uri(route: Route | WebSocketRoute | Mount) -> bool:
    if isinstance(route, Mount):
        return True
    if isinstance(route, (Route, WebSocketRoute)):
        if not hasattr(route, "dependant"):
            return False
        return any(dependency.call is api_permission_check for dependency in route.dependant.dependencies)
    return False


class UriItem(BaseModel):
    method: str | None
    path: str
    name: str | None
    tags: list[str | None]


async def update_or_create_permission_codes(app: ApiApplication) -> None:
    latest_codes: set[str] = set()
    for url_info in gte_all_uris(app, _filter=filter_uri):
        uri_item = UriItem(**url_info)
        if not uri_item.method:
            continue
        code = f"{app.code}:{uri_item.method}:{uri_item.path}"
        await Permission.update_or_create(
            code=code,
            defaults={
                "label": uri_item.name or uri_item.path,
                "permission_type": PermissionTypeEnum.api,
                "is_deprecated": False,
            },
        )
        latest_codes.add(code)
        logger.info("Synced permission {}", code)

    for permission in await Permission.filter(
        permission_type=PermissionTypeEnum.api,
        code__startswith=f"{app.code}:",
    ):
        if permission.code not in latest_codes:
            permission.is_deprecated = True
            await permission.save(update_fields=["is_deprecated"])


async def permission_init(app: FastAPI) -> None:
    for route in app.routes:
        if isinstance(route, Mount):
            await update_or_create_permission_codes(route.app)  # type: ignore[arg-type]


async def create_single_menu_and_perm(menu_item: dict, parent_id: int | None = None) -> None:
    item = dict(menu_item)
    code = item.pop("code", None)
    if not code:
        logger.warning("Skip a menu item without code")
        return

    children = item.pop("children", [])
    permission_codes = item.pop("permissions", [])
    resource, _ = await Resource.update_or_create(
        scene=TokenSceneTypeEnum.web,
        code=code,
        parent_id=parent_id,
        defaults=item,
    )

    if permission_codes:
        permissions = await Permission.filter(code__in=permission_codes, is_deprecated=False)
        if len(permissions) != len(set(permission_codes)):
            missing = set(permission_codes) - {permission.code for permission in permissions}
            raise RuntimeError(f"请先初始化权限：{sorted(missing)}")
        await resource.permissions.clear()
        await resource.permissions.add(*permissions)

    for child in children:
        await create_single_menu_and_perm(child, parent_id=resource.id)


async def create_menu_and_perm() -> None:
    # An empty seed file must never delete manually maintained resources.
    for menu_item in MenuAndPerm:
        await create_single_menu_and_perm(menu_item)


async def main(app_path: str) -> None:
    app = import_app(app_path)
    async with ctx():
        await permission_init(app=app)
        await create_menu_and_perm()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync Knowledge Base API permissions and menu resources")
    parser.add_argument("app_path", type=str, help="ASGI application import path")
    args = parser.parse_args()
    asyncio.run(main(args.app_path))
