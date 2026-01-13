import sys
import asyncio
from uuid import UUID

from loguru import logger

from core.api import ApiApplication
from util.route import gte_all_uris
from core.context import ctx
from ext.ext_tortoise import enums
from api.entrypoint.main import import_app
from deploy.bin.permission.data import MenuAndPerm
from ext.ext_tortoise.models.user_center import Account, Resource, Permission

sys.path.append(".")  # noqa

from fastapi import FastAPI
from pydantic import BaseModel
from starlette.routing import Mount, Route, WebSocketRoute

from api.depend import token_required, api_permission_check


def filter_uri(r: Route | WebSocketRoute | Mount) -> bool:
    if isinstance(r, Mount):
        return True

    if isinstance(r, Route | WebSocketRoute):
        if not hasattr(r, "dependant"):
            return False
        dependencies = r.dependant.dependencies  # type: ignore
        return any(i.call in [token_required, api_permission_check] for i in dependencies)
    return None


class UriItem(BaseModel):
    method: str | None
    path: str
    name: str | None
    tags: list[str | None]


async def update_or_create_perm_code(app: ApiApplication) -> None:
    latest_codes = set()
    url_list = gte_all_uris(app, _filter=filter_uri)
    unique_code = app.code
    for url in url_list:
        url = UriItem(**url)
        if not url.method:
            continue
        code = f"{unique_code}:{url.method}:{url.path}"
        logger.info(f"{code}, {url.name}")
        await Permission.update_or_create(
            code=code,
            defaults={
                "label": url.name,
                "permission_type": enums.PermissionTypeEnum.api,  # type: ignore
                "is_deprecated": False,
            },
        )
        latest_codes.add(code)

    # 清理过期的权限
    for permission in await Permission.filter(
        permission_type=enums.PermissionTypeEnum.api,  # type: ignore
        code__startswith=unique_code,
    ):
        if permission.code not in latest_codes:
            permission.is_deprecated = True
            await permission.save(update_fields=["is_deprecated"])


async def permission_init(app: FastAPI):
    for route in app.routes:
        if isinstance(route, Mount):
            await update_or_create_perm_code(route.app)  # type: ignore


async def create_single_menu_and_perm(
    *,
    menu_item: dict,
    parent_id: int | UUID | None = None,
    parent_id_dict: dict = {},
):
    code = menu_item.pop("code", None)
    if not code:
        logger.warning(f"Menu {code} has no code, skip it.")
        return

    children = menu_item.pop("children", [])
    permissions = menu_item.pop("permissions", [])
    logger.info(f"Creating menu {code}")
    r, _ = await Resource.update_or_create(
        scene=enums.TokenSceneTypeEnum.web,
        code=code,
        parent_id=parent_id,
        defaults=menu_item,
    )
    parent_id_dict[code] = r.id
    perm_count = len(permissions)
    if perm_count > 0:
        for perm in permissions:
            if not await Permission.filter(code=perm).exists():
                logger.error(f"请先初始化权限：{perm}")
                raise Exception(f"请先初始化权限：{perm}")
        perms = await Permission.filter(code__in=permissions)
        logger.info(f"Created menu {code} with {perm_count} permissions.")
        await r.permissions.add(*perms)

    logger.info("-" * 40)
    if len(children) > 0:
        for child in children:
            await create_single_menu_and_perm(
                menu_item=child,
                parent_id=r.id,
                parent_id_dict=parent_id_dict,
            )


async def create_menu_and_perm():
    parent_id_dict = {}

    for menu_item in MenuAndPerm:
        await create_single_menu_and_perm(
            menu_item=menu_item,
            parent_id=None,
            parent_id_dict=parent_id_dict,
        )
    logger.info("Cleaning up extra resources")
    count = await Resource.exclude(id__in=parent_id_dict.values()).delete()
    logger.info(f"Deleted {count} extra resources")


async def refresh_permissions():
    accounts = await Account.filter(deleted_at=0)
    for acc in accounts:
        acc: Account
        logger.info(f"Refreshing permissions for {acc.username}")
        await acc.update_cache_permissions()


async def main(app_path: str):
    app = import_app(app_path)
    async with ctx():
        await permission_init(app=app)
        await create_menu_and_perm()
        await refresh_permissions()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <app_path>")
        sys.exit(1)
    app_path = sys.argv[1]
    asyncio.run(main(app_path))
