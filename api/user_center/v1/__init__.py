from fastapi import Request, APIRouter

from util.route import gte_all_uris
from core.response import Resp
from api.user_center.tags import TagsEnum
from api.user_center.v1.auth import router as auth_router
from api.user_center.v1.role import router as role_router
from api.user_center.v1.common import router as common_router
from api.user_center.v1.account import router as account_router
from api.user_center.v1.resource import router as resource_router

router = APIRouter(prefix="/v1")

router.include_router(auth_router, prefix="/auth", tags=[TagsEnum.authorization])
router.include_router(account_router, prefix="/account", tags=[TagsEnum.account])
router.include_router(resource_router, prefix="/resource", tags=[TagsEnum.resource])
router.include_router(common_router, prefix="/common", tags=[TagsEnum.other])
router.include_router(role_router, prefix="/role", tags=[TagsEnum.role])


@router.get("/uri-list", tags=[TagsEnum.root], summary="全部uri")
def get_all_urls_from_request(request: Request) -> Resp[list]:
    return Resp(data=gte_all_uris(request.app))  # type: ignore
