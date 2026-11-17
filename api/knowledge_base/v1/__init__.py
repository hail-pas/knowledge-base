from fastapi import Request, APIRouter

from util.route import gte_all_uris
from core.response import Resp
from api.knowledge_base.tags import TagsEnum
from api.knowledge_base.v1.config import router as config_router
from api.knowledge_base.v1.chat import router as chat_router
from api.knowledge_base.v1.document import router as document_router
from api.knowledge_base.v1.collection import router as collection_router
from api.knowledge_base.v1.role import router as role_router
from api.knowledge_base.v1.resource import router as resource_router, my_router as my_resource_router
from api.knowledge_base.v1.permission import router as permission_router
from api.knowledge_base.v1.user_role import router as user_role_router

router = APIRouter(prefix="/v1")

router.include_router(config_router, prefix="/config", tags=[TagsEnum.config])
router.include_router(collection_router, prefix="/collection", tags=[TagsEnum.document])
router.include_router(document_router, prefix="/document", tags=[TagsEnum.document])
router.include_router(chat_router, prefix="/chat", tags=[TagsEnum.chat])
router.include_router(role_router, prefix="/role", tags=[TagsEnum.role])
router.include_router(resource_router, prefix="/resource", tags=[TagsEnum.resource])
router.include_router(my_resource_router, prefix="/resource", tags=[TagsEnum.resource])
router.include_router(permission_router, prefix="/permission", tags=[TagsEnum.permission])
router.include_router(user_role_router, prefix="/user-role", tags=[TagsEnum.user_role])


@router.get("/uri-list", tags=[TagsEnum.root], summary="全部uri")
def get_all_urls_from_request(request: Request) -> Resp[list]:
    return Resp(data=gte_all_uris(request.app))  # type: ignore
