from fastapi import Request, APIRouter

from util.route import gte_all_uris
from core.response import Resp
from api.knowledge_base.tags import TagsEnum
from api.knowledge_base.v1.config import router as config_router
from api.knowledge_base.v1.file_source import router as file_source_router
from api.knowledge_base.v1.collection import router as collection_router
from api.knowledge_base.v1.document import router as document_router

router = APIRouter(prefix="/v1")

router.include_router(config_router, prefix="/config", tags=[TagsEnum.config])
router.include_router(file_source_router, prefix="/file-source", tags=[TagsEnum.document])
router.include_router(collection_router, prefix="/collection", tags=[TagsEnum.document])
router.include_router(document_router, prefix="/document", tags=[TagsEnum.document])


@router.get("/uri-list", tags=[TagsEnum.root], summary="全部uri")
def get_all_urls_from_request(request: Request) -> Resp[list]:
    return Resp(data=gte_all_uris(request.app))  # type: ignore
