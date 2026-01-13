from typing import Literal

from core.api import ApiApplication, lifespan
from config.main import local_configs
from api.knowledge_base.v1 import router as v1_router
from api.knowledge_base.v2 import router as v2_router
from core.response import Resp
from core.exception import handler_roster as exception_handler_roster
from core.middleware import roster as middleware_roster

knowledge_api = ApiApplication(
    code="KnowledgeBase",
    settings=local_configs,
    title="KnowledgeBase",
    description="KnowledgeBase",
    lifespan=lifespan,
    version="1.0.0",
    redirection_url="/docs",
    swagger_ui_parameters={
        "url": "openapi.json",
        "persistAuthorization": local_configs.project.debug,
    },
    servers=[
        {
            "url": str(server.url) + "example",
            "description": server.description,
        }
        for server in local_configs.project.swagger_servers
    ],
)

knowledge_api.setup_middleware(roster=middleware_roster)
knowledge_api.setup_exception_handlers(roster=exception_handler_roster)

knowledge_api.amount_app_or_router(roster=[(v1_router, "", "v1")])
knowledge_api.amount_app_or_router(roster=[(v2_router, "", "v2")])


@knowledge_api.get("/health", summary="健康检查")
async def health() -> Resp[dict[Literal["status"], Literal["ok"]]]:
    """
    健康检查
    """
    return Resp(data={"status": "ok"})
