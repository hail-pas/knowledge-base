"""Root ASGI application.

The user center is an independent service (Pero). This process exposes only
Knowledge Base APIs and delegates access-token validation to Pero.
"""

from starlette.middleware.cors import CORSMiddleware

from api.knowledge_base.factory import knowledge_api
from config.main import local_configs
from core.api import ApiApplication, lifespan
from util.secret import validate_encryption_key


class RootApi(ApiApplication):
    async def before_server_start(self) -> None:
        validate_encryption_key()


description = """
==== Knowledge Base Service ====
<br><br>
Knowledge Base: <a href="/knowledge/docs/">知识库服务接口文档</a>
<br><br>
用户身份由外部 Pero 用户中心提供，本服务不提供注册、登录或用户管理接口。
"""


service_api = RootApi(
    code="KnowledgeBaseRoot",
    settings=local_configs,
    lifespan=lifespan,
    title="Knowledge Base Service",
    description=description,
    version="2.0.0",
    servers=[server.model_dump() for server in local_configs.project.swagger_servers],
)

service_api.add_middleware(
    CORSMiddleware,
    allow_origins=local_configs.server.cors.allow_origins,
    allow_credentials=local_configs.server.cors.allow_credentials,
    allow_methods=local_configs.server.cors.allow_methods,
    allow_headers=local_configs.server.cors.allow_headers,
    expose_headers=local_configs.server.cors.expose_headers,
)

service_api.mount("/knowledge", knowledge_api, "知识库服务")
