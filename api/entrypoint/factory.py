from core.api import ApiApplication, lifespan
from config.main import local_configs
from api.knowledge_base.factory import knowledge_api
from api.user_center.factory import user_center_api
from starlette.middleware.cors import CORSMiddleware


class RootApi(ApiApplication):
    async def before_server_start(self) -> None:
        return


description = """
==== 欢迎来到主服务 ====
<br><br>
User-Center: <a href="/user/docs/">用户中心服务接口文档</a>
<br><br>
Knowledge-Base: <a href="/knowledge/docs/">知识管理接口文档</a>
<br><br>
"""

service_api = RootApi(
    code="ServiceRoot",
    settings=local_configs,
    lifespan=lifespan,
    title="主服务",
    description=description,
    version="1.0.0",
    servers=[s.model_dump() for s in local_configs.project.swagger_servers],
)

service_api.add_middleware(
    CORSMiddleware,
    allow_origins=local_configs.server.cors.allow_origins,  # 或指定你的前端地址
    allow_credentials=local_configs.server.cors.allow_credentials,
    allow_methods=local_configs.server.cors.allow_methods,
    allow_headers=local_configs.server.cors.allow_headers,
    expose_headers=local_configs.server.cors.expose_headers,
)

service_api.mount(
    "/user",
    user_center_api,
    "用户中心",
)
service_api.mount(
    "/knowledge",
    knowledge_api,
    "RAG知识管理中心",
)
