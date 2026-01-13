from core.api import ApiApplication, lifespan
from config.main import local_configs
from api.knowledge_base.factory import knowledge_api
from api.user_center.factory import user_center_api


class RootApi(ApiApplication):
    async def before_server_start(self) -> None:
        return


description = """
==== 欢迎来到主服务 ====
<br><br>
User-Center: <a href="/user/docs/">用户中心服务接口文档</a>
<br><br>
Knowledge-Base: <a href="/second/docs/">第二服務接口文档</a>
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
service_api.mount(
    "/user",
    user_center_api,
    "用户中心",
)
service_api.mount(
    "/second",
    knowledge_api,
    "第二服务中心",
)
