from pydantic import BaseModel

from ext.indexing.main import ModelProviderConfig
from ext.ext_httpx.main import HttpxConfig
from ext.ext_redis.main import RedisConfig
from ext.ext_celery.main import CeleryConfig
from ext.ext_tortoise.main import TortoiseConfig


class ExtensionRegistry(BaseModel):
    """
    define here
    """

    redis: RedisConfig
    # relation: TortoiseConfig
    rdb_user_center: TortoiseConfig
    rdb_knowledge_base: TortoiseConfig
    celery: CeleryConfig
    httpx: HttpxConfig = HttpxConfig()
    # register_model_provider: ModelProviderConfig = ModelProviderConfig()
