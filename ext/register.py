from pydantic import BaseModel

from ext.ext_redis.main import RedisConfig
from ext.ext_tortoise.main import TortoiseConfig
from ext.ext_celery.main import CeleryConfig


class ExtensionRegistry(BaseModel):
    """
    define here
    """

    redis: RedisConfig
    # relation: TortoiseConfig
    rdb_user_center: TortoiseConfig
    rdb_knowledge_base: TortoiseConfig
    celery: CeleryConfig
