from pydantic import BaseModel

from ext.file_source.factory import FileSourceConfig
from ext.ext_redis.main import RedisConfig
from ext.ext_tortoise.main import TortoiseConfig


class ExtensionRegistry(BaseModel):
    """
    define here
    """

    redis: RedisConfig
    file_source: FileSourceConfig
    # relation: TortoiseConfig
    # rdb_user_center: TortoiseConfig
    rdb_knowledge_base: TortoiseConfig
