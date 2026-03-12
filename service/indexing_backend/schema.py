from pydantic import Field, BaseModel
from tortoise.contrib.pydantic import pydantic_model_creator

from core.types import ApiException
from enhance.epydantic import as_query
from ext.ext_tortoise.enums import IndexingBackendTypeEnum
from ext.indexing.providers.types import MilvusConfig, ElasticsearchConfig
from ext.ext_tortoise.models.knowledge_base import IndexingBackendConfig


class IndexingBackendConfigCreate(
    pydantic_model_creator(IndexingBackendConfig, name="IndexingBackendConfigCreate", exclude_readonly=True),
):
    @classmethod
    def validate_extra_config_by_type(cls, type_: IndexingBackendTypeEnum, extra_config: dict) -> None:
        if type_ == IndexingBackendTypeEnum.elasticsearch:
            ElasticsearchConfig(**extra_config)
        elif type_ == IndexingBackendTypeEnum.milvus:
            MilvusConfig(**extra_config)

    def validate_required_fields_by_type(self) -> None:
        type_ = self.type

        if type_ == IndexingBackendTypeEnum.elasticsearch:
            if not self.host or not self.port:
                raise ApiException("Elasticsearch 需要 host 和 port")

        elif type_ == IndexingBackendTypeEnum.milvus and (not self.host or not self.port):
            raise ApiException("Milvus 需要 host 和 port")

    def validate_extra_config(self) -> None:
        self.validate_extra_config_by_type(self.type, self.extra_config)


class IndexingBackendConfigList(
    pydantic_model_creator(IndexingBackendConfig, name="IndexingBackendConfigList", exclude=("password",)),
):
    pass


class IndexingBackendConfigDetail(IndexingBackendConfigList):
    pass


@as_query
class IndexingBackendConfigFilterSchema(BaseModel):
    type: IndexingBackendTypeEnum | None = Field(None, description="后端类型")
    is_enabled: bool | None = Field(None, description="是否启用")
    is_default: bool | None = Field(None, description="是否默认")
    name__icontains: str | None = Field(None, description="名称包含")
