from pydantic import Field, BaseModel
from tortoise.contrib.pydantic import pydantic_model_creator

from enhance.epydantic import as_query, optional
from ext.ext_tortoise.enums import EmbeddingModelTypeEnum
from ext.ext_tortoise.models.knowledge_base import EmbeddingModelConfig


class OverridePydanticMeta:
    backward_relations: bool = False


class EmbeddingModelConfigList(
    pydantic_model_creator(  # type: ignore
        EmbeddingModelConfig,
        name="EmbeddingModelConfigList",
        meta_override=OverridePydanticMeta
    )
): ...


class EmbeddingModelConfigDetail(
    pydantic_model_creator(  # type: ignore
        EmbeddingModelConfig,
        name="EmbeddingModelConfigDetail",
        meta_override=OverridePydanticMeta
    )
): ...


class EmbeddingModelConfigCreate(
    pydantic_model_creator(  # type: ignore
        EmbeddingModelConfig,
        name="EmbeddingModelConfigCreate",
        exclude_readonly=True
    )
): ...


@optional()
class EmbeddingModelConfigUpdate(
    pydantic_model_creator(  # type: ignore
        EmbeddingModelConfig,
        name="EmbeddingModelConfigUpdate",
        exclude_readonly=True
    )
): ...


@as_query
class EmbeddingModelConfigFilterSchema(BaseModel):
    name: str | None = Field(None, description="配置名称")  # type: ignore
    name__icontains: str | None = Field(None, description="配置名称包含")  # type: ignore
    type: EmbeddingModelTypeEnum = Field(None, description="模型类型")  # type: ignore
    is_enabled: bool | None = Field(None, description="是否启用")  # type: ignore
    is_default: bool | None = Field(None, description="是否默认")  # type: ignore
