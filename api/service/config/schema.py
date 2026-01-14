from pydantic import Field, BaseModel
from tortoise.contrib.pydantic import pydantic_model_creator

from enhance.epydantic import as_query, optional
from ext.ext_tortoise.enums import (
    EmbeddingModelTypeEnum,
    IndexingBackendTypeEnum,
    LLMModelTypeEnum,
)
from ext.ext_tortoise.models.knowledge_base import (
    EmbeddingModelConfig,
    IndexingBackendConfig,
    LLMModelConfig,
)


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


# IndexingBackendConfig Schemas


class IndexingBackendConfigList(
    pydantic_model_creator(  # type: ignore
        IndexingBackendConfig,
        name="IndexingBackendConfigList",
        meta_override=OverridePydanticMeta
    )
): ...


class IndexingBackendConfigDetail(
    pydantic_model_creator(  # type: ignore
        IndexingBackendConfig,
        name="IndexingBackendConfigDetail",
        meta_override=OverridePydanticMeta
    )
): ...


class IndexingBackendConfigCreate(
    pydantic_model_creator(  # type: ignore
        IndexingBackendConfig,
        name="IndexingBackendConfigCreate",
        exclude_readonly=True
    )
): ...


@optional()
class IndexingBackendConfigUpdate(
    pydantic_model_creator(  # type: ignore
        IndexingBackendConfig,
        name="IndexingBackendConfigUpdate",
        exclude_readonly=True
    )
): ...


@as_query
class IndexingBackendConfigFilterSchema(BaseModel):
    name: str | None = Field(None, description="配置名称")  # type: ignore
    name__icontains: str | None = Field(None, description="配置名称包含")  # type: ignore
    type: IndexingBackendTypeEnum = Field(None, description="后端类型")  # type: ignore
    is_enabled: bool | None = Field(None, description="是否启用")  # type: ignore
    is_default: bool | None = Field(None, description="是否默认")  # type: ignore


# LLMModelConfig Schemas


class LLMModelConfigList(
    pydantic_model_creator(  # type: ignore
        LLMModelConfig,
        name="LLMModelConfigList",
        meta_override=OverridePydanticMeta
    )
): ...


class LLMModelConfigDetail(
    pydantic_model_creator(  # type: ignore
        LLMModelConfig,
        name="LLMModelConfigDetail",
        meta_override=OverridePydanticMeta
    )
): ...


class LLMModelConfigCreate(
    pydantic_model_creator(  # type: ignore
        LLMModelConfig,
        name="LLMModelConfigCreate",
        exclude_readonly=True
    )
): ...


@optional()
class LLMModelConfigUpdate(
    pydantic_model_creator(  # type: ignore
        LLMModelConfig,
        name="LLMModelConfigUpdate",
        exclude_readonly=True
    )
): ...


@as_query
class LLMModelConfigFilterSchema(BaseModel):
    name: str | None = Field(None, description="配置名称")  # type: ignore
    name__icontains: str | None = Field(None, description="配置名称包含")  # type: ignore
    type: LLMModelTypeEnum = Field(None, description="模型类型")  # type: ignore
    model_name: str | None = Field(None, description="模型标识符")  # type: ignore
    model_name__icontains: str | None = Field(None, description="模型标识符包含")  # type: ignore
    is_enabled: bool | None = Field(None, description="是否启用")  # type: ignore
    is_default: bool | None = Field(None, description="是否默认")  # type: ignore
