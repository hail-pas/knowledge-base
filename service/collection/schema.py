from pydantic import Field, BaseModel, field_validator, model_validator
from tortoise.contrib.pydantic import pydantic_model_creator

from core.types import ApiException
from enhance.epydantic import as_query, optional
from service.collection.helper import WorkflowTemplateValidator
from service.embedding_model.schema import EmbeddingModelConfigList
from ext.ext_tortoise.models.knowledge_base import Collection


class ExternalConfigSchema(BaseModel):
    """External KB config - placeholder for future fields"""

    endpoint: str
    authorization: str
    collection_id: str


class CollectionCreate(
    pydantic_model_creator(
        Collection,
        name="CollectionCreateBase",
        exclude_readonly=True,
        exclude=("user_id",),
    ),
):
    workflow_template: dict = Field(default_factory=dict, description="工作流DAG配置，非外部知识库时必需")
    external_config: ExternalConfigSchema | None = Field(None, description="外部知识库配置")
    embedding_model_config_id: int | None = Field(None, description="关联嵌入模型ID")

    @field_validator("external_config", mode="before")
    @classmethod
    def validate_external_config(cls, v: dict | None) -> dict | None:
        if v is None:
            return None
        try:
            ExternalConfigSchema(**v)
        except Exception as e:
            raise ApiException(f"external_config 格式错误: {str(e)}") from e
        return v

    @model_validator(mode="after")
    def validate_instance(self) -> "CollectionCreate":
        """Validate constraints for external collections"""
        if self.is_external:  # type: ignore
            if self.embedding_model_config_id is not None:
                raise ApiException("外部知识库不能设置 embedding_model_config")
            if self.workflow_template:  # type: ignore
                self.workflow_template = {}  # type: ignore
            if not self.external_config or self.external_config == {}:  # type: ignore
                raise ApiException("外部知识库必须提供 external_config")
        else:
            try:
                WorkflowTemplateValidator.validate(self.workflow_template)
            except ValueError as e:
                raise ApiException(str(e)) from e
        return self


@optional()
class CollectionUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    is_public: bool | None = None
    is_temp: bool | None = None
    is_external: bool | None = None
    workflow_template: dict | None = None
    external_config: dict | None = None
    embedding_model_config_id: int | None = None

    @field_validator("workflow_template", mode="before")
    @classmethod
    def validate_workflow_template(cls, v: dict | None) -> dict | None:
        if v is not None:
            try:
                WorkflowTemplateValidator.validate(v)
            except ValueError as e:
                raise ApiException(str(e)) from e
        return v

    @field_validator("external_config")
    @classmethod
    def validate_external_config(cls, v: dict | None) -> dict | None:
        if v is not None:
            try:
                ExternalConfigSchema(**v)
            except Exception as e:
                raise ApiException(f"external_config 格式错误: {str(e)}") from e
        return v


class ExternalConfigPublicSchema(BaseModel):
    """Public collection metadata; never serialize external authorization secrets."""

    endpoint: str
    collection_id: str


class CollectionList(
    pydantic_model_creator(
        Collection,
        name="CollectionList",
        exclude=(
            "user_id",
            "documents",
            "external_config",
            "workflow_template",
        ),
    ),
):
    external_config: ExternalConfigPublicSchema | None = None
    document_count: int = 0

    @field_validator("external_config", mode="before")
    @classmethod
    def sanitize_external_config(cls, value: dict | None) -> dict | None:
        if not value:
            return None
        return {
            key: value[key]
            for key in ("endpoint", "collection_id")
            if key in value
        }


class CollectionDetail(CollectionList):
    workflow_template: dict = Field(default_factory=dict)
    embedding_model_config: EmbeddingModelConfigList | None = None
    summary: str = ""


@as_query
class CollectionFilterSchema(BaseModel):
    name__icontains: str | None = Field(None, description="名称包含")
    is_public: bool | None = Field(None, description="是否公开")
    is_external: bool | None = Field(None, description="是否外部")
    is_temp: bool | None = Field(None, description="是否临时")
