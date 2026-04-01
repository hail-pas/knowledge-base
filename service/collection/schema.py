from typing import Optional

from pydantic import Field, BaseModel, field_validator, model_validator
from _pytest.python_api import raises
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
        exclude=("user_id", "tenant_id", "role_id"),
    ),
):
    workflow_template: dict = Field(default_factory=dict, description="工作流DAG配置，非外部知识库时必需")
    external_config: ExternalConfigSchema | None = Field(None, description="外部知识库配置")
    embedding_model_config_id: int | None = Field(None, description="关联嵌入模型ID")

    @field_validator("external_config", mode="before")
    @classmethod
    def validate_external_config(cls, v: dict) -> dict:
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


class CollectionList(
    pydantic_model_creator(
        Collection,
        name="CollectionList",
        exclude=(
            "user_id",
            "tenant_id",
            "role_id",
            "documents",
        ),
    ),
):
    document_count: int = 0


class CollectionDetail(CollectionList):
    embedding_model_config: EmbeddingModelConfigList | None = None
    summary: str = ""


@as_query
class CollectionFilterSchema(BaseModel):
    name__icontains: str | None = Field(None, description="名称包含")
    is_public: bool | None = Field(None, description="是否公开")
    is_external: bool | None = Field(None, description="是否外部")
    is_temp: bool | None = Field(None, description="是否临时")
