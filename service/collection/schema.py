from urllib.parse import urlsplit

from pydantic import Field, BaseModel, field_validator, model_validator
from tortoise.contrib.pydantic import pydantic_model_creator

from core.types import ApiException
from service.collection.helper import WorkflowTemplateValidator
from service.embedding_model.schema import EmbeddingModelConfigList
from ext.ext_tortoise.models.knowledge_base import Collection


class ExternalConfigSchema(BaseModel):
    """External KB config - placeholder for future fields"""

    endpoint: str
    authorization: str = Field(min_length=1)
    collection_id: str = Field(min_length=1)

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ApiException("外部知识库 endpoint 不允许嵌入凭证、query 或 fragment")
        return value


class CollectionCreate(
    pydantic_model_creator(
        Collection,
        name="CollectionCreateBase",
        exclude_readonly=True,
        exclude=("user_id", "indexing_backend_config"),
    ),
):
    workflow_template: dict = Field(default_factory=dict, description="工作流DAG配置，非外部知识库时必需")
    external_config: ExternalConfigSchema | None = Field(None, description="外部知识库配置")
    embedding_model_config_id: int | None = Field(None, description="关联嵌入模型ID")
    indexing_backend_config_id: int | None = Field(None, description="关联索引后端配置ID；为空时使用系统默认后端")

    @field_validator("external_config", mode="before")
    @classmethod
    def validate_external_config(cls, v: dict | None) -> dict | None:
        if v is None:
            return None
        try:
            ExternalConfigSchema(**v)
        except Exception:
            raise ApiException("external_config 格式错误") from None
        return v

    @model_validator(mode="after")
    def validate_instance(self) -> "CollectionCreate":
        """Validate constraints for external collections"""
        if self.is_external:  # type: ignore
            if self.embedding_model_config_id is not None:
                raise ApiException("外部知识库不能设置 embedding_model_config")
            if self.indexing_backend_config_id is not None:
                raise ApiException("外部知识库不能设置 indexing_backend_config")
            if self.workflow_template:  # type: ignore
                self.workflow_template = {}  # type: ignore
            if not self.external_config or self.external_config == {}:  # type: ignore
                raise ApiException("外部知识库必须提供 external_config")
        else:
            if self.external_config is not None:
                raise ApiException("普通知识库不能设置 external_config")
            try:
                WorkflowTemplateValidator.validate(self.workflow_template)
            except ValueError as e:
                raise ApiException(str(e)) from e
        return self


class CollectionUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    is_public: bool | None = None
    is_temp: bool | None = None
    workflow_template: dict | None = None
    external_config: dict | None = None
    embedding_model_config_id: int | None = None
    indexing_backend_config_id: int | None = None

    @field_validator("workflow_template", mode="before")
    @classmethod
    def validate_workflow_template(cls, v: dict | None) -> dict | None:
        if v is None:
            raise ApiException("workflow_template 不能为空")
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
                return ExternalConfigSchema(**v).model_dump()
            except Exception:
                raise ApiException("external_config 格式错误") from None
        return v


class ExternalConfigPublicSchema(BaseModel):
    """Public collection metadata; never serialize external authorization secrets."""

    endpoint: str
    collection_id: str

    @field_validator("endpoint", mode="before")
    @classmethod
    def sanitize_endpoint(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.netloc:
            return parsed._replace(
                netloc=parsed.netloc.rsplit("@", 1)[-1],
                query="",
                fragment="",
            ).geturl()
        return value.split("?", 1)[0].split("#", 1)[0]


class CollectionList(
    pydantic_model_creator(
        Collection,
        name="CollectionList",
        exclude=(
            "user_id",
            "documents",
            "external_config",
            "workflow_template",
            "indexing_backend_config",
        ),
    ),
):
    external_config: ExternalConfigPublicSchema | None = None
    indexing_backend_config_id: int | None = None
    document_count: int = 0

    @field_validator("external_config", mode="before")
    @classmethod
    def sanitize_external_config(cls, value: dict | None) -> dict | None:
        if not value:
            return None
        endpoint = value.get("endpoint")
        if endpoint:
            parsed_endpoint = urlsplit(endpoint)
            endpoint = parsed_endpoint._replace(
                netloc=parsed_endpoint.netloc.rsplit("@", 1)[-1],
                query="",
                fragment="",
            ).geturl()

        return {
            "endpoint": endpoint,
            "collection_id": value.get("collection_id"),
        }


class CollectionDetail(CollectionList):
    workflow_template: dict = Field(default_factory=dict)
    embedding_model_config: EmbeddingModelConfigList | None = None
    summary: str = ""


class CollectionFilterSchema(BaseModel):
    name__icontains: str | None = Field(None, description="名称包含")
    is_public: bool | None = Field(None, description="是否公开")
    is_external: bool | None = Field(None, description="是否外部")
    is_temp: bool | None = Field(None, description="是否临时")
