from pydantic import Field, BaseModel
from tortoise.contrib.pydantic import pydantic_model_creator

from api.service.file_source.schema import FileSourceSimpleList
from enhance.epydantic import as_query, optional
from ext.ext_tortoise.enums import DocumentStatusEnum
from ext.ext_tortoise.models.knowledge_base import Document


class OverridePydanticMeta:
    backward_relations: bool = False


class DocumentList(
    pydantic_model_creator(  # type: ignore
        Document,
        name="DocumentList",
        meta_override=OverridePydanticMeta
    )
):
    collection_id: int
    file_source: FileSourceSimpleList
    file_source_id: int


class DocumentDetail(
    pydantic_model_creator(  # type: ignore
        Document,
        name="DocumentDetail",
        meta_override=OverridePydanticMeta
    )
): ...


# Document supports two creation modes:
# 1. Upload file: user uploads file directly (file_source_id is optional, use default if not provided)
# 2. URI mode: user provides file_source_id and uri (must validate file exists in source)


class DocumentCreateByUpload(BaseModel):
    """通过上传文件创建文档"""
    collection_id: int = Field(..., description="关联集合ID")
    display_name: str | None = Field(None, description="显示名称，默认使用文件名")
    file_source_id: int | None = Field(None, description="关联文件源ID（可选，不传则使用默认）")


class DocumentCreateByUri(BaseModel):
    """通过URI创建文档（需要验证文件在file_source中存在）"""
    collection_id: int = Field(..., description="关联集合ID")
    file_source_id: int = Field(..., description="关联文件源ID")
    uri: str = Field(..., description="文件唯一标识")
    file_name: str = Field(..., description="文件名")
    display_name: str | None = Field(None, description="显示名称，默认使用file_name")
    source_last_modified: str | None = Field(None, description="文件源最后修改时间")
    source_version_key: str | None = Field(None, description="文件源版本标识")
    source_meta: dict | None = Field(None, description="文件源元数据")


class DocumentUpdate(BaseModel):
    file_name: str = Field(..., description="文件名")
    display_name: str | None = Field(None, description="显示名称，默认使用file_name")
    # short_summary: str | None = Field(None, description="文件摘要")
    # long_summary: str | None = Field(None, description="文件详细摘要")


@as_query
class DocumentFilterSchema(BaseModel):
    collection_id: int | None = Field(None, description="关联集合ID")  # type: ignore
    file_source_id: int | None = Field(None, description="关联文件源ID")  # type: ignore
    file_name: str | None = Field(None, description="文件名")  # type: ignore
    file_name__icontains: str | None = Field(None, description="文件名包含")  # type: ignore
    display_name: str | None = Field(None, description="显示名称")  # type: ignore
    display_name__icontains: str | None = Field(None, description="显示名称包含")  # type: ignore
    extension: str | None = Field(None, description="文件扩展名")  # type: ignore
    status: DocumentStatusEnum = Field(None, description="文件状态")  # type: ignore
    is_deleted_in_source: bool | None = Field(None, description="文件是否在源中被删除")  # type: ignore
