from pydantic import Field, BaseModel
from tortoise.contrib.pydantic import pydantic_model_creator

from service.collection.schema import CollectionList
from service.file_source.schema import FileSourceList
from ext.ext_tortoise.models.knowledge_base import (
    Document,
    DocumentChunk,
    DocumentPages,
    DocumentGeneratedFaq,
)


class DocumentList(
    pydantic_model_creator(
        Document,
        name="DocumentList",
        exclude=("uri", "parsed_uri", "source_meta", "workflow_template"),
    ),
):
    pass


class DocumentDetail(DocumentList):
    collection: CollectionList | None = None
    file_source: FileSourceList | None = None


class DocumentUpdate(BaseModel):
    display_name: str | None = Field(default=None, max_length=255)
    config_flag: int | None = Field(default=None, ge=0, le=32767)
    workflow_template: dict | None = None


class DocumentPageList(
    pydantic_model_creator(
        DocumentPages,
        name="DocumentPageList",
    ),
):
    pass


class DocumentChunkList(
    pydantic_model_creator(
        DocumentChunk,
        name="DocumentChunkList",
    ),
):
    pass


class DocumentGeneratedFaqList(
    pydantic_model_creator(
        DocumentGeneratedFaq,
        name="DocumentGeneratedFaqList",
    ),
):
    pass


class DocumentChunkCreate(BaseModel):
    content: str = Field(..., description="切块内容")
    pages: list[int] = Field(..., min_length=1, description="切块页码列表")
    start: dict = Field(..., description="起始位置（页码+页内偏移）")
    end: dict = Field(..., description="结束位置（页码+页内偏移）")
    overlap_start: dict | None = Field(None, description="重叠起始位置（页码+页内偏移）")
    overlap_end: dict | None = Field(None, description="重叠结束位置（页码+页内偏移）")
    metadata: dict = Field(default_factory=dict, description="元数据")
    manual_add: bool = Field(default=True, description="是否手动添加")


class DocumentChunkUpdate(BaseModel):
    content: str | None = None
    pages: list[int] | None = None
    start: dict | None = None
    end: dict | None = None
    overlap_start: dict | None = None
    overlap_end: dict | None = None
    metadata: dict | None = None
    manual_add: bool | None = None


class DocumentGeneratedFaqCreate(BaseModel):
    content: str | None = Field(None, description="相关文档内容块")
    question: str = Field(..., description="问题")
    answer: str = Field(..., description="答案")
    manual_add: bool = Field(default=True, description="是否手动添加")
    enabled: bool = Field(default=True, description="是否启用")


class DocumentGeneratedFaqUpdate(BaseModel):
    content: str | None = None
    question: str | None = None
    answer: str | None = None
    manual_add: bool | None = None
    enabled: bool | None = None
