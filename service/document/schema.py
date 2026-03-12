from pydantic import Field, BaseModel
from tortoise.contrib.pydantic import pydantic_model_creator

from enhance.epydantic import optional
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
    ),
):
    pass


class DocumentDetail(DocumentList):
    collection: CollectionList | None = None
    file_source: FileSourceList | None = None


@optional()
class DocumentUpdate(
    pydantic_model_creator(
        Document,
        name="DocumentList",
        include=("display_name", "config_flag", "workflow_template"),
    ),
): ...


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


@optional()
class DocumentChunkUpdate(BaseModel):
    content: str
    pages: list[int]
    start: dict
    end: dict
    overlap_start: dict | None
    overlap_end: dict | None
    metadata: dict
    manual_add: bool


class DocumentGeneratedFaqCreate(BaseModel):
    content: str | None = Field(None, description="相关文档内容块")
    question: str = Field(..., description="问题")
    answer: str = Field(..., description="答案")
    manual_add: bool = Field(default=True, description="是否手动添加")
    enabled: bool = Field(default=True, description="是否启用")


@optional()
class DocumentGeneratedFaqUpdate(BaseModel):
    content: str | None
    question: str
    answer: str
    manual_add: bool
    enabled: bool
