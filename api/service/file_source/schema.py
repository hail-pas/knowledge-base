from pydantic import Field, BaseModel
from tortoise.contrib.pydantic import pydantic_model_creator

from enhance.epydantic import as_query, optional
from ext.ext_tortoise.enums import (
    FileSourceTypeEnum,
)
from ext.ext_tortoise.models.knowledge_base import FileSource


class OverridePydanticMeta:
    backward_relations: bool = False

# ============ FileSource Schemas ============


class FileSourceList(
    pydantic_model_creator(  # type: ignore
        FileSource,
        name="FileSourceList",
        meta_override=OverridePydanticMeta
    )
): ...


class FileSourceDetail(
    pydantic_model_creator(  # type: ignore
        FileSource,
        name="FileSourceDetail",
        meta_override=OverridePydanticMeta
    )
): ...


class FileSourceCreate(
    pydantic_model_creator(  # type: ignore
        FileSource,
        name="FileSourceCreate",
        exclude_readonly=True
    )
): ...


@optional()
class FileSourceUpdate(
    pydantic_model_creator(  # type: ignore
        FileSource,
        name="FileSourceUpdate",
        exclude_readonly=True
    )
): ...


@as_query
class FileSourceFilterSchema(BaseModel):
    name: str | None = Field(None, description="文件源名称")  # type: ignore
    name__icontains: str | None = Field(None, description="文件源名称包含")  # type: ignore
    type: FileSourceTypeEnum = Field(None, description="文件源类型")  # type: ignore
    is_enabled: bool | None = Field(None, description="是否启用")  # type: ignore
    is_default: bool | None = Field(None, description="是否默认")  # type: ignore
