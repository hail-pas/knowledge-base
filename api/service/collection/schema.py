from pydantic import Field, BaseModel
from tortoise.contrib.pydantic import pydantic_model_creator

from enhance.epydantic import as_query, optional
from ext.ext_tortoise.models.knowledge_base import Collection


class OverridePydanticMeta:
    backward_relations: bool = False


# ============ Collection Schemas ============


class CollectionList(
    pydantic_model_creator(  # type: ignore
        Collection,
        name="CollectionList",
        meta_override=OverridePydanticMeta
    )
): ...


class CollectionDetail(
    pydantic_model_creator(  # type: ignore
        Collection,
        name="CollectionDetail",
        meta_override=OverridePydanticMeta
    )
): ...


class CollectionCreate(
    pydantic_model_creator(  # type: ignore
        Collection,
        name="CollectionCreate",
        exclude_readonly=True,
        exclude=("is_temp", )
    )
): ...


@optional()
class CollectionUpdate(
    pydantic_model_creator(  # type: ignore
        Collection,
        name="CollectionUpdate",
        exclude_readonly=True,
        include=("name", "description", )
    )
): ...


@as_query
class CollectionFilterSchema(BaseModel):
    name: str | None = Field(None, description="集合名称")  # type: ignore
    name__icontains: str | None = Field(None, description="集合名称包含")  # type: ignore
    is_public: bool | None = Field(None, description="是否公开")  # type: ignore
    is_temp: bool | None = Field(None, description="是否临时")  # type: ignore
