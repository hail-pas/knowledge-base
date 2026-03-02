from tortoise.contrib.pydantic import pydantic_model_creator
from ext.ext_tortoise.models.knowledge_base import Document
from service.collection.schema import CollectionList
from service.file_source.schema import FileSourceList


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
