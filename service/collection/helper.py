from ext.indexing.models import CollectionIndexModelHelper
from ext.ext_tortoise.enums import DocumentStatusEnum
from ext.ext_tortoise.models.knowledge_base import (
    Document,
    Collection,
)


class CollectionService:
    def __init__(self, collection: Collection) -> None:
        self.collection = collection
        self.collection_index_helper = CollectionIndexModelHelper(collection)

    async def can_delete_collection(self) -> bool:
        """
        Check if collection can be deleted.
        Returns True if all documents are in final status (success/failure).
        """
        documents = await Document.filter(collection_id=self.collection.id, deleted_at=0).values_list(
            "status",
            flat=True,
        )

        final_statuses = {
            DocumentStatusEnum.success.value,
            DocumentStatusEnum.failure.value,
        }

        return all(status in final_statuses for status in documents)
