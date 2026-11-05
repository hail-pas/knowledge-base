from ext.ext_tortoise.enums import DocumentStatusEnum
from ext.ext_tortoise.models.knowledge_base import Document

_ALLOWED_TRANSITIONS = {
    DocumentStatusEnum.pending: {DocumentStatusEnum.pending, DocumentStatusEnum.processing, DocumentStatusEnum.failure},
    DocumentStatusEnum.processing: {DocumentStatusEnum.processing, DocumentStatusEnum.success, DocumentStatusEnum.failure},
    DocumentStatusEnum.success: {DocumentStatusEnum.success, DocumentStatusEnum.pending},
    DocumentStatusEnum.failure: {DocumentStatusEnum.failure, DocumentStatusEnum.pending},
}


async def update_document_status(document_id: int, status: DocumentStatusEnum) -> None:
    document = await Document.filter(id=document_id).only("id", "status").first()
    if document is None:
        raise ValueError(f"Document not found: {document_id}")

    current = DocumentStatusEnum(document.status)
    if status == current:
        return
    if status not in _ALLOWED_TRANSITIONS[current]:
        raise ValueError(f"Invalid document status transition: {current.value} -> {status.value}")

    updated = await Document.filter(id=document_id, status=current.value).update(status=status.value)
    if not updated:
        latest = await Document.filter(id=document_id).only("status").first()
        if latest:
            raise ValueError(f"Document status changed concurrently: expected {current.value}, got {latest.status}")
