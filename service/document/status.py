from uuid import UUID

from ext.ext_tortoise.enums import DocumentStatusEnum
from ext.ext_tortoise.models.knowledge_base import Document


_ALLOWED_TRANSITIONS = {
    DocumentStatusEnum.pending: {DocumentStatusEnum.pending, DocumentStatusEnum.processing, DocumentStatusEnum.failure},
    DocumentStatusEnum.processing: {DocumentStatusEnum.processing, DocumentStatusEnum.success, DocumentStatusEnum.failure},
    DocumentStatusEnum.success: {DocumentStatusEnum.success, DocumentStatusEnum.pending},
    DocumentStatusEnum.failure: {DocumentStatusEnum.failure, DocumentStatusEnum.pending},
}


async def update_document_status(
    document_id: int,
    workflow_uid: UUID,
    status: DocumentStatusEnum,
) -> bool:
    """Update status only while the supplied workflow still owns the document."""
    document = await Document.filter(
        id=document_id,
        current_workflow_uid=workflow_uid,
    ).only("id", "status").first()
    if document is None:
        return False

    current = DocumentStatusEnum(document.status)
    if status == current:
        return True
    if status not in _ALLOWED_TRANSITIONS[current]:
        raise ValueError(f"Invalid document status transition: {current.value} -> {status.value}")

    updated = await Document.filter(
        id=document_id,
        current_workflow_uid=workflow_uid,
        status=current.value,
    ).update(status=status.value)
    if updated:
        return True

    latest = await Document.filter(
        id=document_id,
        current_workflow_uid=workflow_uid,
    ).only("status").first()
    if latest is None:
        return False
    if DocumentStatusEnum(latest.status) == status:
        return True
    raise ValueError(
        f"Document status changed concurrently: expected {current.value}, got {latest.status}"
    )
