from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import Field

from service.chat.domain.common import StrictModel
from service.chat.domain.errors import ChatContextError


class ChatRequestContext(StrictModel):
    account: Any | None = None
    session_id: UUID | None = None
    conversation_id: int | None = Field(default=None, ge=1)

    @property
    def account_id(self) -> int | None:
        account_id = getattr(self.account, "id", None)
        return int(account_id) if account_id is not None else None

    @property
    def is_staff(self) -> bool:
        return bool(getattr(self.account, "is_staff", False)) if self.account is not None else False

    def require_session_id(self) -> UUID:
        if self.session_id is None:
            raise ChatContextError("chat request context missing session_id", field="session_id")
        return self.session_id

    def with_conversation(self, conversation_id: int | None) -> ChatRequestContext:
        return self.model_copy(update={"conversation_id": conversation_id})


__all__ = [
    "ChatRequestContext",
]
