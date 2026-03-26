"""Chat domain exception hierarchy."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class ChatError(Exception):
    """Base exception for all chat-domain errors."""


class ChatCancelledError(ChatError):
    """Raised when a turn is cancelled by the user."""


class ChatToolError(ChatError):
    """Raised when a tool call fails."""

    def __init__(self, message: str, *, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(message)


class ChatMCPError(ChatError):
    """Raised when an MCP call fails."""

    def __init__(self, message: str, *, server_name: str, tool_name: str) -> None:
        self.server_name = server_name
        self.tool_name = tool_name
        super().__init__(message)


class ChatContextError(ChatError):
    """Raised when session/context state is invalid."""

    def __init__(self, message: str, *, field: str | None = None) -> None:
        self.field = field
        super().__init__(message)


class ChatAgentError(ChatError):
    """Raised when a sub-agent execution fails."""

    def __init__(self, message: str, *, agent_key: str | None = None) -> None:
        self.agent_key = agent_key
        super().__init__(message)


class ChatPayloadError(ChatError):
    """Raised when payload validation fails."""

    def __init__(self, message: str, *, payload_type: str | None = None) -> None:
        self.payload_type = payload_type
        super().__init__(message)


class ChatTurnError(ChatError):
    """Raised when a turn lifecycle error occurs."""

    def __init__(self, message: str, *, turn_id: int | None = None) -> None:
        self.turn_id = turn_id
        super().__init__(message)


__all__ = [
    "ChatError",
    "ChatCancelledError",
    "ChatToolError",
    "ChatMCPError",
    "ChatContextError",
    "ChatAgentError",
    "ChatPayloadError",
    "ChatTurnError",
]
