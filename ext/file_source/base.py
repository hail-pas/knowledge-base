"""Shared file-source provider contract and common file operations."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generic, TypeVar

import aiofiles
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from ext.file_source.types import BaseFileSourceExtraConfig

ExtraConfigT = TypeVar("ExtraConfigT", bound=BaseFileSourceExtraConfig)


class FileMetadata(BaseModel):
    """Provider-independent file metadata."""

    model_config = ConfigDict(extra="forbid")

    uri: str
    file_name: str
    file_size: int = Field(ge=0)
    last_modified: datetime | None = None
    etag: str | None = None
    content_type: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class BaseFileSourceProvider(ABC, Generic[ExtraConfigT]):
    """Common asynchronous interface for local and object storage providers."""

    extra_config_type: type[BaseFileSourceExtraConfig] = BaseFileSourceExtraConfig

    def __init__(
        self,
        access_key: str | None,
        secret_key: str | None,
        endpoint: str | None,
        region: str | None,
        storage_location: str | None,
        use_ssl: bool,
        verify_ssl: bool,
        timeout: int,
        max_retries: int,
        concurrent_limit: int,
        max_connections: int,
        extra_config: dict[str, Any] | None,
    ) -> None:
        self.access_key = access_key
        self.secret_key = secret_key
        self.endpoint = endpoint
        self.region = region
        self.storage_location = storage_location
        self.use_ssl = use_ssl
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.max_retries = max_retries
        self.concurrent_limit = concurrent_limit
        self.max_connections = max_connections
        self.extra_config = self.extra_config_type.model_validate(extra_config or {})
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate provider-specific configuration."""

    @abstractmethod
    async def validate_connection(self) -> bool: ...

    @abstractmethod
    async def list_files(
        self,
        prefix: str = "",
        recursive: bool = False,
        limit: int | None = None,
    ) -> list[FileMetadata]: ...

    @abstractmethod
    async def get_file(self, uri: str) -> bytes: ...

    @abstractmethod
    def get_file_stream(self, uri: str, chunk_size: int = 8192) -> AsyncIterator[bytes]: ...

    @abstractmethod
    async def get_file_metadata(self, uri: str) -> FileMetadata: ...

    @abstractmethod
    async def file_exists(self, uri: str) -> bool: ...

    async def upload_file(self, uri: str, content: bytes, content_type: str | None = None) -> FileMetadata:
        raise NotImplementedError(f"{type(self).__name__} does not support upload")

    async def delete_file(self, uri: str) -> bool:
        """Return False only when the target does not exist; propagate operational errors."""
        raise NotImplementedError(f"{type(self).__name__} does not support delete")

    async def upload_from_local(self, local_path: str, uri: str, content_type: str | None = None) -> FileMetadata:
        async with aiofiles.open(local_path, "rb") as file:
            return await self.upload_file(uri, await file.read(), content_type)

    async def download_to_local(self, uri: str, local_path: str) -> None:
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(target, "wb") as file:
            async for chunk in self.get_file_stream(uri):
                await file.write(chunk)

    async def health_check(self) -> dict[str, str | bool]:
        try:
            connected = await self.validate_connection()
            return {
                "status": "healthy" if connected else "unhealthy",
                "connected": connected,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            logger.exception("File-source health check failed")
            return {
                "status": "error",
                "error": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
