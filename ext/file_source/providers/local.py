"""Local filesystem provider with root confinement and symlink controls."""

import hashlib
import os
import stat
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path

import aiofiles

from ext.file_source.base import BaseFileSourceProvider, FileMetadata
from ext.file_source.types import LocalFileSourceExtraConfig


class LocalFileSourceProvider(BaseFileSourceProvider[LocalFileSourceExtraConfig]):
    extra_config_type = LocalFileSourceExtraConfig

    def _validate_config(self) -> None:
        if not self.storage_location:
            raise ValueError("storage_location is required for local file source")
        root = Path(self.storage_location).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("storage_location must be an existing directory")
        self._root = root

    def _resolve_path(self, uri: str, *, allow_missing: bool = False) -> Path:
        candidate = Path(uri)
        if candidate.is_absolute():
            path = candidate
        else:
            path = self._root / candidate
        if not self.extra_config.follow_symlinks:
            current = self._root
            try:
                relative = path.relative_to(self._root)
            except ValueError:
                raise ValueError("File path is outside the configured storage root") from None
            for part in relative.parts:
                current = current / part
                try:
                    if stat.S_ISLNK(current.lstat().st_mode):
                        raise ValueError("Symbolic links are disabled for this file source")
                except FileNotFoundError:
                    if not allow_missing:
                        raise
        resolved = path.resolve(strict=not allow_missing)
        if not resolved.is_relative_to(self._root):
            raise ValueError("File path is outside the configured storage root")
        return resolved

    async def validate_connection(self) -> bool:
        return self._root.is_dir() and os.access(self._root, os.R_OK | os.X_OK)

    async def list_files(
        self,
        prefix: str = "",
        recursive: bool = False,
        limit: int | None = None,
    ) -> list[FileMetadata]:
        if limit is not None and limit <= 0:
            return []
        search_root = self._resolve_path(prefix, allow_missing=False) if prefix else self._root
        pattern = "**/*" if recursive else "*"
        result: list[FileMetadata] = []
        for path in search_root.glob(pattern):
            if path.is_symlink() and not self.extra_config.follow_symlinks:
                continue
            resolved = self._resolve_path(str(path))
            if not resolved.is_file():
                continue
            if self.extra_config.allowed_extensions and resolved.suffix.lower() not in self.extra_config.allowed_extensions:
                continue
            if self.extra_config.excluded_extensions and resolved.suffix.lower() in self.extra_config.excluded_extensions:
                continue
            info = resolved.stat()
            if self.extra_config.max_file_size is not None and info.st_size > self.extra_config.max_file_size:
                continue
            if self.extra_config.require_readable and not os.access(resolved, os.R_OK):
                continue
            result.append(self._metadata(resolved, str(resolved), info))
            if limit is not None and len(result) >= limit:
                break
        return result

    async def get_file(self, uri: str) -> bytes:
        async with aiofiles.open(self._resolve_path(uri), "rb") as file:
            return await file.read()

    async def get_file_stream(self, uri: str, chunk_size: int = 8192) -> AsyncIterator[bytes]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        async with aiofiles.open(self._resolve_path(uri), "rb") as file:
            while chunk := await file.read(chunk_size):
                yield chunk

    async def get_file_metadata(self, uri: str) -> FileMetadata:
        path = self._resolve_path(uri)
        return self._metadata(path, uri, path.stat())

    async def file_exists(self, uri: str) -> bool:
        try:
            return self._resolve_path(uri).is_file()
        except FileNotFoundError:
            return False

    async def upload_file(self, uri: str, content: bytes, content_type: str | None = None) -> FileMetadata:
        path = self._resolve_path(uri, allow_missing=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path = self._resolve_path(uri, allow_missing=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        if not self.extra_config.follow_symlinks:
            flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as file:
            file.write(content)
        return await self.get_file_metadata(uri)

    async def delete_file(self, uri: str) -> bool:
        try:
            self._resolve_path(uri).unlink()
            return True
        except FileNotFoundError:
            return False

    @staticmethod
    def _metadata(path: Path, uri: str, info: os.stat_result) -> FileMetadata:
        with path.open("rb") as file:
            etag = hashlib.file_digest(file, "md5").hexdigest()
        return FileMetadata(
            uri=uri,
            file_name=path.name,
            file_size=info.st_size,
            last_modified=datetime.fromtimestamp(info.st_mtime, tz=timezone.utc),
            etag=etag,
        )
