from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import socket
import tempfile
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from config.main import local_configs
from ext.document_parser.core.engine_base import BaseEngine
from ext.document_parser.core.parse_result import ParseResult
from ext.document_parser.config.engine_registry import get_engine


class URLEngine(BaseEngine):
    engine_name = "url"
    supported_formats = []

    CONTENT_TYPE_ENGINE_MAP = {
        "text/html": ["trafilatura"],
        "application/pdf": ["pymupdf", "pdfplumber", "markitdown"],
        "image/png": ["markitdown", "paddleocr"],
        "image/jpeg": ["markitdown", "paddleocr"],
        "image/gif": ["markitdown"],
        "image/bmp": ["markitdown"],
        "image/tiff": ["paddleocr", "markitdown"],
        "application/vnd.ms-powerpoint": ["pptx", "markitdown"],
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": ["pptx", "markitdown"],
        "application/msword": ["docx", "markitdown"],
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ["docx", "markitdown"],
        "application/vnd.ms-excel": ["xlsx", "markitdown"],
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ["xlsx", "markitdown"],
        "text/plain": ["markdown", "markitdown"],
        "text/csv": ["csv"],
        "application/json": ["json"],
        "text/markdown": ["markdown", "markitdown"],
    }

    EXTENSION_MAP = {
        ".html": "trafilatura",
        ".htm": "trafilatura",
        ".pdf": ["pymupdf", "pdfplumber", "markitdown"],
        ".png": ["markitdown", "paddleocr"],
        ".jpg": ["markitdown", "paddleocr"],
        ".jpeg": ["markitdown", "paddleocr"],
        ".gif": ["markitdown"],
        ".bmp": ["markitdown"],
        ".tiff": ["paddleocr", "markitdown"],
        ".pptx": ["pptx", "markitdown"],
        ".ppt": ["pptx", "markitdown"],
        ".docx": ["docx", "markitdown"],
        ".doc": ["docx", "markitdown"],
        ".xlsx": ["xlsx", "markitdown"],
        ".xls": ["xlsx", "markitdown"],
        ".txt": ["markdown", "markitdown"],
        ".md": ["markdown", "markitdown"],
        ".csv": ["csv"],
        ".json": ["json"],
    }

    def can_parse(self, file_path: str) -> bool:
        return file_path.lower().startswith(("http://", "https://"))

    async def _validate_public_url(self, url: str) -> None:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Only absolute HTTP(S) URLs are supported")
        if parsed.username or parsed.password:
            raise ValueError("URLs containing credentials are not allowed")

        host = parsed.hostname
        try:
            addresses = {ipaddress.ip_address(host)}
        except ValueError:
            loop = asyncio.get_running_loop()
            records = await loop.getaddrinfo(
                host,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
            addresses = {ipaddress.ip_address(record[4][0]) for record in records}

        if not addresses or any(not address.is_global for address in addresses):
            raise ValueError("URL host must resolve exclusively to public IP addresses")

    async def _download_url(self, url: str) -> tuple[str, str, str | None]:
        """Download a public URL to a bounded temporary file."""
        client = local_configs.extensions.httpx.instance
        current_url = url
        max_size = 100 * 1024 * 1024

        for redirect_count in range(6):
            await self._validate_public_url(current_url)
            async with client.stream(
                "GET",
                current_url,
                follow_redirects=False,
                timeout=30.0,
            ) as response:
                # DNS validation alone is vulnerable to rebinding. Verify the actual peer
                # for every response, including redirects, and fail closed if unavailable.
                network_stream = response.extensions.get("network_stream")
                if network_stream is None:
                    raise ValueError("Unable to verify remote URL connection address")

                server_addr = network_stream.get_extra_info("server_addr")
                if not server_addr:
                    raise ValueError("Unable to verify remote URL connection address")

                peer = server_addr[0] if isinstance(server_addr, (tuple, list)) else server_addr
                try:
                    peer_ip = ipaddress.ip_address(peer)
                except ValueError as exc:
                    raise ValueError("Remote URL returned an invalid peer address") from exc
                if not peer_ip.is_global:
                    raise ValueError("URL connected to a non-public IP address")

                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location or redirect_count == 5:
                        raise ValueError("URL redirect limit exceeded or Location header missing")
                    current_url = urljoin(current_url, location)
                    continue

                response.raise_for_status()

                content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
                extension = self._get_extension_from_content_type(content_type) or self._get_extension_from_url(
                    current_url,
                )
                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        declared_size = int(content_length)
                    except ValueError as exc:
                        raise ValueError("Remote document has an invalid Content-Length") from exc
                    if declared_size < 0 or declared_size > max_size:
                        raise ValueError("Remote document exceeds the 100 MiB download limit")

                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=extension or ".tmp") as tmp:
                        tmp_path = tmp.name
                        total_size = 0
                        async for chunk in response.aiter_bytes():
                            total_size += len(chunk)
                            if total_size > max_size:
                                raise ValueError("Remote document exceeds the 100 MiB download limit")
                            tmp.write(chunk)
                    return tmp_path, content_type, extension
                except Exception:
                    if tmp_path and Path(tmp_path).exists():
                        with contextlib.suppress(OSError):
                            Path(tmp_path).unlink()
                    raise

        raise ValueError("URL redirect limit exceeded")

    @staticmethod
    def _safe_source_url(url: str) -> str:
        """Remove query strings and fragments before storing URL metadata."""
        return urlsplit(url)._replace(query="", fragment="").geturl()

    async def parse(self, file_path: str, options: dict | None = None) -> ParseResult:
        tmp_path = None
        safe_url = self._safe_source_url(file_path)
        try:
            tmp_path, content_type, extension = await self._download_url(file_path)
            engine_names = self._get_engines_for_content(content_type, extension)
            last_error = None

            for engine_name in engine_names:
                try:
                    engine_instance = get_engine(engine_name)
                    if not engine_instance:
                        continue

                    result = await engine_instance.parse(tmp_path, options)
                    result.metadata["source_url"] = safe_url
                    result.metadata["content_type"] = content_type
                    return result
                except Exception as exc:
                    last_error = exc

            raise RuntimeError(
                f"All delegated engines failed for remote document {safe_url}"
                + (f" ({type(last_error).__name__})" if last_error else ""),
            )
        finally:
            if tmp_path and Path(tmp_path).exists():
                with contextlib.suppress(OSError):
                    Path(tmp_path).unlink()

    def _get_extension_from_content_type(self, content_type: str) -> str | None:
        """Map content-type to file extension."""
        content_type_ext_map = {
            "application/pdf": ".pdf",
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/gif": ".gif",
            "image/bmp": ".bmp",
            "image/tiff": ".tiff",
            "text/html": ".html",
            "text/plain": ".txt",
            "text/markdown": ".md",
            "text/csv": ".csv",
            "application/json": ".json",
            "application/vnd.ms-powerpoint": ".ppt",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
            "application/msword": ".doc",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
            "application/vnd.ms-excel": ".xls",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        }
        return content_type_ext_map.get(content_type)

    def _get_extension_from_url(self, url: str) -> str | None:
        """Extract extension from URL."""
        path = Path(urlsplit(url).path)
        if path.suffix:
            return path.suffix.lower()
        return None

    def _get_engines_for_content(self, content_type: str, ext: str | None) -> list[str]:
        """Determine which engines to try for the given content."""
        # First, try content-type based mapping
        engines = self.CONTENT_TYPE_ENGINE_MAP.get(content_type, [])

        # If no engines from content-type, try extension-based mapping
        if not engines and ext:
            ext_engines = self.EXTENSION_MAP.get(ext.lower(), [])
            if ext_engines:
                engines = ext_engines

        # Default to markitdown as universal fallback
        if not engines:
            engines = ["markitdown"]

        # Normalize to list
        if isinstance(engines, str):
            engines = [engines]

        return engines
