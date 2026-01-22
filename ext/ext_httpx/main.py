import httpx
from typing import override
from config.default import RegisterExtensionConfig, InstanceExtensionConfig
from loguru import logger


class HttpxConfig(RegisterExtensionConfig, InstanceExtensionConfig):
    """httpx 配置类，负责 httpx 客户端的生命周期管理"""

    _client: httpx.AsyncClient | None = None

    max_connections: int = 200
    max_keepalive_connections: int = 80
    timeout: float = 60.0
    request_from_header: str = ""

    @property
    def instance(self) -> httpx.AsyncClient:
        """获取当前实例的 httpx 客户端"""
        if self._client is None:
            raise RuntimeError("Httpx client not initialized. Make sure register() has been called.")
        return self._client

    @override
    async def register(self) -> None:
        """初始化 httpx.AsyncClient"""
        if self._client is not None:
            return

        headers = {}
        if self.request_from_header:
            headers["X-Request-From"] = self.request_from_header

        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            limits=httpx.Limits(
                max_connections=self.max_connections,
                max_keepalive_connections=self.max_keepalive_connections,
            ),
            http2=False,
            headers=headers,
        )
        logger.info("Httpx client initialized")

    @override
    async def unregister(self) -> None:
        """关闭 httpx.AsyncClient"""
        if self._client is None:
            return

        try:
            await self._client.aclose()
        except RuntimeError as e:
            if "Event loop is closed" not in str(e):
                raise
            logger.debug("Event loop already closed, skipping httpx client cleanup")
        self._client = None
        logger.info("Httpx client closed")
