from typing import override
from functools import cached_property
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from pydantic import RedisDsn
from redis.retry import Retry
from redis.asyncio import Redis, ConnectionPool
from redis.backoff import NoBackoff

from config.default import InstanceExtensionConfig, RegisterExtensionConfig


class RedisConfig(RegisterExtensionConfig, InstanceExtensionConfig[AsyncGenerator[Redis, None]]):
    url: RedisDsn
    max_connections: int = 10

    @cached_property
    def connection_pool(self) -> ConnectionPool:
        return ConnectionPool.from_url(  # type: ignore
            url=str(self.url),
            max_connections=self.max_connections,
            decode_responses=True,
            encoding_errors="strict",
            retry=Retry(NoBackoff(), retries=10),
            health_check_interval=30,
        )

    @property
    @asynccontextmanager
    @override
    async def instance(self) -> AsyncGenerator[Redis, None]:  # type: ignore
        r: Redis | None = None
        try:
            r = Redis.from_pool(
                connection_pool=self.connection_pool,
            )
            yield r
        finally:
            if r:
                await r.close()

    @override
    async def register(self) -> None: ...

    @override
    async def unregister(self) -> None:
        if self.connection_pool:
            await self.connection_pool.aclose()
