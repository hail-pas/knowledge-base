from typing import override
from functools import cached_property
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from pydantic import RedisDsn
from redis.retry import Retry
from redis.asyncio import Redis
from redis.backoff import NoBackoff

from config.default import InstanceExtensionConfig, RegisterExtensionConfig


class RedisConfig(RegisterExtensionConfig, InstanceExtensionConfig[AsyncGenerator[Redis]]):
    url: RedisDsn
    max_connections: int = 10

    @cached_property
    def client(self) -> Redis:
        return Redis.from_url(
            str(self.url),
            max_connections=self.max_connections,
            decode_responses=True,
            encoding_errors="strict",
            retry=Retry(NoBackoff(), retries=10),
            health_check_interval=30,
        )

    @property
    @asynccontextmanager
    @override
    async def instance(self) -> AsyncGenerator[Redis]:
        yield self.client

    @override
    async def register(self) -> None: ...

    @override
    async def unregister(self) -> None:
        client = self.__dict__.get("client")
        if client is not None:
            await client.aclose()
