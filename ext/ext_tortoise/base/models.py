import uuid
import datetime

from collections.abc import Iterable
from ulid import ULID
from tortoise import fields, manager
from tortoise.models import Model
from tortoise.queryset import QuerySet
from tortoise.backends.base.client import BaseDBAsyncClient

from ext.ext_tortoise.base.fields import TimestampField, BinaryUUIDField


class NotDeletedManager(manager.Manager):
    def get_queryset(self) -> QuerySet:
        return super().get_queryset().filter(deleted_at=0)


class UUIDPrimaryKeyModel(Model):
    id = BinaryUUIDField(
        description="主键",
        pk=True,
        default=lambda: ULID().to_uuid(),
    )

    class Meta: # type: ignore
        abstract = True


class BigIntegerIDPrimaryKeyModel(Model):
    id = fields.BigIntField(description="主键", pk=True)

    class Meta: # type: ignore
        abstract = True


class TimeStampModel(Model):
    created_at = fields.DatetimeField(
        auto_now_add=True,
        description="创建时间",
        index=True,
    )
    updated_at = fields.DatetimeField(auto_now=True, description="更新时间")
    deleted_at = TimestampField(
        index=True,
        description="删除时间",
    )

    all_objects = manager.Manager()

    class Meta: # type: ignore
        abstract = True

    async def save(
        self,
        using_db: BaseDBAsyncClient | None = None,
        update_fields: Iterable[str] | None = None,
        force_create: bool = False,
        force_update: bool = False,
    ) -> None:
        if update_fields:
            update_fields = list(update_fields) + ["updated_at" ]
        await super().save(using_db, update_fields, force_create, force_update)

    async def real_delete(
        self,
        using_db: BaseDBAsyncClient | None = None,
    ) -> None:
        await super().delete(using_db)

    async def delete(
        self,
        using_db: BaseDBAsyncClient | None = None,
    ) -> None:
        """fake delete"""
        self.deleted_at = datetime.datetime.now(
            tz=self._meta.fields_map["deleted_at"].timezone,  # type: ignore
        )
        await self.save(
            using_db=using_db,
            update_fields=["deleted_at"],
            force_update=True,
        )

    @classmethod
    async def delete_by_ids(cls, ids: list[int | str | uuid.UUID]) -> int:
        """batch fake delete"""
        now = datetime.datetime.now(
            tz=cls._meta.fields_map["deleted_at"].timezone,  # type: ignore
        )
        return await cls.filter(id__in=ids, deleted_at=0).update(deleted_at=now, updated_at=now)


class BaseModel(BigIntegerIDPrimaryKeyModel, TimeStampModel):
    class Meta: # type: ignore
        abstract = True


class CreateOnlyModel(Model):
    created_at = fields.DatetimeField(
        auto_now_add=True,
        description="创建时间",
        index=True,
    )

    class Meta: # type: ignore
        abstract = True
