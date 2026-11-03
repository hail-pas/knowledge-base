from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "collection"
            DROP COLUMN "tenant_id",
            DROP COLUMN "role_id";
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "collection"
            ADD COLUMN "tenant_id" UUID NULL,
            ADD COLUMN "role_id" UUID NULL;
    """
