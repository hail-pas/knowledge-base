from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "file_source"
            ADD COLUMN "allowed_prefix_template" VARCHAR(500) NULL,
            ALTER COLUMN "access_key" TYPE VARCHAR(2048),
            ALTER COLUMN "secret_key" TYPE VARCHAR(2048);

        ALTER TABLE "embedding_model_config"
            ALTER COLUMN "api_key" TYPE VARCHAR(2048);

        ALTER TABLE "indexing_backend_config"
            ALTER COLUMN "password" TYPE VARCHAR(2048);

        ALTER TABLE "llm_model_config"
            ALTER COLUMN "api_key" TYPE VARCHAR(2048);
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        -- Keep widened credential columns to avoid truncating AES-GCM ciphertext.
        ALTER TABLE "file_source"
            DROP COLUMN "allowed_prefix_template";
    """
