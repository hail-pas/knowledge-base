from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "file_source"
            ADD COLUMN "allowed_prefix_template" VARCHAR(500) NULL,
            ALTER COLUMN "access_key" TYPE VARCHAR(1024),
            ALTER COLUMN "secret_key" TYPE VARCHAR(1024);

        ALTER TABLE "embedding_model_config"
            ALTER COLUMN "api_key" TYPE VARCHAR(1024);

        ALTER TABLE "indexing_backend_config"
            ALTER COLUMN "password" TYPE VARCHAR(1024);

        ALTER TABLE "llm_model_config"
            ALTER COLUMN "api_key" TYPE VARCHAR(1024);
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "llm_model_config"
            ALTER COLUMN "api_key" TYPE VARCHAR(500);

        ALTER TABLE "indexing_backend_config"
            ALTER COLUMN "password" TYPE VARCHAR(500);

        ALTER TABLE "embedding_model_config"
            ALTER COLUMN "api_key" TYPE VARCHAR(500);

        ALTER TABLE "file_source"
            ALTER COLUMN "access_key" TYPE VARCHAR(500),
            ALTER COLUMN "secret_key" TYPE VARCHAR(500),
            DROP COLUMN "allowed_prefix_template";
    """
