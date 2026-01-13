from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE `embedding_model_config` ADD `max_token_per_text` INT NOT NULL COMMENT '单个文本的最大 token 长度' DEFAULT 512;"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE `embedding_model_config` DROP COLUMN `max_token_per_text`;"""
