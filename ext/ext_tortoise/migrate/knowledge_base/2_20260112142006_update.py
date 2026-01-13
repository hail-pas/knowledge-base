from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE `embedding_model_config` DROP INDEX `uid_embedding_m_name_ddb042`;
        ALTER TABLE `embedding_model_config` MODIFY COLUMN `type` VARCHAR(6) NOT NULL COMMENT '模型类型（openai/sentence_transformers等）';
        ALTER TABLE `embedding_model_config` ADD UNIQUE INDEX `uid_embedding_m_model_n_12ca67` (`model_name_or_path`, `deleted_at`);"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE `embedding_model_config` DROP INDEX `uid_embedding_m_model_n_12ca67`;
        ALTER TABLE `embedding_model_config` MODIFY COLUMN `type` VARCHAR(21) NOT NULL COMMENT '模型类型（openai/sentence_transformers等）';
        ALTER TABLE `embedding_model_config` ADD UNIQUE INDEX `uid_embedding_m_name_ddb042` (`name`, `deleted_at`);"""
