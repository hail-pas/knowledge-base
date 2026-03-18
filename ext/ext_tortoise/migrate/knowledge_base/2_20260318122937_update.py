from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE `chat_capability_profile` DROP INDEX `uid_chat_capabi_name_4d5495`;
        ALTER TABLE `chat_capability_profile` ADD `owner_account_id` BIGINT COMMENT '所属账户ID，空表示全局';
        ALTER TABLE `chat_capability_profile` ADD UNIQUE INDEX `uid_chat_capabi_owner_a_9a888b` (`owner_account_id`, `name`, `deleted_at`);
        ALTER TABLE `chat_capability_profile` ADD INDEX `idx_chat_capabi_owner_a_1331eb` (`owner_account_id`, `kind`, `is_enabled`);
        ALTER TABLE `chat_capability_profile` ADD INDEX `idx_chat_capabi_owner_a_53129e` (`owner_account_id`, `is_enabled`);"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE `chat_capability_profile` DROP INDEX `idx_chat_capabi_owner_a_53129e`;
        ALTER TABLE `chat_capability_profile` DROP INDEX `idx_chat_capabi_owner_a_1331eb`;
        ALTER TABLE `chat_capability_profile` DROP INDEX `uid_chat_capabi_owner_a_9a888b`;
        ALTER TABLE `chat_capability_profile` DROP COLUMN `owner_account_id`;
        ALTER TABLE `chat_capability_profile` ADD UNIQUE INDEX `uid_chat_capabi_name_4d5495` (`name`, `deleted_at`);"""
