from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS `chat_capability_package` (
    `id` BIGINT NOT NULL PRIMARY KEY AUTO_INCREMENT COMMENT '主键',
    `created_at` DATETIME(6) NOT NULL COMMENT '创建时间' DEFAULT CURRENT_TIMESTAMP(6),
    `updated_at` DATETIME(6) NOT NULL COMMENT '更新时间' DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    `deleted_at` BIGINT COMMENT '删除时间',
    `owner_account_id` BIGINT COMMENT '所属账户ID，空表示全局',
    `kind` VARCHAR(20) NOT NULL COMMENT '能力包类型',
    `capability_key` VARCHAR(128) NOT NULL COMMENT '能力包唯一键',
    `name` VARCHAR(128) NOT NULL COMMENT '能力包名称',
    `description` LONGTEXT NOT NULL COMMENT '能力包描述',
    `manifest` JSON NOT NULL COMMENT '能力包 manifest',
    `is_enabled` BOOL NOT NULL COMMENT '是否启用' DEFAULT 1,
    `metadata` JSON NOT NULL COMMENT '附加元数据',
    `version` INT NOT NULL COMMENT '版本号' DEFAULT 1,
    UNIQUE KEY `uid_chat_cap_pkg_owner_kind_key` (`owner_account_id`, `kind`, `capability_key`, `deleted_at`),
    KEY `idx_chat_cap_pkg_created_at` (`created_at`),
    KEY `idx_chat_cap_pkg_deleted_at` (`deleted_at`),
    KEY `idx_chat_cap_pkg_owner_kind_enabled` (`owner_account_id`, `kind`, `is_enabled`),
    KEY `idx_chat_cap_pkg_kind_key_enabled` (`kind`, `capability_key`, `is_enabled`)
) CHARACTER SET utf8mb4 COMMENT='聊天能力包定义表';
        DROP TABLE IF EXISTS `chat_skill`;
        DROP TABLE IF EXISTS `chat_capability_binding`;
        DROP TABLE IF EXISTS `chat_capability_profile`;"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS `chat_capability_package`;"""
