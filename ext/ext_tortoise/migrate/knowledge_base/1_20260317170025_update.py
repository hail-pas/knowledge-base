from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS `chat_capability_profile` (
    `id` BIGINT NOT NULL PRIMARY KEY AUTO_INCREMENT COMMENT '主键',
    `created_at` DATETIME(6) NOT NULL COMMENT '创建时间' DEFAULT CURRENT_TIMESTAMP(6),
    `updated_at` DATETIME(6) NOT NULL COMMENT '更新时间' DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    `deleted_at` BIGINT COMMENT '删除时间',
    `name` VARCHAR(128) NOT NULL COMMENT 'Capability 名称',
    `kind` VARCHAR(64) NOT NULL COMMENT 'Capability 类型',
    `description` LONGTEXT NOT NULL COMMENT '描述',
    `config` JSON NOT NULL COMMENT 'Capability 配置',
    `is_enabled` BOOL NOT NULL COMMENT '是否启用' DEFAULT 1,
    `metadata` JSON NOT NULL COMMENT '附加元数据',
    `version` INT NOT NULL COMMENT '版本号' DEFAULT 1,
    UNIQUE KEY `uid_chat_capabi_name_4d5495` (`name`, `deleted_at`),
    KEY `idx_chat_capabi_created_1a833d` (`created_at`),
    KEY `idx_chat_capabi_deleted_c1b3a3` (`deleted_at`),
    KEY `idx_chat_capabi_kind_d5b602` (`kind`, `is_enabled`),
    KEY `idx_chat_capabi_is_enab_201b50` (`is_enabled`)
    ) CHARACTER SET utf8mb4 COMMENT='聊天能力配置表';
        CREATE TABLE IF NOT EXISTS `chat_capability_binding` (
    `id` BIGINT NOT NULL PRIMARY KEY AUTO_INCREMENT COMMENT '主键',
    `created_at` DATETIME(6) NOT NULL COMMENT '创建时间' DEFAULT CURRENT_TIMESTAMP(6),
    `updated_at` DATETIME(6) NOT NULL COMMENT '更新时间' DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    `deleted_at` BIGINT COMMENT '删除时间',
    `owner_type` VARCHAR(32) NOT NULL COMMENT '绑定对象类型',
    `owner_id` BIGINT COMMENT '绑定对象ID',
    `priority` INT NOT NULL COMMENT '执行优先级' DEFAULT 100,
    `is_enabled` BOOL NOT NULL COMMENT '是否启用' DEFAULT 1,
    `metadata` JSON NOT NULL COMMENT '附加元数据',
    `capability_profile_id` BIGINT NOT NULL COMMENT '关联能力配置',
    UNIQUE KEY `uid_chat_capabi_owner_t_c8f6ae` (`owner_type`, `owner_id`, `capability_profile_id`, `deleted_at`),
    CONSTRAINT `fk_chat_cap_chat_cap_a15caf28` FOREIGN KEY (`capability_profile_id`) REFERENCES `chat_capability_profile` (`id`) ON DELETE CASCADE,
    KEY `idx_chat_capabi_created_f4a696` (`created_at`),
    KEY `idx_chat_capabi_deleted_319e6d` (`deleted_at`),
    KEY `idx_chat_capabi_owner_t_1e75cd` (`owner_type`, `owner_id`, `is_enabled`),
    KEY `idx_chat_capabi_capabil_641cb1` (`capability_profile_id`, `is_enabled`)
) CHARACTER SET utf8mb4 COMMENT='聊天能力绑定表';"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS `chat_capability_profile`;
        DROP TABLE IF EXISTS `chat_capability_binding`;"""
