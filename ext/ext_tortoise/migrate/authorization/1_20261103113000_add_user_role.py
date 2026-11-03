from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE `role`
            ADD COLUMN `is_super_admin` BOOL NOT NULL DEFAULT 0 COMMENT '是否为系统超级管理员';

        CREATE TABLE IF NOT EXISTS `user_role` (
            `id` BIGINT NOT NULL PRIMARY KEY AUTO_INCREMENT COMMENT '主键',
            `created_at` DATETIME(6) NOT NULL COMMENT '创建时间' DEFAULT CURRENT_TIMESTAMP(6),
            `updated_at` DATETIME(6) NOT NULL COMMENT '更新时间' DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
            `deleted_at` BIGINT COMMENT '删除时间',
            `user_id` CHAR(36) NOT NULL COMMENT 'Pero 用户 UUID',
            `role_id` BIGINT NOT NULL COMMENT '本地角色ID',
            CONSTRAINT `fk_user_role_role` FOREIGN KEY (`role_id`) REFERENCES `role` (`id`) ON DELETE CASCADE,
            UNIQUE KEY `uid_user_role_user_role_deleted` (`user_id`, `role_id`, `deleted_at`),
            KEY `idx_user_role_user_id` (`user_id`),
            KEY `idx_user_role_created_at` (`created_at`)
        ) CHARACTER SET utf8mb4 COMMENT='Pero 用户与本地角色关联';
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS `user_role`;
        ALTER TABLE `role` DROP COLUMN `is_super_admin`;
    """
