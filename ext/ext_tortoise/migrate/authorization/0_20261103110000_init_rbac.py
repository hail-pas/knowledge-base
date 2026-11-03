from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    """Create local RBAC tables without creating any user/account table.

    Existing installations may already have role/permission/resource tables in
    the former rdb_user_center database. IF NOT EXISTS preserves those records.
    """

    return """
        CREATE TABLE IF NOT EXISTS `aerich` (
            `id` INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
            `version` VARCHAR(255) NOT NULL,
            `app` VARCHAR(100) NOT NULL,
            `content` JSON NOT NULL
        ) CHARACTER SET utf8mb4;

        CREATE TABLE IF NOT EXISTS `role` (
            `id` BIGINT NOT NULL PRIMARY KEY AUTO_INCREMENT COMMENT '主键',
            `created_at` DATETIME(6) NOT NULL COMMENT '创建时间' DEFAULT CURRENT_TIMESTAMP(6),
            `updated_at` DATETIME(6) NOT NULL COMMENT '更新时间' DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
            `deleted_at` BIGINT COMMENT '删除时间',
            `label` VARCHAR(50) NOT NULL COMMENT '角色名称',
            `remark` VARCHAR(200) COMMENT '备注',
            UNIQUE KEY `uid_role_label_47a519` (`label`, `deleted_at`),
            KEY `idx_role_created_7f5f71` (`created_at`),
            KEY `idx_role_deleted_2bed69` (`deleted_at`)
        ) CHARACTER SET utf8mb4 COMMENT='角色';

        CREATE TABLE IF NOT EXISTS `permission` (
            `code` VARCHAR(256) NOT NULL PRIMARY KEY COMMENT '权限编码',
            `label` VARCHAR(128) NOT NULL COMMENT '权限名称',
            `permission_type` VARCHAR(16) NOT NULL COMMENT '权限类型' DEFAULT 'api',
            `is_deprecated` BOOL NOT NULL COMMENT '是否废弃' DEFAULT 0
        ) CHARACTER SET utf8mb4 COMMENT='权限';

        CREATE TABLE IF NOT EXISTS `resource` (
            `id` BIGINT NOT NULL PRIMARY KEY AUTO_INCREMENT COMMENT '主键',
            `created_at` DATETIME(6) NOT NULL COMMENT '创建时间' DEFAULT CURRENT_TIMESTAMP(6),
            `code` VARCHAR(32) NOT NULL COMMENT '资源编码',
            `label` VARCHAR(64) NOT NULL COMMENT '资源名称',
            `front_route` VARCHAR(128) COMMENT '前端路由',
            `resource_type` VARCHAR(16) NOT NULL COMMENT '资源类型',
            `sub_resource_type` VARCHAR(16) NOT NULL COMMENT '资源子类型',
            `order_num` INT NOT NULL COMMENT '排列序号' DEFAULT 1,
            `enabled` BOOL NOT NULL COMMENT '是否启用' DEFAULT 1,
            `assignable` BOOL NOT NULL COMMENT '是否允许分配' DEFAULT 1,
            `parent_id` BIGINT COMMENT '父级资源',
            `scene` VARCHAR(16) NOT NULL COMMENT '客户端场景',
            UNIQUE KEY `uid_resource_code_8b659a` (`code`, `parent_id`, `scene`),
            CONSTRAINT `fk_resource_resource_61c52602` FOREIGN KEY (`parent_id`) REFERENCES `resource` (`id`) ON DELETE CASCADE,
            KEY `idx_resource_created_669b66` (`created_at`),
            KEY `idx_resource_code_edb401` (`code`),
            KEY `idx_resource_label_a90213` (`label`)
        ) CHARACTER SET utf8mb4 COMMENT='系统资源';

        CREATE TABLE IF NOT EXISTS `resource_permission` (
            `resource_id` BIGINT NOT NULL,
            `permission_id` VARCHAR(256) NOT NULL,
            FOREIGN KEY (`resource_id`) REFERENCES `resource` (`id`) ON DELETE CASCADE,
            FOREIGN KEY (`permission_id`) REFERENCES `permission` (`code`) ON DELETE CASCADE,
            UNIQUE KEY `uidx_resource_pe_resourc_4b71b7` (`resource_id`, `permission_id`)
        ) CHARACTER SET utf8mb4;

        CREATE TABLE IF NOT EXISTS `resource_role` (
            `resource_id` BIGINT NOT NULL,
            `role_id` BIGINT NOT NULL,
            FOREIGN KEY (`resource_id`) REFERENCES `resource` (`id`) ON DELETE CASCADE,
            FOREIGN KEY (`role_id`) REFERENCES `role` (`id`) ON DELETE CASCADE,
            UNIQUE KEY `uidx_resource_ro_resourc_a19b78` (`resource_id`, `role_id`)
        ) CHARACTER SET utf8mb4;
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS `resource_role`;
        DROP TABLE IF EXISTS `resource_permission`;
        DROP TABLE IF EXISTS `resource`;
        DROP TABLE IF EXISTS `permission`;
        DROP TABLE IF EXISTS `role`;
    """
