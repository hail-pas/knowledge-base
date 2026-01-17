from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS `config` (
    `id` INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    `key` VARCHAR(128) NOT NULL UNIQUE COMMENT '配置项key',
    `value` JSON NOT NULL COMMENT '配置项值',
    `description` VARCHAR(255) NOT NULL COMMENT '配置项描述'
) CHARACTER SET utf8mb4 COMMENT='系统配置';
CREATE TABLE IF NOT EXISTS `permission` (
    `code` VARCHAR(256) NOT NULL PRIMARY KEY COMMENT '权限码',
    `label` VARCHAR(128) NOT NULL COMMENT '权限名称',
    `permission_type` VARCHAR(16) NOT NULL COMMENT '权限类型, api: API' DEFAULT 'api',
    `is_deprecated` BOOL NOT NULL COMMENT '是否废弃' DEFAULT 0
) CHARACTER SET utf8mb4 COMMENT='权限';
CREATE TABLE IF NOT EXISTS `resource` (
    `id` BIGINT NOT NULL PRIMARY KEY AUTO_INCREMENT COMMENT '主键',
    `created_at` DATETIME(6) NOT NULL COMMENT '创建时间' DEFAULT CURRENT_TIMESTAMP(6),
    `code` VARCHAR(32) NOT NULL COMMENT '资源编码{parent}:{current}',
    `label` VARCHAR(64) NOT NULL COMMENT '资源名称',
    `front_route` VARCHAR(128) COMMENT '前端路由',
    `resource_type` VARCHAR(16) NOT NULL COMMENT '资源类型, menu: 菜单, button: 按钮, api: 接口',
    `sub_resource_type` VARCHAR(16) NOT NULL COMMENT '资源类型, add_tab: 选项卡, dialog: 弹窗, ajax: Ajax请求, link: 链接',
    `order_num` INT NOT NULL COMMENT '排列序号' DEFAULT 1,
    `enabled` BOOL NOT NULL COMMENT '是否可用' DEFAULT 1,
    `assignable` BOOL NOT NULL COMMENT '是否可分配' DEFAULT 1,
    `scene` VARCHAR(16) NOT NULL COMMENT '场景',
    `parent_id` BIGINT COMMENT '父级',
    UNIQUE KEY `uid_resource_code_8b659a` (`code`, `parent_id`, `scene`),
    CONSTRAINT `fk_resource_resource_61c52602` FOREIGN KEY (`parent_id`) REFERENCES `resource` (`id`) ON DELETE CASCADE,
    KEY `idx_resource_created_669b66` (`created_at`),
    KEY `idx_resource_code_edb401` (`code`),
    KEY `idx_resource_label_a90213` (`label`)
) CHARACTER SET utf8mb4 COMMENT='系统资源';
CREATE TABLE IF NOT EXISTS `role` (
    `id` BIGINT NOT NULL PRIMARY KEY AUTO_INCREMENT COMMENT '主键',
    `created_at` DATETIME(6) NOT NULL COMMENT '创建时间' DEFAULT CURRENT_TIMESTAMP(6),
    `updated_at` DATETIME(6) NOT NULL COMMENT '更新时间' DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    `deleted_at` BIGINT COMMENT '删除时间',
    `label` VARCHAR(50) NOT NULL COMMENT '名称',
    `remark` VARCHAR(200) COMMENT '备注',
    UNIQUE KEY `uid_role_label_47a519` (`label`, `deleted_at`),
    KEY `idx_role_created_7f5f71` (`created_at`),
    KEY `idx_role_deleted_2bed69` (`deleted_at`)
) CHARACTER SET utf8mb4 COMMENT='角色';
CREATE TABLE IF NOT EXISTS `account` (
    `id` BIGINT NOT NULL PRIMARY KEY AUTO_INCREMENT COMMENT '主键',
    `created_at` DATETIME(6) NOT NULL COMMENT '创建时间' DEFAULT CURRENT_TIMESTAMP(6),
    `updated_at` DATETIME(6) NOT NULL COMMENT '更新时间' DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    `deleted_at` BIGINT COMMENT '删除时间',
    `username` VARCHAR(20) NOT NULL COMMENT '用户名',
    `phone` VARCHAR(11) NOT NULL COMMENT '手机号',
    `email` VARCHAR(50) COMMENT '邮箱',
    `is_staff` BOOL NOT NULL COMMENT '是否是后台管理员' DEFAULT 0,
    `is_super_admin` BOOL NOT NULL COMMENT '是否是后台超级管理员' DEFAULT 0,
    `status` VARCHAR(16) NOT NULL COMMENT '状态' DEFAULT 'enable',
    `last_login_at` DATETIME(6) COMMENT '最近一次登录时间',
    `remark` VARCHAR(200) NOT NULL COMMENT '备注' DEFAULT '',
    `password` VARCHAR(255) NOT NULL COMMENT '密码',
    `role_id` BIGINT NOT NULL COMMENT '角色',
    UNIQUE KEY `uid_account_usernam_4f1849` (`username`, `deleted_at`),
    UNIQUE KEY `uid_account_phone_9b9e7e` (`phone`, `deleted_at`),
    UNIQUE KEY `uid_account_email_a28fc7` (`email`, `deleted_at`),
    CONSTRAINT `fk_account_role_7f75e8d5` FOREIGN KEY (`role_id`) REFERENCES `role` (`id`) ON DELETE CASCADE,
    KEY `idx_account_created_028865` (`created_at`),
    KEY `idx_account_deleted_c0aa6b` (`deleted_at`)
) CHARACTER SET utf8mb4 COMMENT='用户';
CREATE TABLE IF NOT EXISTS `aerich` (
    `id` INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    `version` VARCHAR(255) NOT NULL,
    `app` VARCHAR(100) NOT NULL,
    `content` JSON NOT NULL
) CHARACTER SET utf8mb4;
CREATE TABLE IF NOT EXISTS `resource_role` (
    `resource_id` BIGINT NOT NULL,
    `role_id` BIGINT NOT NULL,
    FOREIGN KEY (`resource_id`) REFERENCES `resource` (`id`) ON DELETE CASCADE,
    FOREIGN KEY (`role_id`) REFERENCES `role` (`id`) ON DELETE CASCADE,
    UNIQUE KEY `uidx_resource_ro_resourc_a19b78` (`resource_id`, `role_id`)
) CHARACTER SET utf8mb4;
CREATE TABLE IF NOT EXISTS `resource_permission` (
    `resource_id` BIGINT NOT NULL,
    `permission_id` VARCHAR(256) NOT NULL,
    FOREIGN KEY (`resource_id`) REFERENCES `resource` (`id`) ON DELETE CASCADE,
    FOREIGN KEY (`permission_id`) REFERENCES `permission` (`code`) ON DELETE CASCADE,
    UNIQUE KEY `uidx_resource_pe_resourc_4b71b7` (`resource_id`, `permission_id`)
) CHARACTER SET utf8mb4;"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        """
