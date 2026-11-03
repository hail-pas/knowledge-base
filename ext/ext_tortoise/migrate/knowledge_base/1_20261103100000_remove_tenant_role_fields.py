from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE `collection`
            DROP COLUMN `tenant_id`,
            DROP COLUMN `role_id`;
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE `collection`
            ADD COLUMN `tenant_id` CHAR(36) NULL COMMENT '租户ID',
            ADD COLUMN `role_id` CHAR(36) NULL COMMENT '角色ID';
    """
