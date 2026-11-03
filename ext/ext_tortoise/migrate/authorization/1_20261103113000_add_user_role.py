from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "role"
            ADD COLUMN "is_super_admin" BOOLEAN NOT NULL DEFAULT FALSE;

        COMMENT ON COLUMN "role"."is_super_admin" IS '是否为系统超级管理员';

        CREATE TABLE IF NOT EXISTS "user_role" (

             "id" BIGSERIAL PRIMARY KEY,
            "created_at" TIMESTAMP(6) WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
            "updated_at" TIMESTAMP(6) WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
            "deleted_at" BIGINT,
            "user_id" UUID NOT NULL,
            "role_id" BIGINT NOT NULL,
            CONSTRAINT "fk_user_role_role" FOREIGN KEY ("role_id") REFERENCES "role" ("id") ON DELETE CASCADE,
            CONSTRAINT "uid_user_role_user_role_deleted" UNIQUE ("user_id", "role_id", "deleted_at")
);
    

DROP TRIGGER IF EXISTS "trg_user_role_updated_at" ON "user_role";
CREATE TRIGGER "trg_user_role_updated_at"
BEFORE UPDATE ON "user_role"
FOR EACH ROW EXECUTE FUNCTION set_updated_at_column();


CREATE INDEX IF NOT EXISTS "idx_user_role_user_id" ON "user_role" ("user_id");
CREATE INDEX IF NOT EXISTS "idx_user_role_created_at" ON "user_role" ("created_at");


COMMENT ON COLUMN "user_role"."id" IS '主键';
COMMENT ON COLUMN "user_role"."created_at" IS '创建时间';
COMMENT ON COLUMN "user_role"."updated_at" IS '更新时间';
COMMENT ON COLUMN "user_role"."deleted_at" IS '删除时间';
COMMENT ON COLUMN "user_role"."user_id" IS 'Pero 用户 UUID';
COMMENT ON COLUMN "user_role"."role_id" IS '本地角色ID';
COMMENT ON TABLE "user_role" IS 'Pero 用户与本地角色关联';
"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TRIGGER IF EXISTS "trg_user_role_updated_at" ON "user_role";
        DROP TABLE IF EXISTS "user_role";
        ALTER TABLE "role" DROP COLUMN "is_super_admin";
    """
