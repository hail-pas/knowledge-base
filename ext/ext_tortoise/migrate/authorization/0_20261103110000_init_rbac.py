from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    """Create local RBAC tables without creating any user/account table.

    IF NOT EXISTS avoids replacing RBAC tables already present in PostgreSQL. MySQL data must be imported separately.
    """

    return """
        CREATE TABLE IF NOT EXISTS "aerich" (

             "id" SERIAL PRIMARY KEY,
            "version" VARCHAR(255) NOT NULL,
            "app" VARCHAR(100) NOT NULL,
            "content" JSONB NOT NULL

);

        CREATE TABLE IF NOT EXISTS "role" (

             "id" BIGSERIAL PRIMARY KEY,
            "created_at" TIMESTAMP(6) WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
            "updated_at" TIMESTAMP(6) WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
            "deleted_at" BIGINT,
            "label" VARCHAR(50) NOT NULL,
            "remark" VARCHAR(200),
            CONSTRAINT "uid_role_label_47a519" UNIQUE ("label", "deleted_at")
);

        CREATE TABLE IF NOT EXISTS "permission" (

            "code" VARCHAR(256) NOT NULL PRIMARY KEY,
            "label" VARCHAR(128) NOT NULL,
            "permission_type" VARCHAR(16) NOT NULL DEFAULT 'api',
            "is_deprecated" BOOLEAN NOT NULL DEFAULT FALSE

);

        CREATE TABLE IF NOT EXISTS "resource" (

             "id" BIGSERIAL PRIMARY KEY,
            "created_at" TIMESTAMP(6) WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
            "code" VARCHAR(32) NOT NULL,
            "label" VARCHAR(64) NOT NULL,
            "front_route" VARCHAR(128),
            "resource_type" VARCHAR(16) NOT NULL,
            "sub_resource_type" VARCHAR(16) NOT NULL,
            "order_num" INT NOT NULL DEFAULT 1,
            "enabled" BOOLEAN NOT NULL DEFAULT TRUE,
            "assignable" BOOLEAN NOT NULL DEFAULT TRUE,
            "parent_id" BIGINT,
            "scene" VARCHAR(16) NOT NULL,
            CONSTRAINT "uid_resource_code_8b659a" UNIQUE ("code", "parent_id", "scene"),
            CONSTRAINT "fk_resource_resource_61c52602" FOREIGN KEY ("parent_id") REFERENCES "resource" ("id") ON DELETE CASCADE
);

        CREATE TABLE IF NOT EXISTS "resource_permission" (

            "resource_id" BIGINT NOT NULL,
            "permission_id" VARCHAR(256) NOT NULL,
            FOREIGN KEY ("resource_id") REFERENCES "resource" ("id") ON DELETE CASCADE,
            FOREIGN KEY ("permission_id") REFERENCES "permission" ("code") ON DELETE CASCADE,
            CONSTRAINT "uidx_resource_pe_resourc_4b71b7" UNIQUE ("resource_id", "permission_id")
);

        CREATE TABLE IF NOT EXISTS "resource_role" (

            "resource_id" BIGINT NOT NULL,
            "role_id" BIGINT NOT NULL,
            FOREIGN KEY ("resource_id") REFERENCES "resource" ("id") ON DELETE CASCADE,
            FOREIGN KEY ("role_id") REFERENCES "role" ("id") ON DELETE CASCADE,
            CONSTRAINT "uidx_resource_ro_resourc_a19b78" UNIQUE ("resource_id", "role_id")
);
    

CREATE OR REPLACE FUNCTION set_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS "trg_role_updated_at" ON "role";
CREATE TRIGGER "trg_role_updated_at"
BEFORE UPDATE ON "role"
FOR EACH ROW EXECUTE FUNCTION set_updated_at_column();


CREATE INDEX IF NOT EXISTS "idx_role_created_7f5f71" ON "role" ("created_at");
CREATE INDEX IF NOT EXISTS "idx_role_deleted_2bed69" ON "role" ("deleted_at");
CREATE INDEX IF NOT EXISTS "idx_resource_created_669b66" ON "resource" ("created_at");
CREATE INDEX IF NOT EXISTS "idx_resource_code_edb401" ON "resource" ("code");
CREATE INDEX IF NOT EXISTS "idx_resource_label_a90213" ON "resource" ("label");


COMMENT ON COLUMN "role"."id" IS '主键';
COMMENT ON COLUMN "role"."created_at" IS '创建时间';
COMMENT ON COLUMN "role"."updated_at" IS '更新时间';
COMMENT ON COLUMN "role"."deleted_at" IS '删除时间';
COMMENT ON COLUMN "role"."label" IS '角色名称';
COMMENT ON COLUMN "role"."remark" IS '备注';
COMMENT ON TABLE "role" IS '角色';
COMMENT ON COLUMN "permission"."code" IS '权限编码';
COMMENT ON COLUMN "permission"."label" IS '权限名称';
COMMENT ON COLUMN "permission"."permission_type" IS '权限类型';
COMMENT ON COLUMN "permission"."is_deprecated" IS '是否废弃';
COMMENT ON TABLE "permission" IS '权限';
COMMENT ON COLUMN "resource"."id" IS '主键';
COMMENT ON COLUMN "resource"."created_at" IS '创建时间';
COMMENT ON COLUMN "resource"."code" IS '资源编码';
COMMENT ON COLUMN "resource"."label" IS '资源名称';
COMMENT ON COLUMN "resource"."front_route" IS '前端路由';
COMMENT ON COLUMN "resource"."resource_type" IS '资源类型';
COMMENT ON COLUMN "resource"."sub_resource_type" IS '资源子类型';
COMMENT ON COLUMN "resource"."order_num" IS '排列序号';
COMMENT ON COLUMN "resource"."enabled" IS '是否启用';
COMMENT ON COLUMN "resource"."assignable" IS '是否允许分配';
COMMENT ON COLUMN "resource"."parent_id" IS '父级资源';
COMMENT ON COLUMN "resource"."scene" IS '客户端场景';
COMMENT ON TABLE "resource" IS '系统资源';
"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TRIGGER IF EXISTS "trg_role_updated_at" ON "role";
        DROP FUNCTION IF EXISTS set_updated_at_column();
        DROP TABLE IF EXISTS "resource_role";
        DROP TABLE IF EXISTS "resource_permission";
        DROP TABLE IF EXISTS "resource";
        DROP TABLE IF EXISTS "permission";
        DROP TABLE IF EXISTS "role";
    """
