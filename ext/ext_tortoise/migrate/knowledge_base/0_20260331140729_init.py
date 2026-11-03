from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "activity" (

     "id" SERIAL PRIMARY KEY,
    "created_at" TIMESTAMP(6) WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    "workflow_uid" UUID NOT NULL,
    "uid" UUID NOT NULL UNIQUE,
    "name" VARCHAR(200) NOT NULL,
    "input" JSONB NOT NULL,
    "output" JSONB NOT NULL,
    "retry_count" INT NOT NULL DEFAULT 0,
    "execute_params" JSONB NOT NULL,
    "status" VARCHAR(9) NOT NULL DEFAULT 'pending',
    "started_at" TIMESTAMP(6) WITHOUT TIME ZONE,
    "completed_at" TIMESTAMP(6) WITHOUT TIME ZONE,
    "error_message" TEXT,
    "stack_trace" TEXT,
    "canceled_at" TIMESTAMP(6) WITHOUT TIME ZONE,
    "celery_task_id" VARCHAR(255),
            CONSTRAINT "uid_activity_workflo_0e4b51" UNIQUE ("workflow_uid", "name")
);
CREATE TABLE IF NOT EXISTS "embedding_model_config" (

     "id" BIGSERIAL PRIMARY KEY,
    "created_at" TIMESTAMP(6) WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    "updated_at" TIMESTAMP(6) WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    "deleted_at" BIGINT,
    "name" VARCHAR(100) NOT NULL UNIQUE,
    "type" VARCHAR(6) NOT NULL,
    "model_name" VARCHAR(255) NOT NULL,
    "api_key" VARCHAR(500),
    "base_url" VARCHAR(500),
    "dimension" INT NOT NULL,
    "max_chunk_length" INT NOT NULL DEFAULT 8192,
    "batch_size" INT NOT NULL DEFAULT 100,
    "max_retries" INT NOT NULL DEFAULT 3,
    "timeout" INT NOT NULL DEFAULT 60,
    "rate_limit" INT NOT NULL DEFAULT 60,
    "extra_config" JSONB NOT NULL,
    "is_enabled" BOOLEAN NOT NULL DEFAULT TRUE,
    "is_default" BOOLEAN NOT NULL DEFAULT FALSE,
    "description" TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS "collection" (

     "id" BIGSERIAL PRIMARY KEY,
    "created_at" TIMESTAMP(6) WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    "updated_at" TIMESTAMP(6) WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    "deleted_at" BIGINT,
    "name" VARCHAR(100) NOT NULL,
    "description" TEXT,
    "user_id" UUID,
    "tenant_id" UUID,
    "role_id" UUID,
    "is_public" BOOLEAN NOT NULL DEFAULT FALSE,
    "is_temp" BOOLEAN NOT NULL DEFAULT FALSE,
    "is_external" BOOLEAN NOT NULL DEFAULT FALSE,
    "workflow_template" JSONB NOT NULL,
    "external_config" JSONB NOT NULL,
    "embedding_model_config_id" BIGINT,
            CONSTRAINT "uid_collection_user_id_a5fa2b" UNIQUE ("user_id", "name"),
    CONSTRAINT "fk_collecti_embeddin_fe4e9165" FOREIGN KEY ("embedding_model_config_id") REFERENCES "embedding_model_config" ("id") ON DELETE RESTRICT
);
CREATE TABLE IF NOT EXISTS "file_source" (

     "id" BIGSERIAL PRIMARY KEY,
    "created_at" TIMESTAMP(6) WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    "updated_at" TIMESTAMP(6) WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    "deleted_at" BIGINT,
    "name" VARCHAR(100) NOT NULL,
    "type" VARCHAR(10) NOT NULL,
    "access_key" VARCHAR(500),
    "secret_key" VARCHAR(500),
    "storage_location" VARCHAR(1000),
    "endpoint" VARCHAR(500),
    "region" VARCHAR(100),
    "use_ssl" BOOLEAN NOT NULL DEFAULT TRUE,
    "verify_ssl" BOOLEAN NOT NULL DEFAULT TRUE,
    "timeout" INT NOT NULL DEFAULT 30,
    "max_retries" INT NOT NULL DEFAULT 3,
    "concurrent_limit" INT NOT NULL DEFAULT 10,
    "max_connections" INT NOT NULL DEFAULT 100,
    "is_enabled" BOOLEAN NOT NULL DEFAULT TRUE,
    "is_default" BOOLEAN NOT NULL DEFAULT FALSE,
    "user_id" UUID,
    "description" TEXT NOT NULL,
    "extra_config" JSONB NOT NULL,
            CONSTRAINT "uid_file_source_name_80971a" UNIQUE ("name", "deleted_at")
);
CREATE TABLE IF NOT EXISTS "document" (

     "id" BIGSERIAL PRIMARY KEY,
    "created_at" TIMESTAMP(6) WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    "updated_at" TIMESTAMP(6) WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    "deleted_at" BIGINT,
    "uri" VARCHAR(1000) NOT NULL,
    "parsed_uri" VARCHAR(1000),
    "file_name" VARCHAR(255) NOT NULL,
    "display_name" VARCHAR(255) NOT NULL,
    "extension" VARCHAR(50) NOT NULL,
    "file_size" BIGINT,
    "source_last_modified" TIMESTAMP(6) WITHOUT TIME ZONE,
    "source_version_key" VARCHAR(100),
    "is_deleted_in_source" BOOLEAN NOT NULL DEFAULT FALSE,
    "source_meta" JSONB,
    "short_summary" VARCHAR(255),
    "long_summary" TEXT,
    "status" VARCHAR(10) NOT NULL,
    "current_workflow_uid" UUID,
    "config_flag" SMALLINT NOT NULL DEFAULT 0,
    "workflow_template" JSONB NOT NULL,
    "collection_id" BIGINT NOT NULL,
    "file_source_id" BIGINT NOT NULL,
            CONSTRAINT "uid_document_collect_815685" UNIQUE ("collection_id", "uri", "deleted_at"),
            CONSTRAINT "uid_document_collect_d93e8a" UNIQUE ("collection_id", "file_name", "deleted_at"),
    CONSTRAINT "fk_document_collecti_6805aced" FOREIGN KEY ("collection_id") REFERENCES "collection" ("id") ON DELETE CASCADE,
    CONSTRAINT "fk_document_file_sou_77594071" FOREIGN KEY ("file_source_id") REFERENCES "file_source" ("id") ON DELETE RESTRICT
);
CREATE TABLE IF NOT EXISTS "document_chunk" (

     "id" BIGSERIAL PRIMARY KEY,
    "created_at" TIMESTAMP(6) WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    "updated_at" TIMESTAMP(6) WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    "deleted_at" BIGINT,
    "content" TEXT NOT NULL,
    "pages" JSONB NOT NULL,
    "min_page" SMALLINT NOT NULL,
    "max_page" SMALLINT NOT NULL,
    "start" JSONB NOT NULL,
    "end" JSONB NOT NULL,
    "overlap_start" JSONB,
    "overlap_end" JSONB,
    "metadata" JSONB NOT NULL,
    "manual_add" BOOLEAN NOT NULL DEFAULT FALSE,
    "document_id" BIGINT NOT NULL,
    CONSTRAINT "fk_document_document_c3048c62" FOREIGN KEY ("document_id") REFERENCES "document" ("id") ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS "document_gfaq" (

     "id" BIGSERIAL PRIMARY KEY,
    "created_at" TIMESTAMP(6) WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    "updated_at" TIMESTAMP(6) WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    "deleted_at" BIGINT,
    "content" TEXT,
    "question" TEXT,
    "answer" TEXT,
    "manual_add" BOOLEAN NOT NULL DEFAULT FALSE,
    "enabled" BOOLEAN NOT NULL DEFAULT TRUE,
    "document_id" BIGINT NOT NULL,
    CONSTRAINT "fk_document_document_6ae6cb1d" FOREIGN KEY ("document_id") REFERENCES "document" ("id") ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS "document_page" (

     "id" BIGSERIAL PRIMARY KEY,
    "created_at" TIMESTAMP(6) WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    "updated_at" TIMESTAMP(6) WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    "deleted_at" BIGINT,
    "page_number" SMALLINT NOT NULL,
    "content" TEXT NOT NULL,
    "tables" JSONB NOT NULL,
    "images" JSONB NOT NULL,
    "metadata" JSONB NOT NULL,
    "document_id" BIGINT NOT NULL,
    CONSTRAINT "fk_document_document_0e602c07" FOREIGN KEY ("document_id") REFERENCES "document" ("id") ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS "indexing_backend_config" (

     "id" BIGSERIAL PRIMARY KEY,
    "created_at" TIMESTAMP(6) WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    "updated_at" TIMESTAMP(6) WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    "deleted_at" BIGINT,
    "name" VARCHAR(100) NOT NULL UNIQUE,
    "type" VARCHAR(13) NOT NULL,
    "host" VARCHAR(255),
    "port" INT,
    "username" VARCHAR(255),
    "password" VARCHAR(500),
    "use_ssl" BOOLEAN NOT NULL DEFAULT TRUE,
    "verify_ssl" BOOLEAN NOT NULL DEFAULT TRUE,
    "timeout" INT NOT NULL DEFAULT 30,
    "max_retries" INT NOT NULL DEFAULT 3,
    "max_connections" INT NOT NULL DEFAULT 100,
    "extra_config" JSONB NOT NULL,
    "is_enabled" BOOLEAN NOT NULL DEFAULT TRUE,
    "is_default" BOOLEAN NOT NULL DEFAULT FALSE,
    "description" TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS "llm_model_config" (

     "id" BIGSERIAL PRIMARY KEY,
    "created_at" TIMESTAMP(6) WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    "updated_at" TIMESTAMP(6) WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    "deleted_at" BIGINT,
    "name" VARCHAR(100) NOT NULL UNIQUE,
    "type" VARCHAR(12) NOT NULL,
    "model_name" VARCHAR(255) NOT NULL,
    "api_key" VARCHAR(500),
    "base_url" VARCHAR(500),
    "max_tokens" INT NOT NULL DEFAULT 4096,
    "temperature" DOUBLE PRECISION PRECISION NOT NULL DEFAULT 0.7,
    "top_p" DOUBLE PRECISION PRECISION NOT NULL DEFAULT 1,
    "presence_penalty" DOUBLE PRECISION PRECISION,
    "frequency_penalty" DOUBLE PRECISION PRECISION,
    "seed" INT,
    "timeout" INT NOT NULL DEFAULT 60,
    "parallel_tool_calls" BOOLEAN,
    "supports_tools" BOOLEAN NOT NULL DEFAULT FALSE,
    "supports_image_output" BOOLEAN NOT NULL DEFAULT FALSE,
    "supports_json_schema_output" BOOLEAN NOT NULL DEFAULT FALSE,
    "supports_json_object_output" BOOLEAN NOT NULL DEFAULT FALSE,
    "default_structured_output_mode" VARCHAR(16) NOT NULL DEFAULT 'tool',
    "native_output_requires_schema_in_instructions" BOOLEAN NOT NULL DEFAULT FALSE,
    "extra_config" JSONB NOT NULL,
    "is_enabled" BOOLEAN NOT NULL DEFAULT TRUE,
    "is_default" BOOLEAN NOT NULL DEFAULT FALSE,
    "description" TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS "workflow" (

     "id" BIGSERIAL PRIMARY KEY,
    "created_at" TIMESTAMP(6) WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    "updated_at" TIMESTAMP(6) WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    "deleted_at" BIGINT,
    "uid" UUID NOT NULL UNIQUE,
    "config" JSONB NOT NULL,
    "config_format" VARCHAR(4) NOT NULL DEFAULT 'yaml',
    "status" VARCHAR(9) NOT NULL DEFAULT 'pending',
    "started_at" TIMESTAMP(6) WITHOUT TIME ZONE,
    "completed_at" TIMESTAMP(6) WITHOUT TIME ZONE,
    "canceled_at" TIMESTAMP(6) WITHOUT TIME ZONE
);

CREATE OR REPLACE FUNCTION set_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS "trg_embedding_model_config_updated_at" ON "embedding_model_config";
CREATE TRIGGER "trg_embedding_model_config_updated_at"
BEFORE UPDATE ON "embedding_model_config"
FOR EACH ROW EXECUTE FUNCTION set_updated_at_column();

DROP TRIGGER IF EXISTS "trg_collection_updated_at" ON "collection";
CREATE TRIGGER "trg_collection_updated_at"
BEFORE UPDATE ON "collection"
FOR EACH ROW EXECUTE FUNCTION set_updated_at_column();

DROP TRIGGER IF EXISTS "trg_file_source_updated_at" ON "file_source";
CREATE TRIGGER "trg_file_source_updated_at"
BEFORE UPDATE ON "file_source"
FOR EACH ROW EXECUTE FUNCTION set_updated_at_column();

DROP TRIGGER IF EXISTS "trg_document_updated_at" ON "document";
CREATE TRIGGER "trg_document_updated_at"
BEFORE UPDATE ON "document"
FOR EACH ROW EXECUTE FUNCTION set_updated_at_column();

DROP TRIGGER IF EXISTS "trg_document_chunk_updated_at" ON "document_chunk";
CREATE TRIGGER "trg_document_chunk_updated_at"
BEFORE UPDATE ON "document_chunk"
FOR EACH ROW EXECUTE FUNCTION set_updated_at_column();

DROP TRIGGER IF EXISTS "trg_document_gfaq_updated_at" ON "document_gfaq";
CREATE TRIGGER "trg_document_gfaq_updated_at"
BEFORE UPDATE ON "document_gfaq"
FOR EACH ROW EXECUTE FUNCTION set_updated_at_column();

DROP TRIGGER IF EXISTS "trg_document_page_updated_at" ON "document_page";
CREATE TRIGGER "trg_document_page_updated_at"
BEFORE UPDATE ON "document_page"
FOR EACH ROW EXECUTE FUNCTION set_updated_at_column();

DROP TRIGGER IF EXISTS "trg_indexing_backend_config_updated_at" ON "indexing_backend_config";
CREATE TRIGGER "trg_indexing_backend_config_updated_at"
BEFORE UPDATE ON "indexing_backend_config"
FOR EACH ROW EXECUTE FUNCTION set_updated_at_column();

DROP TRIGGER IF EXISTS "trg_llm_model_config_updated_at" ON "llm_model_config";
CREATE TRIGGER "trg_llm_model_config_updated_at"
BEFORE UPDATE ON "llm_model_config"
FOR EACH ROW EXECUTE FUNCTION set_updated_at_column();

DROP TRIGGER IF EXISTS "trg_workflow_updated_at" ON "workflow";
CREATE TRIGGER "trg_workflow_updated_at"
BEFORE UPDATE ON "workflow"
FOR EACH ROW EXECUTE FUNCTION set_updated_at_column();


CREATE INDEX IF NOT EXISTS "idx_activity_created_5a6493" ON "activity" ("created_at");
CREATE INDEX IF NOT EXISTS "idx_activity_workflo_fd04b5" ON "activity" ("workflow_uid");
CREATE INDEX IF NOT EXISTS "idx_activity_status_6357fc" ON "activity" ("status");
CREATE INDEX IF NOT EXISTS "idx_activity_celery__4333c8" ON "activity" ("celery_task_id");
CREATE INDEX IF NOT EXISTS "idx_embedding_m_created_9e170e" ON "embedding_model_config" ("created_at");
CREATE INDEX IF NOT EXISTS "idx_embedding_m_deleted_496ead" ON "embedding_model_config" ("deleted_at");
CREATE INDEX IF NOT EXISTS "idx_embedding_m_is_enab_d158c4" ON "embedding_model_config" ("is_enabled");
CREATE INDEX IF NOT EXISTS "idx_embedding_m_is_defa_d46056" ON "embedding_model_config" ("is_default");
CREATE INDEX IF NOT EXISTS "idx_embedding_m_type_491b8f" ON "embedding_model_config" ("type");
CREATE INDEX IF NOT EXISTS "idx_collection_created_385054" ON "collection" ("created_at");
CREATE INDEX IF NOT EXISTS "idx_collection_deleted_d4b7b6" ON "collection" ("deleted_at");
CREATE INDEX IF NOT EXISTS "idx_file_source_created_05ed79" ON "file_source" ("created_at");
CREATE INDEX IF NOT EXISTS "idx_file_source_deleted_a0dbf6" ON "file_source" ("deleted_at");
CREATE INDEX IF NOT EXISTS "idx_file_source_type_0a0019" ON "file_source" ("type", "is_enabled");
CREATE INDEX IF NOT EXISTS "idx_file_source_is_defa_df421e" ON "file_source" ("is_default", "is_enabled");
CREATE INDEX IF NOT EXISTS "idx_document_created_b6cfaa" ON "document" ("created_at");
CREATE INDEX IF NOT EXISTS "idx_document_deleted_b64b7d" ON "document" ("deleted_at");
CREATE INDEX IF NOT EXISTS "idx_document_collect_2893ec" ON "document" ("collection_id");
CREATE INDEX IF NOT EXISTS "idx_document_collect_e0dffb" ON "document" ("collection_id", "status");
CREATE INDEX IF NOT EXISTS "idx_document_collect_34bbbd" ON "document" ("collection_id", "extension");
CREATE INDEX IF NOT EXISTS "idx_document_file_so_b54b49" ON "document" ("file_source_id");
CREATE INDEX IF NOT EXISTS "idx_document_ch_created_832cef" ON "document_chunk" ("created_at");
CREATE INDEX IF NOT EXISTS "idx_document_ch_deleted_861940" ON "document_chunk" ("deleted_at");
CREATE INDEX IF NOT EXISTS "idx_document_ch_documen_6d4d6c" ON "document_chunk" ("document_id");
CREATE INDEX IF NOT EXISTS "idx_document_ch_documen_830978" ON "document_chunk" ("document_id", "min_page");
CREATE INDEX IF NOT EXISTS "idx_document_ch_documen_95700d" ON "document_chunk" ("document_id", "max_page");
CREATE INDEX IF NOT EXISTS "idx_document_gf_created_42c227" ON "document_gfaq" ("created_at");
CREATE INDEX IF NOT EXISTS "idx_document_gf_deleted_8441f3" ON "document_gfaq" ("deleted_at");
CREATE INDEX IF NOT EXISTS "idx_document_gf_documen_813344" ON "document_gfaq" ("document_id");
CREATE INDEX IF NOT EXISTS "idx_document_gf_documen_64cc9a" ON "document_gfaq" ("document_id", "enabled");
CREATE INDEX IF NOT EXISTS "idx_document_pa_created_aa6a2c" ON "document_page" ("created_at");
CREATE INDEX IF NOT EXISTS "idx_document_pa_deleted_b9b7ee" ON "document_page" ("deleted_at");
CREATE INDEX IF NOT EXISTS "idx_document_pa_documen_0755d0" ON "document_page" ("document_id");
CREATE INDEX IF NOT EXISTS "idx_document_pa_documen_5fe339" ON "document_page" ("document_id", "page_number");
CREATE INDEX IF NOT EXISTS "idx_indexing_ba_created_840754" ON "indexing_backend_config" ("created_at");
CREATE INDEX IF NOT EXISTS "idx_indexing_ba_deleted_b9dd20" ON "indexing_backend_config" ("deleted_at");
CREATE INDEX IF NOT EXISTS "idx_indexing_ba_type_0a4885" ON "indexing_backend_config" ("type", "is_enabled");
CREATE INDEX IF NOT EXISTS "idx_indexing_ba_is_defa_73e737" ON "indexing_backend_config" ("is_default", "is_enabled");
CREATE INDEX IF NOT EXISTS "idx_llm_model_c_created_483236" ON "llm_model_config" ("created_at");
CREATE INDEX IF NOT EXISTS "idx_llm_model_c_deleted_25ea9c" ON "llm_model_config" ("deleted_at");
CREATE INDEX IF NOT EXISTS "idx_llm_model_c_is_enab_578b90" ON "llm_model_config" ("is_enabled");
CREATE INDEX IF NOT EXISTS "idx_llm_model_c_is_defa_011321" ON "llm_model_config" ("is_default");
CREATE INDEX IF NOT EXISTS "idx_llm_model_c_type_ab2c6f" ON "llm_model_config" ("type");
CREATE INDEX IF NOT EXISTS "idx_workflow_created_c2a8eb" ON "workflow" ("created_at");
CREATE INDEX IF NOT EXISTS "idx_workflow_deleted_d3d134" ON "workflow" ("deleted_at");


COMMENT ON COLUMN "activity"."created_at" IS '创建时间';
COMMENT ON COLUMN "activity"."workflow_uid" IS '工作流UID';
COMMENT ON COLUMN "activity"."name" IS '名称';
COMMENT ON COLUMN "activity"."input" IS '输入参数';
COMMENT ON COLUMN "activity"."output" IS '输出参数';
COMMENT ON COLUMN "activity"."retry_count" IS '失败重试次数';
COMMENT ON COLUMN "activity"."execute_params" IS '额外的 Celery 执行参数，覆盖默认值';
COMMENT ON COLUMN "activity"."status" IS '状态';
COMMENT ON COLUMN "activity"."started_at" IS '开始执行时间';
COMMENT ON COLUMN "activity"."completed_at" IS '完成时间';
COMMENT ON COLUMN "activity"."error_message" IS '错误信息';
COMMENT ON COLUMN "activity"."stack_trace" IS '错误堆栈跟踪';
COMMENT ON COLUMN "activity"."canceled_at" IS '取消时间';
COMMENT ON COLUMN "activity"."celery_task_id" IS 'Celery 任务ID, direct执行模式下为空';
COMMENT ON TABLE "activity" IS '工作流活动表';
COMMENT ON COLUMN "embedding_model_config"."id" IS '主键';
COMMENT ON COLUMN "embedding_model_config"."created_at" IS '创建时间';
COMMENT ON COLUMN "embedding_model_config"."updated_at" IS '更新时间';
COMMENT ON COLUMN "embedding_model_config"."deleted_at" IS '删除时间';
COMMENT ON COLUMN "embedding_model_config"."name" IS '配置名称';
COMMENT ON COLUMN "embedding_model_config"."type" IS '模型类型';
COMMENT ON COLUMN "embedding_model_config"."model_name" IS '模型标识符';
COMMENT ON COLUMN "embedding_model_config"."api_key" IS 'API密钥';
COMMENT ON COLUMN "embedding_model_config"."base_url" IS 'API基础URL';
COMMENT ON COLUMN "embedding_model_config"."dimension" IS '向量维度';
COMMENT ON COLUMN "embedding_model_config"."max_chunk_length" IS '单条chunk最大长度';
COMMENT ON COLUMN "embedding_model_config"."batch_size" IS '批处理大小';
COMMENT ON COLUMN "embedding_model_config"."max_retries" IS '最大重试次数';
COMMENT ON COLUMN "embedding_model_config"."timeout" IS '请求超时时间(秒)';
COMMENT ON COLUMN "embedding_model_config"."rate_limit" IS '每分钟最大请求数(0=无限制)';
COMMENT ON COLUMN "embedding_model_config"."extra_config" IS 'provider特定扩展配置';
COMMENT ON COLUMN "embedding_model_config"."is_enabled" IS '是否启用';
COMMENT ON COLUMN "embedding_model_config"."is_default" IS '是否默认配置';
COMMENT ON COLUMN "embedding_model_config"."description" IS '描述信息';
COMMENT ON TABLE "embedding_model_config" IS 'Embedding模型配置表';
COMMENT ON COLUMN "collection"."id" IS '主键';
COMMENT ON COLUMN "collection"."created_at" IS '创建时间';
COMMENT ON COLUMN "collection"."updated_at" IS '更新时间';
COMMENT ON COLUMN "collection"."deleted_at" IS '删除时间';
COMMENT ON COLUMN "collection"."name" IS '集合名称';
COMMENT ON COLUMN "collection"."description" IS '描述';
COMMENT ON COLUMN "collection"."user_id" IS '用户ID';
COMMENT ON COLUMN "collection"."tenant_id" IS '租户ID';
COMMENT ON COLUMN "collection"."role_id" IS '角色ID';
COMMENT ON COLUMN "collection"."is_public" IS '是否公开';
COMMENT ON COLUMN "collection"."is_temp" IS '是否临时';
COMMENT ON COLUMN "collection"."is_external" IS '是否外部';
COMMENT ON COLUMN "collection"."workflow_template" IS '默认DAG工作流模版';
COMMENT ON COLUMN "collection"."external_config" IS '外部知识库配置, 占位后续扩展';
COMMENT ON COLUMN "collection"."embedding_model_config_id" IS '关联嵌入模型';
COMMENT ON TABLE "collection" IS '文件集合表';
COMMENT ON COLUMN "file_source"."id" IS '主键';
COMMENT ON COLUMN "file_source"."created_at" IS '创建时间';
COMMENT ON COLUMN "file_source"."updated_at" IS '更新时间';
COMMENT ON COLUMN "file_source"."deleted_at" IS '删除时间';
COMMENT ON COLUMN "file_source"."name" IS '文件源名称';
COMMENT ON COLUMN "file_source"."type" IS '文件源类型';
COMMENT ON COLUMN "file_source"."access_key" IS '访问密钥';
COMMENT ON COLUMN "file_source"."secret_key" IS '密钥（加密存储）';
COMMENT ON COLUMN "file_source"."storage_location" IS '存储位置  - type=local_file: 本地文件路径（如 /data/documents）  - type=s3/minio/aliyun_oss: 存储桶名称（如 my-bucket）, sharepoint: 站点路径, api: API路径';
COMMENT ON COLUMN "file_source"."endpoint" IS '服务端点URL';
COMMENT ON COLUMN "file_source"."region" IS '区域/地域';
COMMENT ON COLUMN "file_source"."use_ssl" IS '是否使用SSL/TLS';
COMMENT ON COLUMN "file_source"."verify_ssl" IS '是否验证SSL证书';
COMMENT ON COLUMN "file_source"."timeout" IS '连接/读取超时时间(秒)';
COMMENT ON COLUMN "file_source"."max_retries" IS '最大重试次数';
COMMENT ON COLUMN "file_source"."concurrent_limit" IS '并发限制';
COMMENT ON COLUMN "file_source"."max_connections" IS '最大连接数(连接池)';
COMMENT ON COLUMN "file_source"."is_enabled" IS '是否启用';
COMMENT ON COLUMN "file_source"."is_default" IS '是否默认';
COMMENT ON COLUMN "file_source"."user_id" IS '用户ID, 为空时表示公共';
COMMENT ON COLUMN "file_source"."description" IS '描述信息';
COMMENT ON COLUMN "file_source"."extra_config" IS 'Provider特定扩展配置（JSON格式）';
COMMENT ON TABLE "file_source" IS '文件源配置表';
COMMENT ON COLUMN "document"."id" IS '主键';
COMMENT ON COLUMN "document"."created_at" IS '创建时间';
COMMENT ON COLUMN "document"."updated_at" IS '更新时间';
COMMENT ON COLUMN "document"."deleted_at" IS '删除时间';
COMMENT ON COLUMN "document"."uri" IS '文件唯一标识（本地为绝对路径）';
COMMENT ON COLUMN "document"."parsed_uri" IS '解析后的文件唯一标识';
COMMENT ON COLUMN "document"."file_name" IS '文件名';
COMMENT ON COLUMN "document"."display_name" IS '显示名称';
COMMENT ON COLUMN "document"."extension" IS '文件扩展名';
COMMENT ON COLUMN "document"."file_size" IS '文件大小(字节)';
COMMENT ON COLUMN "document"."source_last_modified" IS '文件源最后修改时间';
COMMENT ON COLUMN "document"."source_version_key" IS '文件源版本标识';
COMMENT ON COLUMN "document"."is_deleted_in_source" IS '文件是否在源中被删除';
COMMENT ON COLUMN "document"."source_meta" IS '文件源元数据（JSON格式）';
COMMENT ON COLUMN "document"."short_summary" IS '文件摘要';
COMMENT ON COLUMN "document"."long_summary" IS '文件详细摘要';
COMMENT ON COLUMN "document"."status" IS '文件状态';
COMMENT ON COLUMN "document"."current_workflow_uid" IS '当前关联的最新工作流UID';
COMMENT ON COLUMN "document"."config_flag" IS '配置标志';
COMMENT ON COLUMN "document"."workflow_template" IS 'DAG工作流模版';
COMMENT ON COLUMN "document"."collection_id" IS '关联集合';
COMMENT ON COLUMN "document"."file_source_id" IS '关联文件源';
COMMENT ON TABLE "document" IS '文件记录表';
COMMENT ON COLUMN "document_chunk"."id" IS '主键';
COMMENT ON COLUMN "document_chunk"."created_at" IS '创建时间';
COMMENT ON COLUMN "document_chunk"."updated_at" IS '更新时间';
COMMENT ON COLUMN "document_chunk"."deleted_at" IS '删除时间';
COMMENT ON COLUMN "document_chunk"."content" IS '切块内容';
COMMENT ON COLUMN "document_chunk"."pages" IS '切块页码列表';
COMMENT ON COLUMN "document_chunk"."min_page" IS '最小页码';
COMMENT ON COLUMN "document_chunk"."max_page" IS '最大页码';
COMMENT ON COLUMN "document_chunk"."start" IS '起始位置（页码+页内偏移）';
COMMENT ON COLUMN "document_chunk"."end" IS '结束位置（页码+页内偏移）';
COMMENT ON COLUMN "document_chunk"."overlap_start" IS '重叠起始位置（页码+页内偏移）';
COMMENT ON COLUMN "document_chunk"."overlap_end" IS '重叠结束位置（页码+页内偏移）';
COMMENT ON COLUMN "document_chunk"."metadata" IS '元数据';
COMMENT ON COLUMN "document_chunk"."manual_add" IS '是否手动添加';
COMMENT ON COLUMN "document_chunk"."document_id" IS '文档ID';
COMMENT ON TABLE "document_chunk" IS '文档切块记录表';
COMMENT ON COLUMN "document_gfaq"."id" IS '主键';
COMMENT ON COLUMN "document_gfaq"."created_at" IS '创建时间';
COMMENT ON COLUMN "document_gfaq"."updated_at" IS '更新时间';
COMMENT ON COLUMN "document_gfaq"."deleted_at" IS '删除时间';
COMMENT ON COLUMN "document_gfaq"."content" IS '相关文档内容块';
COMMENT ON COLUMN "document_gfaq"."question" IS '问题';
COMMENT ON COLUMN "document_gfaq"."answer" IS '答案';
COMMENT ON COLUMN "document_gfaq"."manual_add" IS '是否手动添加';
COMMENT ON COLUMN "document_gfaq"."enabled" IS '是否启用';
COMMENT ON COLUMN "document_gfaq"."document_id" IS '文档';
COMMENT ON TABLE "document_gfaq" IS '文档生成FAQ表';
COMMENT ON COLUMN "document_page"."id" IS '主键';
COMMENT ON COLUMN "document_page"."created_at" IS '创建时间';
COMMENT ON COLUMN "document_page"."updated_at" IS '更新时间';
COMMENT ON COLUMN "document_page"."deleted_at" IS '删除时间';
COMMENT ON COLUMN "document_page"."page_number" IS '页码';
COMMENT ON COLUMN "document_page"."content" IS '页面内容，当只有一页的时候content太长可以不存，但是必须要有page记录，content可以从parsed_uri获取';
COMMENT ON COLUMN "document_page"."tables" IS '页面表格信息';
COMMENT ON COLUMN "document_page"."images" IS '页面图片信息, list[keys]';
COMMENT ON COLUMN "document_page"."metadata" IS '页面元数据';
COMMENT ON COLUMN "document_page"."document_id" IS '文档ID';
COMMENT ON COLUMN "indexing_backend_config"."id" IS '主键';
COMMENT ON COLUMN "indexing_backend_config"."created_at" IS '创建时间';
COMMENT ON COLUMN "indexing_backend_config"."updated_at" IS '更新时间';
COMMENT ON COLUMN "indexing_backend_config"."deleted_at" IS '删除时间';
COMMENT ON COLUMN "indexing_backend_config"."name" IS '配置名称';
COMMENT ON COLUMN "indexing_backend_config"."type" IS '后端类型';
COMMENT ON COLUMN "indexing_backend_config"."host" IS '主机地址';
COMMENT ON COLUMN "indexing_backend_config"."port" IS '端口号';
COMMENT ON COLUMN "indexing_backend_config"."username" IS '用户名';
COMMENT ON COLUMN "indexing_backend_config"."password" IS '密码（加密）';
COMMENT ON COLUMN "indexing_backend_config"."use_ssl" IS '是否使用SSL/TLS';
COMMENT ON COLUMN "indexing_backend_config"."verify_ssl" IS '是否验证SSL证书';
COMMENT ON COLUMN "indexing_backend_config"."timeout" IS '连接超时时间(秒)';
COMMENT ON COLUMN "indexing_backend_config"."max_retries" IS '最大重试次数';
COMMENT ON COLUMN "indexing_backend_config"."max_connections" IS '最大连接数(连接池)';
COMMENT ON COLUMN "indexing_backend_config"."extra_config" IS 'Provider特定扩展配置';
COMMENT ON COLUMN "indexing_backend_config"."is_enabled" IS '是否启用';
COMMENT ON COLUMN "indexing_backend_config"."is_default" IS '是否默认配置';
COMMENT ON COLUMN "indexing_backend_config"."description" IS '描述信息';
COMMENT ON TABLE "indexing_backend_config" IS '索引后端配置表';
COMMENT ON COLUMN "llm_model_config"."id" IS '主键';
COMMENT ON COLUMN "llm_model_config"."created_at" IS '创建时间';
COMMENT ON COLUMN "llm_model_config"."updated_at" IS '更新时间';
COMMENT ON COLUMN "llm_model_config"."deleted_at" IS '删除时间';
COMMENT ON COLUMN "llm_model_config"."name" IS '配置名称';
COMMENT ON COLUMN "llm_model_config"."type" IS '模型类型';
COMMENT ON COLUMN "llm_model_config"."model_name" IS '模型标识符';
COMMENT ON COLUMN "llm_model_config"."api_key" IS 'API密钥（加密）';
COMMENT ON COLUMN "llm_model_config"."base_url" IS 'API基础URL';
COMMENT ON COLUMN "llm_model_config"."max_tokens" IS '默认最大输出 token 数';
COMMENT ON COLUMN "llm_model_config"."temperature" IS '默认 temperature';
COMMENT ON COLUMN "llm_model_config"."top_p" IS '默认 top_p';
COMMENT ON COLUMN "llm_model_config"."presence_penalty" IS '默认 presence_penalty';
COMMENT ON COLUMN "llm_model_config"."frequency_penalty" IS '默认 frequency_penalty';
COMMENT ON COLUMN "llm_model_config"."seed" IS '默认 seed';
COMMENT ON COLUMN "llm_model_config"."timeout" IS '默认请求超时时间(秒)';
COMMENT ON COLUMN "llm_model_config"."parallel_tool_calls" IS '默认 parallel_tool_calls';
COMMENT ON COLUMN "llm_model_config"."supports_tools" IS '是否支持 tools';
COMMENT ON COLUMN "llm_model_config"."supports_image_output" IS '是否支持图片输出';
COMMENT ON COLUMN "llm_model_config"."supports_json_schema_output" IS '是否支持 JSON Schema 输出';
COMMENT ON COLUMN "llm_model_config"."supports_json_object_output" IS '是否支持 JSON Object 输出';
COMMENT ON COLUMN "llm_model_config"."default_structured_output_mode" IS '默认结构化输出模式(tool/native/prompted)';
COMMENT ON COLUMN "llm_model_config"."native_output_requires_schema_in_instructions" IS 'native 输出模式是否要求把 schema 写入 instructions';
COMMENT ON COLUMN "llm_model_config"."extra_config" IS 'provider特定扩展配置';
COMMENT ON COLUMN "llm_model_config"."is_enabled" IS '是否启用';
COMMENT ON COLUMN "llm_model_config"."is_default" IS '是否默认配置';
COMMENT ON COLUMN "llm_model_config"."description" IS '描述信息';
COMMENT ON TABLE "llm_model_config" IS 'LLM模型配置表';
COMMENT ON COLUMN "workflow"."id" IS '主键';
COMMENT ON COLUMN "workflow"."created_at" IS '创建时间';
COMMENT ON COLUMN "workflow"."updated_at" IS '更新时间';
COMMENT ON COLUMN "workflow"."deleted_at" IS '删除时间';
COMMENT ON COLUMN "workflow"."uid" IS '幂等性唯一标识';
COMMENT ON COLUMN "workflow"."config" IS '工作流配置（DAG图结构，支持YAML/JSON格式）, {content: vv}';
COMMENT ON COLUMN "workflow"."config_format" IS '配置格式（yaml/json/python）';
COMMENT ON COLUMN "workflow"."status" IS '工作流状态';
COMMENT ON COLUMN "workflow"."started_at" IS '开始执行时间';
COMMENT ON COLUMN "workflow"."completed_at" IS '完成时间';
COMMENT ON COLUMN "workflow"."canceled_at" IS '是否被取消';
COMMENT ON TABLE "workflow" IS '工作流定义表';
"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        """
