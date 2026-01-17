from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS `activity` (
    `id` INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    `created_at` DATETIME(6) NOT NULL COMMENT '创建时间' DEFAULT CURRENT_TIMESTAMP(6),
    `workflow_uid` CHAR(36) NOT NULL COMMENT '工作流UID',
    `uid` CHAR(36) NOT NULL UNIQUE,
    `name` VARCHAR(200) NOT NULL COMMENT '名称',
    `input` JSON NOT NULL COMMENT '输入参数',
    `output` JSON NOT NULL COMMENT '输出参数',
    `retry_count` INT NOT NULL COMMENT '失败重试次数' DEFAULT 0,
    `execute_params` JSON NOT NULL COMMENT 'Celery 执行参数',
    `status` VARCHAR(9) NOT NULL COMMENT '状态' DEFAULT 'pending',
    `started_at` DATETIME(6) COMMENT '开始执行时间',
    `completed_at` DATETIME(6) COMMENT '完成时间',
    `error_message` LONGTEXT COMMENT '错误信息',
    `stack_trace` LONGTEXT COMMENT '错误堆栈跟踪',
    `canceled_at` DATETIME(6) COMMENT '取消时间',
    `celery_task_id` VARCHAR(255) COMMENT 'Celery 任务ID',
    UNIQUE KEY `uid_activity_workflo_0e4b51` (`workflow_uid`, `name`),
    KEY `idx_activity_created_5a6493` (`created_at`),
    KEY `idx_activity_workflo_fd04b5` (`workflow_uid`),
    KEY `idx_activity_status_6357fc` (`status`),
    KEY `idx_activity_celery__4333c8` (`celery_task_id`)
) CHARACTER SET utf8mb4 COMMENT='工作流活动表';
CREATE TABLE IF NOT EXISTS `collection` (
    `id` BIGINT NOT NULL PRIMARY KEY AUTO_INCREMENT COMMENT '主键',
    `created_at` DATETIME(6) NOT NULL COMMENT '创建时间' DEFAULT CURRENT_TIMESTAMP(6),
    `updated_at` DATETIME(6) NOT NULL COMMENT '更新时间' DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    `deleted_at` BIGINT COMMENT '删除时间',
    `name` VARCHAR(100) NOT NULL COMMENT '集合名称',
    `description` LONGTEXT COMMENT '描述',
    `user_id` CHAR(36) COMMENT '用户ID',
    `tenant_id` CHAR(36) COMMENT '租户ID',
    `role_id` CHAR(36) COMMENT '角色ID',
    `is_public` BOOL NOT NULL COMMENT '是否公开' DEFAULT 0,
    `is_temp` BOOL NOT NULL COMMENT '是否临时' DEFAULT 0,
    UNIQUE KEY `uid_collection_user_id_a5fa2b` (`user_id`, `name`),
    KEY `idx_collection_created_385054` (`created_at`),
    KEY `idx_collection_deleted_d4b7b6` (`deleted_at`)
) CHARACTER SET utf8mb4 COMMENT='文件集合表';
CREATE TABLE IF NOT EXISTS `embedding_model_config` (
    `id` BIGINT NOT NULL PRIMARY KEY AUTO_INCREMENT COMMENT '主键',
    `created_at` DATETIME(6) NOT NULL COMMENT '创建时间' DEFAULT CURRENT_TIMESTAMP(6),
    `updated_at` DATETIME(6) NOT NULL COMMENT '更新时间' DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    `deleted_at` BIGINT COMMENT '删除时间',
    `name` VARCHAR(100) NOT NULL COMMENT '模型配置名称',
    `type` VARCHAR(6) NOT NULL COMMENT '模型类型（openai/sentence_transformers等）',
    `model_name_or_path` VARCHAR(255) NOT NULL COMMENT '模型标识符或路径（如 text-embedding-3-small, sentence-transformers/...）',
    `config` JSON NOT NULL COMMENT '模型配置参数（JSON格式，如 api_key、endpoint 等）',
    `dimension` INT COMMENT '向量维度',
    `max_batch_size` INT NOT NULL COMMENT '最大批处理大小，避免文本过大导致服务端无法处理' DEFAULT 32,
    `max_token_per_request` INT NOT NULL COMMENT '单次请求最大 token 数' DEFAULT 8191,
    `max_token_per_text` INT NOT NULL COMMENT '单个文本的最大 token 长度' DEFAULT 512,
    `is_enabled` BOOL NOT NULL COMMENT '是否启用' DEFAULT 1,
    `is_default` BOOL NOT NULL COMMENT '是否默认模型' DEFAULT 0,
    `description` LONGTEXT NOT NULL COMMENT '描述信息',
    UNIQUE KEY `uid_embedding_m_model_n_12ca67` (`model_name_or_path`, `deleted_at`),
    KEY `idx_embedding_m_created_9e170e` (`created_at`),
    KEY `idx_embedding_m_deleted_496ead` (`deleted_at`),
    KEY `idx_embedding_m_type_491b8f` (`type`),
    KEY `idx_embedding_m_is_enab_d158c4` (`is_enabled`),
    KEY `idx_embedding_m_is_defa_d46056` (`is_default`)
) CHARACTER SET utf8mb4 COMMENT='Embedding 模型配置表';
CREATE TABLE IF NOT EXISTS `file_source` (
    `id` BIGINT NOT NULL PRIMARY KEY AUTO_INCREMENT COMMENT '主键',
    `created_at` DATETIME(6) NOT NULL COMMENT '创建时间' DEFAULT CURRENT_TIMESTAMP(6),
    `updated_at` DATETIME(6) NOT NULL COMMENT '更新时间' DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    `deleted_at` BIGINT COMMENT '删除时间',
    `name` VARCHAR(100) NOT NULL COMMENT '文件源名称',
    `type` VARCHAR(10) NOT NULL COMMENT '文件源类型',
    `config` JSON NOT NULL COMMENT '连接配置（JSON格式）',
    `is_enabled` BOOL NOT NULL COMMENT '是否启用' DEFAULT 1,
    `is_default` BOOL NOT NULL COMMENT '是否默认' DEFAULT 0,
    `description` LONGTEXT NOT NULL COMMENT '描述信息',
    UNIQUE KEY `uid_file_source_name_80971a` (`name`, `deleted_at`),
    KEY `idx_file_source_created_05ed79` (`created_at`),
    KEY `idx_file_source_deleted_a0dbf6` (`deleted_at`)
) CHARACTER SET utf8mb4 COMMENT='文件源配置表';
CREATE TABLE IF NOT EXISTS `document` (
    `id` BIGINT NOT NULL PRIMARY KEY AUTO_INCREMENT COMMENT '主键',
    `created_at` DATETIME(6) NOT NULL COMMENT '创建时间' DEFAULT CURRENT_TIMESTAMP(6),
    `updated_at` DATETIME(6) NOT NULL COMMENT '更新时间' DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    `deleted_at` BIGINT COMMENT '删除时间',
    `uri` VARCHAR(1000) NOT NULL COMMENT '文件唯一标识（本地为绝对路径）',
    `file_name` VARCHAR(255) NOT NULL COMMENT '文件名',
    `display_name` VARCHAR(255) NOT NULL COMMENT '显示名称',
    `extension` VARCHAR(50) NOT NULL COMMENT '文件扩展名',
    `file_size` BIGINT COMMENT '文件大小(字节)',
    `source_last_modified` DATETIME(6) COMMENT '文件源最后修改时间',
    `source_version_key` VARCHAR(100) COMMENT '文件源版本标识',
    `is_deleted_in_source` BOOL NOT NULL COMMENT '文件是否在源中被删除',
    `source_meta` JSON COMMENT '文件源元数据（JSON格式）',
    `short_summary` VARCHAR(255) COMMENT '文件摘要',
    `long_summary` LONGTEXT COMMENT '文件详细摘要',
    `status` VARCHAR(11) NOT NULL COMMENT '文件状态',
    `current_workflow_uid` CHAR(36) COMMENT '当前关联的最新工作流UID',
    `collection_id` BIGINT NOT NULL COMMENT '关联集合',
    `file_source_id` BIGINT NOT NULL COMMENT '关联文件源',
    UNIQUE KEY `uid_document_collect_815685` (`collection_id`, `uri`, `deleted_at`),
    UNIQUE KEY `uid_document_collect_d93e8a` (`collection_id`, `file_name`, `deleted_at`),
    CONSTRAINT `fk_document_collecti_6805aced` FOREIGN KEY (`collection_id`) REFERENCES `collection` (`id`) ON DELETE CASCADE,
    CONSTRAINT `fk_document_file_sou_77594071` FOREIGN KEY (`file_source_id`) REFERENCES `file_source` (`id`) ON DELETE RESTRICT,
    KEY `idx_document_created_b6cfaa` (`created_at`),
    KEY `idx_document_deleted_b64b7d` (`deleted_at`),
    KEY `idx_document_collect_2893ec` (`collection_id`),
    KEY `idx_document_collect_e0dffb` (`collection_id`, `status`),
    KEY `idx_document_collect_34bbbd` (`collection_id`, `extension`),
    KEY `idx_document_file_so_b54b49` (`file_source_id`)
) CHARACTER SET utf8mb4 COMMENT='文件记录表';
CREATE TABLE IF NOT EXISTS `indexing_backend_config` (
    `id` BIGINT NOT NULL PRIMARY KEY AUTO_INCREMENT COMMENT '主键',
    `created_at` DATETIME(6) NOT NULL COMMENT '创建时间' DEFAULT CURRENT_TIMESTAMP(6),
    `updated_at` DATETIME(6) NOT NULL COMMENT '更新时间' DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    `deleted_at` BIGINT COMMENT '删除时间',
    `name` VARCHAR(100) NOT NULL COMMENT '配置名称',
    `type` VARCHAR(13) NOT NULL COMMENT '后端类型（elasticsearch/milvus/qdrant等）',
    `host` VARCHAR(255) NOT NULL COMMENT '主机地址',
    `port` INT COMMENT '端口',
    `username` VARCHAR(100) COMMENT '用户名',
    `password` VARCHAR(255) COMMENT '密码',
    `api_key` VARCHAR(255) COMMENT 'API密钥',
    `secure` BOOL NOT NULL COMMENT '是否使用HTTPS/SSL' DEFAULT 0,
    `config` JSON NOT NULL COMMENT '额外配置参数（JSON格式）',
    `is_enabled` BOOL NOT NULL COMMENT '是否启用' DEFAULT 1,
    `is_default` BOOL NOT NULL COMMENT '是否默认配置' DEFAULT 0,
    `description` LONGTEXT NOT NULL COMMENT '描述信息',
    UNIQUE KEY `uid_indexing_ba_name_948346` (`name`, `deleted_at`),
    KEY `idx_indexing_ba_created_840754` (`created_at`),
    KEY `idx_indexing_ba_deleted_b9dd20` (`deleted_at`),
    KEY `idx_indexing_ba_type_28cd3f` (`type`),
    KEY `idx_indexing_ba_is_enab_f20d21` (`is_enabled`),
    KEY `idx_indexing_ba_is_defa_3e5519` (`is_default`)
) CHARACTER SET utf8mb4 COMMENT='索引后端配置表';
CREATE TABLE IF NOT EXISTS `llm_model_config` (
    `id` BIGINT NOT NULL PRIMARY KEY AUTO_INCREMENT COMMENT '主键',
    `created_at` DATETIME(6) NOT NULL COMMENT '创建时间' DEFAULT CURRENT_TIMESTAMP(6),
    `updated_at` DATETIME(6) NOT NULL COMMENT '更新时间' DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    `deleted_at` BIGINT COMMENT '删除时间',
    `name` VARCHAR(100) NOT NULL COMMENT '模型配置名称，不唯一',
    `type` VARCHAR(12) NOT NULL COMMENT '模型类型（openai/azure_openai/deepseek等）',
    `model_name` VARCHAR(255) NOT NULL COMMENT '模型标识符（如 gpt-4o, claude-3-opus, gemini-pro 等）',
    `config` JSON NOT NULL COMMENT '模型配置参数（JSON格式，如 api_key、base_url、temperature 等）',
    `capabilities` JSON NOT NULL COMMENT '模型能力配置（JSON格式），包含：function_calling, json_output, multimodal, streaming, vision, audio_input, audio_output 等',
    `max_tokens` INT NOT NULL COMMENT '模型最大输出 token 数' DEFAULT 4096,
    `max_retries` INT NOT NULL COMMENT '最大重试次数' DEFAULT 3,
    `timeout` INT NOT NULL COMMENT '请求超时时间（秒）' DEFAULT 60,
    `rate_limit` INT NOT NULL COMMENT '每分钟最大请求次数（0表示无限制）' DEFAULT 60,
    `is_enabled` BOOL NOT NULL COMMENT '是否启用' DEFAULT 1,
    `is_default` BOOL NOT NULL COMMENT '是否默认模型' DEFAULT 0,
    `description` LONGTEXT NOT NULL COMMENT '描述信息',
    UNIQUE KEY `uid_llm_model_c_model_n_8e7a23` (`model_name`, `deleted_at`),
    KEY `idx_llm_model_c_created_483236` (`created_at`),
    KEY `idx_llm_model_c_deleted_25ea9c` (`deleted_at`),
    KEY `idx_llm_model_c_type_ab2c6f` (`type`),
    KEY `idx_llm_model_c_is_enab_578b90` (`is_enabled`),
    KEY `idx_llm_model_c_is_defa_011321` (`is_default`)
) CHARACTER SET utf8mb4 COMMENT='LLM 模型配置表';
CREATE TABLE IF NOT EXISTS `workflow` (
    `id` BIGINT NOT NULL PRIMARY KEY AUTO_INCREMENT COMMENT '主键',
    `created_at` DATETIME(6) NOT NULL COMMENT '创建时间' DEFAULT CURRENT_TIMESTAMP(6),
    `updated_at` DATETIME(6) NOT NULL COMMENT '更新时间' DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    `deleted_at` BIGINT COMMENT '删除时间',
    `uid` CHAR(36) NOT NULL UNIQUE COMMENT '幂等性唯一标识',
    `config` JSON NOT NULL COMMENT '工作流配置（DAG图结构，支持YAML/JSON格式）, {content: vv}',
    `config_format` VARCHAR(4) NOT NULL COMMENT '配置格式（yaml/json/python）' DEFAULT 'yaml',
    `status` VARCHAR(9) NOT NULL COMMENT '工作流状态' DEFAULT 'pending',
    `started_at` DATETIME(6) COMMENT '开始执行时间',
    `completed_at` DATETIME(6) COMMENT '完成时间',
    `canceled_at` DATETIME(6) COMMENT '是否被取消',
    `schedule_celery_task_id` VARCHAR(255) COMMENT 'schedule 任务ID',
    KEY `idx_workflow_created_c2a8eb` (`created_at`),
    KEY `idx_workflow_deleted_d3d134` (`deleted_at`)
) CHARACTER SET utf8mb4 COMMENT='工作流定义表';
CREATE TABLE IF NOT EXISTS `aerich` (
    `id` INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    `version` VARCHAR(255) NOT NULL,
    `app` VARCHAR(100) NOT NULL,
    `content` JSON NOT NULL
) CHARACTER SET utf8mb4;"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        """
