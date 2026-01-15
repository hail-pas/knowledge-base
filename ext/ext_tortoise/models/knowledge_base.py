from tortoise import fields
from ext.ext_tortoise.base.models import BaseModel, CreateOnlyModel
from ext.ext_tortoise.main import ConnectionNameEnum
from ext.ext_tortoise.enums import (
    ActivityStatusEnum,
    DocumentStatusEnum,
    EmbeddingModelTypeEnum,
    FileSourceTypeEnum,
    IndexingBackendTypeEnum,
    WorkflowConfigFormatEnum,
    WorkflowStatusEnum,
    LLMModelTypeEnum
)

_KBConnectionName = ConnectionNameEnum.knowledge_base.value


class Collection(BaseModel):
    """文件集合表

    用于文件分组管理和用户权限隔离
    """
    name = fields.CharField(max_length=100, description="集合名称")
    description = fields.TextField(description="描述", null=True)
    user_id = fields.UUIDField(description="用户ID", null=True)
    tenant_id = fields.UUIDField(description="租户ID", null=True)
    role_id = fields.UUIDField(description="角色ID", null=True)
    is_public = fields.BooleanField(default=False, description="是否公开")
    is_temp = fields.BooleanField(default=False, description="是否临时")

    class Meta: # type: ignore
        table = "collection"
        table_description = "文件集合表"
        app = _KBConnectionName
        # user_id 和 name 联合唯一
        unique_together = ("user_id", "name")
        ordering = ["-id"]

    def __str__(self):
        return f"{self.name} ({self.user_id})"


class FileSource(BaseModel):
    """文件源配置表

    存储各类文件源的连接配置信息，支持多种文件源类型
    """
    name = fields.CharField(max_length=100, description="文件源名称")
    type = fields.CharEnumField(FileSourceTypeEnum, description="文件源类型")
    config = fields.JSONField(description="连接配置（JSON格式）", default=dict)
    is_enabled = fields.BooleanField(default=True, description="是否启用")
    is_default = fields.BooleanField(default=False, description="是否默认")
    description = fields.TextField(description="描述信息", default="")

    class Meta: # type: ignore
        table = "file_source"
        table_description = "文件源配置表"
        app = _KBConnectionName
        unique_together = [("name", "deleted_at")]
        ordering = ["-id"]

    def __str__(self):
        return f"{self.name} ({self.type})"


class Document(BaseModel):
    """文件记录表

    存储从文件源获取的文件元数据信息
    - uri:
        - oss_key
        - /abs/path
        - site+driveid+itemid
    """
    collection = fields.ForeignKeyField(
        f"{_KBConnectionName}.Collection",
        related_name="documents",
        on_delete=fields.CASCADE,
        description="关联集合"
    )
    file_source = fields.ForeignKeyField(
        f"{_KBConnectionName}.FileSource",
        related_name="documents",
        on_delete=fields.RESTRICT,
        description="关联文件源"
    )
    uri = fields.CharField(max_length=1000, description="文件唯一标识（本地为绝对路径）")
    file_name = fields.CharField(max_length=255, description="文件名")
    display_name = fields.CharField(max_length=255, description="显示名称")
    extension = fields.CharField(max_length=50, description="文件扩展名")
    file_size = fields.BigIntField(description="文件大小(字节)", null=True)
    source_last_modified = fields.DatetimeField(description="文件源最后修改时间", null=True)
    source_version_key = fields.CharField(max_length=100, description="文件源版本标识", null=True)
    is_deleted_in_source = fields.BooleanField(description="文件是否在源中被删除")
    source_meta = fields.JSONField(description="文件源元数据（JSON格式）", null=True)
    short_summary = fields.CharField(max_length=255, description="文件摘要", null=True)
    long_summary = fields.TextField(description="文件详细摘要", null=True)
    workflow_version = fields.SmallIntField(default=1, description="工作流版本")
    status = fields.CharEnumField(DocumentStatusEnum, description="文件状态")
    current_workflow_uid = fields.UUIDField(null=True, description="当前关联的最新工作流UID")

    class Meta: # type: ignore
        table = "document"
        table_description = "文件记录表"
        app = _KBConnectionName
        unique_together = [
            ("collection_id", "uri", "deleted_at"),
            ("collection_id", "file_name", "deleted_at")
        ]
        indexes = [
            ("collection_id", ),
            ("collection_id", "status"),
            ("collection_id", "extension"),
            ("file_source_id",),
        ]
        ordering = ["-id"]

    def __str__(self):
        return f"{self.file_name} ({self.status})"


class Workflow(BaseModel):
    """工作流定义表

    存储工作流的定义和配置，支持通过YAML/JSON自定义node执行graph
    """
    uid = fields.UUIDField(unique=True, description="幂等性唯一标识")
    config = fields.JSONField(description="工作流配置（DAG图结构，支持YAML/JSON格式）, {content: vv}", default=dict)
    config_format = fields.CharEnumField(
        WorkflowConfigFormatEnum,
        default=WorkflowConfigFormatEnum.yaml.value,
        description="配置格式（yaml/json/python）"
    )
    status = fields.CharEnumField(
        WorkflowStatusEnum,
        default=WorkflowStatusEnum.pending.value,
        description="工作流状态"
    )
    started_at = fields.DatetimeField(null=True, description="开始执行时间")
    completed_at = fields.DatetimeField(null=True, description="完成时间")
    canceled_at = fields.DatetimeField(null=True, description="是否被取消")

    schedule_celery_task_id = fields.CharField(max_length=255, null=True, description="schedule 任务ID")

    class Meta: # type: ignore
        table = "workflow"
        table_description = "工作流定义表"
        app = _KBConnectionName
        ordering = ["-id"]

    def __str__(self):
        return f"Workflow({self.uid})"


class Activity(CreateOnlyModel):
    """工作流活动节点定义表

    存储工作流中的节点定义和配置
    """
    workflow_uid = fields.UUIDField(description="工作流UID")
    uid = fields.UUIDField(unique=True)
    name = fields.CharField(max_length=200, description="名称")
    input = fields.JSONField(description="输入参数", default=dict)
    output = fields.JSONField(description="输出参数", default=dict)
    retry_count = fields.IntField(default=0, description="失败重试次数")
    execute_params = fields.JSONField(description="Celery 执行参数", default=dict)
    status = fields.CharEnumField(
        ActivityStatusEnum,
        default=ActivityStatusEnum.pending.value,
        description="状态"
    )

    # 监测支持
    started_at = fields.DatetimeField(null=True, description="开始执行时间")
    completed_at = fields.DatetimeField(null=True, description="完成时间")
    error_message = fields.TextField(null=True, description="错误信息")
    stack_trace = fields.TextField(null=True, description="错误堆栈跟踪")

    canceled_at = fields.DatetimeField(null=True, description="取消时间")

    # 幂等性支持
    celery_task_id = fields.CharField(max_length=255, null=True, description="Celery 任务ID")

    class Meta: # type: ignore
        table = "activity"
        table_description = "工作流活动表"
        app = _KBConnectionName
        unique_together = [("workflow_uid", "name")]
        indexes = [
            ("workflow_uid",),
            ("status",),
            ("celery_task_id",),
        ]
        ordering = ["-id"]

    def __str__(self):
        return f"{self.workflow_uid}: {self.name} ({self.status})" # type: ignore


class EmbeddingModelConfig(BaseModel):
    """Embedding 模型配置表

    存储各类 embedding 模型的配置信息，支持动态切换不同的 embedding 服务
    """
    name = fields.CharField(max_length=100, description="模型配置名称")
    type = fields.CharEnumField(
        EmbeddingModelTypeEnum,
        description="模型类型（openai/sentence_transformers等）"
    )
    model_name_or_path = fields.CharField(
        max_length=255,
        description="模型标识符或路径（如 text-embedding-3-small, sentence-transformers/...）"
    )
    config = fields.JSONField(
        description="模型配置参数（JSON格式，如 api_key、endpoint 等）",
        default=dict
    )
    dimension = fields.IntField(description="向量维度", null=True)
    max_batch_size = fields.IntField(
        default=32,
        description="最大批处理大小，避免文本过大导致服务端无法处理"
    )
    max_token_per_request = fields.IntField(
        default=8191,
        description="单次请求最大 token 数"
    )
    max_token_per_text = fields.IntField(
        default=512,
        description="单个文本的最大 token 长度"
    )
    is_enabled = fields.BooleanField(default=True, description="是否启用")
    is_default = fields.BooleanField(default=False, description="是否默认模型")
    description = fields.TextField(description="描述信息", default="")

    class Meta: # type: ignore
        table = "embedding_model_config"
        table_description = "Embedding 模型配置表"
        app = _KBConnectionName
        unique_together = [("model_name_or_path", "deleted_at")]
        indexes = [
            ("type",),
            ("is_enabled",),
            ("is_default",),
        ]
        ordering = ["-id"]

    def __str__(self):
        return f"{self.name} ({self.type})"


class IndexingBackendConfig(BaseModel):
    """索引后端配置表

    存储各类索引后端服务（如Elasticsearch、Milvus等）的连接配置信息
    """
    name = fields.CharField(max_length=100, description="配置名称")
    type = fields.CharEnumField(
        IndexingBackendTypeEnum,
        description="后端类型（elasticsearch/milvus/qdrant等）"
    )
    host = fields.CharField(max_length=255, description="主机地址")
    port = fields.IntField(description="端口", null=True)
    username = fields.CharField(max_length=100, description="用户名", null=True)
    password = fields.CharField(max_length=255, description="密码", null=True)
    api_key = fields.CharField(max_length=255, description="API密钥", null=True)
    secure = fields.BooleanField(default=False, description="是否使用HTTPS/SSL")
    config = fields.JSONField(description="额外配置参数（JSON格式）", default=dict)
    is_enabled = fields.BooleanField(default=True, description="是否启用")
    is_default = fields.BooleanField(default=False, description="是否默认配置")
    description = fields.TextField(description="描述信息", default="")

    class Meta: # type: ignore
        table = "indexing_backend_config"
        table_description = "索引后端配置表"
        app = _KBConnectionName
        unique_together = [("name", "deleted_at")]
        indexes = [
            ("type",),
            ("is_enabled",),
            ("is_default",),
        ]
        ordering = ["-id"]

    def __str__(self):
        return f"{self.name} ({self.type})"


class LLMModelConfig(BaseModel):
    """LLM 模型配置表

    存储各类大语言模型的配置信息，支持动态切换不同的 LLM 服务和模型能力
    """
    name = fields.CharField(max_length=100, description="模型配置名称，不唯一")
    type = fields.CharEnumField(
        LLMModelTypeEnum,
        description="模型类型（openai/azure_openai/deepseek等）"
    )
    model_name = fields.CharField(
        max_length=255,
        description="模型标识符（如 gpt-4o, claude-3-opus, gemini-pro 等）"
    )
    config = fields.JSONField(
        description="模型配置参数（JSON格式，如 api_key、base_url、temperature 等）",
        default=dict
    )

    # 能力配置（JSON格式）
    capabilities = fields.JSONField(
        description="模型能力配置（JSON格式），包含：function_calling, json_output, multimodal, streaming, vision, audio_input, audio_output 等",
        default=dict
    )

    # 模型限制参数
    max_tokens = fields.IntField(
        default=4096,
        description="模型最大输出 token 数"
    )
    max_retries = fields.IntField(default=3, description="最大重试次数")
    timeout = fields.IntField(default=60, description="请求超时时间（秒）")
    rate_limit = fields.IntField(
        default=60,
        description="每分钟最大请求次数（0表示无限制）"
    )

    # 状态标记
    is_enabled = fields.BooleanField(default=True, description="是否启用")
    is_default = fields.BooleanField(default=False, description="是否默认模型")
    description = fields.TextField(description="描述信息", default="")

    class Meta:  # type: ignore
        table = "llm_model_config"
        table_description = "LLM 模型配置表"
        app = _KBConnectionName
        unique_together = [("model_name", "deleted_at")]
        indexes = [
            ("type",),
            ("is_enabled",),
            ("is_default",),
        ]
        ordering = ["-id"]

    def __str__(self):
        return f"{self.name} ({self.type}/{self.model_name})"
