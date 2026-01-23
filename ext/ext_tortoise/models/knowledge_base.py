from tortoise import fields
from config import default
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
    LLMModelTypeEnum,
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
    workflow_template = fields.JSONField(default=dict, description="默认DAG工作流模版")
    extra_config = fields.JSONField(
        default=dict, description="额外配置：gen_faq: bool, use_mineru: bool 等，可继续扩展"
    )
    # workflow template 默认值为：
    # {
    #     "fetch_file": {
    #         "input": {"file_path": file_path},
    #         "execute_params": {"task_name": "workflow_activity.FetchFileTask"},
    #         "depends_on": []
    #     },
    #     "load_file": {
    #         "execute_params": {"task_name": "workflow_activity.LoadFileTask"},
    #         "depends_on": ["fetch_file"]
    #     },
    #     "replace_content": {
    #         "execute_params": {"task_name": "workflow_activity.ReplaceContentTask"},
    #         "depends_on": ["load_file"],
    #         "input": {"replace_rules": []}
    #     },
    #     "summary": {
    #         "execute_params": {"task_name": "workflow_activity.SummaryTask"},
    #         "depends_on": ["replace_content"],
    #         "input": {"max_length": 100}
    #     },
    #     "split_text": {
    #         "execute_params": {"task_name": "workflow_activity.SplitTask"},
    #         "depends_on": ["replace_content"],
    #         "input": {"split_policy": "markdown"}
    #     },
    #     "index_to_milvus": {
    #         "execute_params": {"task_name": "workflow_activity.IndexMilvusTask"},
    #         "depends_on": ["split_text"],
    #         "input": {}
    #     },
    #     "index_to_es": {
    #         "execute_params": {"task_name": "workflow_activity.IndexEsTask"},
    #         "depends_on": ["replace_content"],
    #         "input": {}
    #     },
    #     "generate_tag": {
    #         "execute_params": {"task_name": "workflow_activity.GenTagTask"},
    #         "depends_on": ["replace_content"],
    #         "input": {}
    #     }
    # }

    class Meta:  # type: ignore
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
    采用公共字段明确 + JSON 存储provider特定配置的模式
    """

    # ========== 基础信息（所有 provider 共有）==========
    name = fields.CharField(max_length=100, description="文件源名称")
    type = fields.CharEnumField(FileSourceTypeEnum, description="文件源类型")

    # ========== 认证信息（S3 兼容 providers）==========
    access_key = fields.CharField(max_length=500, null=True, description="访问密钥")
    secret_key = fields.CharField(max_length=500, null=True, description="密钥（加密存储）")

    # ========== 连接信息（合并字段）==========
    storage_location = fields.CharField(
        max_length=1000,
        null=True,
        description="存储位置"
        "  - type=local_file: 本地文件路径（如 /data/documents）"
        "  - type=s3/minio/aliyun_oss: 存储桶名称（如 my-bucket）",
    )

    # ========== 连接信息（对象存储通用）==========
    endpoint = fields.CharField(max_length=500, null=True, description="服务端点URL")
    region = fields.CharField(max_length=100, null=True, description="区域/地域")

    # ========== 安全配置（所有 provider）==========
    use_ssl = fields.BooleanField(default=True, description="是否使用SSL/TLS")
    verify_ssl = fields.BooleanField(default=True, description="是否验证SSL证书")
    timeout = fields.IntField(default=30, description="连接/读取超时时间(秒)")

    # ========== 性能配置（所有 provider）==========
    max_retries = fields.IntField(default=3, description="最大重试次数")
    concurrent_limit = fields.IntField(default=10, description="并发限制")
    max_connections = fields.IntField(default=100, description="最大连接数(连接池)")

    # ========== 状态管理 ==========
    is_enabled = fields.BooleanField(default=True, description="是否启用")
    is_default = fields.BooleanField(default=False, description="是否默认")
    description = fields.TextField(description="描述信息", default="")

    # ========== Provider 特定扩展配置 ==========
    extra_config = fields.JSONField(default=dict, description="Provider特定扩展配置（JSON格式）")

    class Meta:  # type: ignore
        table = "file_source"
        table_description = "文件源配置表"
        app = _KBConnectionName
        unique_together = [("name", "deleted_at")]
        indexes = [("type", "is_enabled"), ("is_default", "is_enabled"), ("created_at",)]
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
        f"{_KBConnectionName}.Collection", related_name="documents", on_delete=fields.CASCADE, description="关联集合"
    )
    file_source = fields.ForeignKeyField(
        f"{_KBConnectionName}.FileSource", related_name="documents", on_delete=fields.RESTRICT, description="关联文件源"
    )
    uri = fields.CharField(max_length=1000, description="文件唯一标识（本地为绝对路径）")
    file_name = fields.CharField(max_length=255, description="文件名")
    display_name = fields.CharField(max_length=255, description="显示名称")
    extension = fields.CharField(max_length=50, description="文件扩展名")
    file_size = fields.BigIntField(description="文件大小(字节)", null=True)
    source_last_modified = fields.DatetimeField(description="文件源最后修改时间", null=True)
    source_version_key = fields.CharField(max_length=100, description="文件源版本标识", null=True)
    is_deleted_in_source = fields.BooleanField(default=False, description="文件是否在源中被删除")
    source_meta = fields.JSONField(description="文件源元数据（JSON格式）", null=True)
    short_summary = fields.CharField(max_length=255, description="文件摘要", null=True)
    long_summary = fields.TextField(description="文件详细摘要", null=True)
    status = fields.CharEnumField(DocumentStatusEnum, description="文件状态")
    current_workflow_uid = fields.UUIDField(null=True, description="当前关联的最新工作流UID")

    class Meta:  # type: ignore
        table = "document"
        table_description = "文件记录表"
        app = _KBConnectionName
        unique_together = [("collection_id", "uri", "deleted_at"), ("collection_id", "file_name", "deleted_at")]
        indexes = [
            ("collection_id",),
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
        description="配置格式（yaml/json/python）",
    )
    status = fields.CharEnumField(
        WorkflowStatusEnum, default=WorkflowStatusEnum.pending.value, description="工作流状态"
    )
    started_at = fields.DatetimeField(null=True, description="开始执行时间")
    completed_at = fields.DatetimeField(null=True, description="完成时间")
    canceled_at = fields.DatetimeField(null=True, description="是否被取消")

    schedule_celery_task_id = fields.CharField(max_length=255, null=True, description="schedule 任务ID")

    class Meta:  # type: ignore
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
    status = fields.CharEnumField(ActivityStatusEnum, default=ActivityStatusEnum.pending.value, description="状态")

    # 监测支持
    started_at = fields.DatetimeField(null=True, description="开始执行时间")
    completed_at = fields.DatetimeField(null=True, description="完成时间")
    error_message = fields.TextField(null=True, description="错误信息")
    stack_trace = fields.TextField(null=True, description="错误堆栈跟踪")

    canceled_at = fields.DatetimeField(null=True, description="取消时间")

    # 幂等性支持
    celery_task_id = fields.CharField(max_length=255, null=True, description="Celery 任务ID")

    class Meta:  # type: ignore
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
        return f"{self.workflow_uid}: {self.name} ({self.status})"  # type: ignore


class EmbeddingModelConfig(BaseModel):
    """Embedding 模型配置表

    存储各类 embedding 模型的配置信息，支持动态切换不同的 embedding 服务
    """

    # 基本配置
    name = fields.CharField(max_length=100, unique=True, description="配置名称")
    type = fields.CharEnumField(EmbeddingModelTypeEnum, description="模型类型")
    model_name = fields.CharField(max_length=255, description="模型标识符")

    # API配置（必填）
    api_key = fields.CharField(max_length=500, null=True, description="API密钥")
    base_url = fields.CharField(max_length=500, null=True, description="API基础URL")

    # 模型配置（必填）
    dimension = fields.IntField(null=False, description="向量维度")
    max_chunk_length = fields.IntField(default=8192, description="单条chunk最大长度")

    # 批处理配置
    batch_size = fields.IntField(default=100, description="批处理大小")
    max_retries = fields.IntField(default=3, description="最大重试次数")
    timeout = fields.IntField(default=60, description="请求超时时间(秒)")
    rate_limit = fields.IntField(default=60, description="每分钟最大请求数(0=无限制)")

    # 扩展配置（provider特定配置）
    extra_config = fields.JSONField(default=dict, description="provider特定扩展配置")

    # 状态配置
    is_enabled = fields.BooleanField(default=True, description="是否启用")
    is_default = fields.BooleanField(default=False, description="是否默认配置")
    description = fields.TextField(description="描述信息", default="")

    class Meta:  # type: ignore
        table = "embedding_model_config"
        table_description = "Embedding模型配置表"
        app = _KBConnectionName
        indexes = [
            ("is_enabled",),
            ("is_default",),
            ("type",),
        ]
        ordering = ["-id"]


class IndexingBackendConfig(BaseModel):
    """索引后端配置表

    存储各类索引后端服务（如Elasticsearch、Milvus等）的连接配置信息
    """


class LLMModelConfig(BaseModel):
    """LLM 模型配置表

    存储各类 LLM 模型的配置信息，支持动态切换不同的 LLM 服务
    """

    # 基础信息
    name = fields.CharField(max_length=100, unique=True, description="配置名称")
    type = fields.CharEnumField(LLMModelTypeEnum, description="模型类型")
    model_name = fields.CharField(max_length=255, description="模型标识符")

    # API配置
    api_key = fields.CharField(max_length=500, null=True, description="API密钥（加密）")
    base_url = fields.CharField(max_length=500, null=True, description="API基础URL")

    # 模型能力标识
    max_tokens = fields.IntField(default=4096, description="最大token数（输入+输出）")
    supports_chat = fields.BooleanField(default=True, description="支持对话模式")
    supports_completion = fields.BooleanField(default=False, description="支持补全模式")
    supports_streaming = fields.BooleanField(default=True, description="支持流式输出")
    supports_function_calling = fields.BooleanField(default=False, description="支持函数调用")
    supports_vision = fields.BooleanField(default=False, description="支持视觉/图像")

    # 默认参数配置
    default_temperature = fields.FloatField(default=0.7, description="默认温度参数")
    default_top_p = fields.FloatField(default=1.0, description="默认top_p")
    max_retries = fields.IntField(default=3, description="最大重试次数")
    timeout = fields.IntField(default=60, description="请求超时时间(秒)")

    # Provider特定配置
    extra_config = fields.JSONField(default=dict, description="provider特定扩展配置")

    # 状态配置
    is_enabled = fields.BooleanField(default=True, description="是否启用")
    is_default = fields.BooleanField(default=False, description="是否默认配置")
    description = fields.TextField(description="描述信息", default="")

    class Meta:  # type: ignore
        table = "llm_model_config"
        table_description = "LLM模型配置表"
        app = _KBConnectionName
        indexes = [
            ("is_enabled",),
            ("is_default",),
            ("type",),
        ]
        ordering = ["-id"]

    def __str__(self):
        return f"{self.name} ({self.type.value})"


class ConversationMemory(BaseModel):
    """对话记忆表

    用于持久化存储 Agent 的对话历史
    """

    uid = fields.UUIDField(unique=True, description="会话唯一标识")
    session_id = fields.CharField(max_length=255, description="会话ID")
    user_id = fields.UUIDField(null=True, description="用户ID")
    agent_type = fields.CharField(max_length=50, description="Agent类型")

    # 对话历史（存储为 JSON 数组）
    messages = fields.JSONField(default=list, description="对话历史消息列表")

    # 元数据
    agent_config = fields.JSONField(default=dict, description="Agent配置")
    tags = fields.JSONField(default=list, description="标签列表")
    metadata = fields.JSONField(default=dict, description="其他元数据")

    # 时间戳
    last_updated = fields.DatetimeField(auto_now=True, description="最后更新时间")

    class Meta:  # type: ignore
        table = "conversation_memory"
        table_description = "对话记忆表"
        app = _KBConnectionName
        indexes = [
            ("session_id",),
            ("user_id",),
            ("agent_type",),
            ("last_updated",),
        ]
        ordering = ["-last_updated"]

    def __str__(self):
        return f"ConversationMemory({self.session_id})"
