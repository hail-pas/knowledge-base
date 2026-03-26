from enum import IntEnum

from tortoise import fields

from constant.symbol import PAGE_SEPARATOR
from ext.ext_tortoise.main import ConnectionNameEnum
from ext.ext_tortoise.enums import (
    ChatAgentMountModeEnum,
    ChatAgentRoleEnum,
    ChatCapabilityCategoryEnum,
    ChatDataKindEnum,
    ChatCapabilityRuntimeKindEnum,
    ChatStepKindEnum,
    LLMModelTypeEnum,
    ActivityStatusEnum,
    ChatStepStatusEnum,
    ChatTurnStatusEnum,
    DocumentStatusEnum,
    FileSourceTypeEnum,
    WorkflowStatusEnum,
    ChatTurnTriggerEnum,
    ChatCapabilityKindEnum,
    EmbeddingModelTypeEnum,
    IndexingBackendTypeEnum,
    WorkflowConfigFormatEnum,
)
from ext.ext_tortoise.base.models import BaseModel, CreateOnlyModel

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
    is_external = fields.BooleanField(default=False, description="是否外部")
    # workflow DAG template 格式参考 ext/workflow 的实现
    workflow_template = fields.JSONField(default=dict, description="默认DAG工作流模版")
    external_config = fields.JSONField(default=dict, description="外部知识库配置, 占位后续扩展")
    embedding_model_config = fields.ForeignKeyField(
        f"{_KBConnectionName}.EmbeddingModelConfig",
        related_name="collections",
        null=True,
        on_delete=fields.RESTRICT,
        description="关联嵌入模型",
    )

    class Meta:  # type: ignore
        table = "collection"
        table_description = "文件集合表"
        app = _KBConnectionName
        # user_id 和 name 联合唯一
        unique_together = ("user_id", "name")
        ordering = ["-id"]

    def __str__(self) -> str:
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
        "  - type=s3/minio/aliyun_oss: 存储桶名称（如 my-bucket）, sharepoint: 站点路径, api: API路径",
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
    user_id = fields.UUIDField(description="用户ID, 为空时表示公共", null=True)
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

    def __str__(self) -> str:
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
        description="关联集合",
    )
    file_source = fields.ForeignKeyField(
        f"{_KBConnectionName}.FileSource",
        related_name="documents",
        on_delete=fields.RESTRICT,
        description="关联文件源",
    )
    uri = fields.CharField(max_length=1000, description="文件唯一标识（本地为绝对路径）")
    parsed_uri = fields.CharField(max_length=1000, null=True, description="解析后的文件唯一标识")
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
    config_flag = fields.SmallIntField(default=0, description="配置标志")
    workflow_template = fields.JSONField(default=dict, description="DAG工作流模版")

    class ConfigType(IntEnum):
        parse_image = 0  # bit 0
        faq_file = 1  # bit 1

    def check_config_flag(self, config_type: ConfigType) -> bool:
        return bool(self.config_flag & (1 << config_type.value))

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

    def __str__(self) -> str:
        return f"{self.file_name} ({self.status})"


class DocumentPages(BaseModel):
    document = fields.ForeignKeyField(f"{_KBConnectionName}.Document", related_name="pages", description="文档ID")
    page_number = fields.SmallIntField(description="页码")
    content = fields.TextField(
        description="页面内容，当只有一页的时候content太长可以不存，但是必须要有page记录，content可以从parsed_uri获取",
    )
    tables = fields.JSONField(description="页面表格信息")
    images = fields.JSONField(description="页面图片信息, list[keys]")
    metadata = fields.JSONField(description="页面元数据")

    @classmethod
    def pages_to_content(cls, pages: list["DocumentPages"]) -> str:
        # 和 text chunker 的 coordinate mapper 保持一致
        return PAGE_SEPARATOR.join([page.content for page in pages])

    class Meta:  # type: ignore
        table = "document_page"
        indexes = [
            ("document_id",),
            ("document_id", "page_number"),
        ]
        ordering = ["-id"]

    def __str__(self) -> str:
        return f"{self.document_id} - Page {self.page_number}"  # type: ignore


class DocumentChunk(BaseModel):
    """文档切块记录表"""

    document = fields.ForeignKeyField(f"{_KBConnectionName}.Document", related_name="chunks", description="文档ID")
    content = fields.TextField(description="切块内容")
    pages = fields.JSONField(description="切块页码列表")
    min_page = fields.SmallIntField(description="最小页码")
    max_page = fields.SmallIntField(description="最大页码")
    start = fields.JSONField(description="起始位置（页码+页内偏移）")
    end = fields.JSONField(description="结束位置（页码+页内偏移）")
    overlap_start = fields.JSONField(null=True, description="重叠起始位置（页码+页内偏移）")
    overlap_end = fields.JSONField(null=True, description="重叠结束位置（页码+页内偏移）")
    metadata = fields.JSONField(description="元数据", default=dict)
    manual_add = fields.BooleanField(default=False, description="是否手动添加")

    class Meta:  # type: ignore
        table = "document_chunk"
        indexes = [
            ("document_id",),
            ("document_id", "min_page"),
            ("document_id", "max_page"),
        ]
        ordering = ["-id"]

    def __str__(self) -> str:
        return f"{self.document_id} - Page {self.pages}"  # type: ignore


class DocumentGeneratedFaq(BaseModel):
    """文档生成FAQ表"""

    document = fields.ForeignKeyField(
        f"{_KBConnectionName}.Document",
        related_name="generated_faqs",
        description="文档",
    )
    content = fields.TextField(null=True, description="相关文档内容块")
    question = fields.TextField(null=True, description="问题")
    answer = fields.TextField(null=True, description="答案")
    manual_add = fields.BooleanField(default=False, description="是否手动添加")
    enabled = fields.BooleanField(default=True, description="是否启用")

    class Meta:  # type: ignore
        table = "document_gfaq"
        indexes = [
            ("document_id",),
            ("document_id", "enabled"),
        ]
        ordering = ["-id"]

    def __str__(self) -> str:
        return f"{self.document_id} - {self.question}"  # type: ignore


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
        WorkflowStatusEnum,
        default=WorkflowStatusEnum.pending.value,
        description="工作流状态",
    )
    started_at = fields.DatetimeField(null=True, description="开始执行时间")
    completed_at = fields.DatetimeField(null=True, description="完成时间")
    canceled_at = fields.DatetimeField(null=True, description="是否被取消")

    class Meta:  # type: ignore
        table = "workflow"
        table_description = "工作流定义表"
        app = _KBConnectionName
        ordering = ["-id"]

    def __str__(self) -> str:
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
    execute_params = fields.JSONField(description="额外的 Celery 执行参数，覆盖默认值", default=dict)
    status = fields.CharEnumField(ActivityStatusEnum, default=ActivityStatusEnum.pending.value, description="状态")

    # 监测支持
    started_at = fields.DatetimeField(null=True, description="开始执行时间")
    completed_at = fields.DatetimeField(null=True, description="完成时间")
    error_message = fields.TextField(null=True, description="错误信息")
    stack_trace = fields.TextField(null=True, description="错误堆栈跟踪")

    canceled_at = fields.DatetimeField(null=True, description="取消时间")

    celery_task_id = fields.CharField(max_length=255, null=True, description="Celery 任务ID, direct执行模式下为空")

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

    def __str__(self) -> str:
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

    # 基础信息
    name = fields.CharField(max_length=100, unique=True, description="配置名称")
    type = fields.CharEnumField(IndexingBackendTypeEnum, description="后端类型")

    # 连接配置（公共字段）
    host = fields.CharField(max_length=255, null=True, description="主机地址")
    port = fields.IntField(null=True, description="端口号")
    username = fields.CharField(max_length=255, null=True, description="用户名")
    password = fields.CharField(max_length=500, null=True, description="密码（加密）")

    # 安全配置（公共）
    use_ssl = fields.BooleanField(default=True, description="是否使用SSL/TLS")
    verify_ssl = fields.BooleanField(default=True, description="是否验证SSL证书")
    timeout = fields.IntField(default=30, description="连接超时时间(秒)")
    max_retries = fields.IntField(default=3, description="最大重试次数")

    # 连接池配置（公共）
    max_connections = fields.IntField(default=100, description="最大连接数(连接池)")

    # Provider 特定配置（JSON 存储差异）
    extra_config = fields.JSONField(default=dict, description="Provider特定扩展配置")

    # 状态配置
    is_enabled = fields.BooleanField(default=True, description="是否启用")
    is_default = fields.BooleanField(default=False, description="是否默认配置")
    description = fields.TextField(default="", description="描述信息")

    class Meta:  # type: ignore
        table = "indexing_backend_config"
        table_description = "索引后端配置表"
        app = _KBConnectionName
        indexes = [
            ("type", "is_enabled"),
            ("is_default", "is_enabled"),
        ]
        ordering = ["-id"]

    def __str__(self) -> str:
        return f"{self.name} ({self.type.value})"


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

    # ModelSettings 对齐字段
    max_tokens = fields.IntField(default=4096, description="默认最大输出 token 数")
    temperature = fields.FloatField(default=0.7, description="默认 temperature")
    top_p = fields.FloatField(default=1.0, description="默认 top_p")
    presence_penalty = fields.FloatField(null=True, description="默认 presence_penalty")
    frequency_penalty = fields.FloatField(null=True, description="默认 frequency_penalty")
    seed = fields.IntField(null=True, description="默认 seed")
    timeout = fields.IntField(default=60, description="默认请求超时时间(秒)")
    parallel_tool_calls = fields.BooleanField(null=True, description="默认 parallel_tool_calls")

    # ModelProfile 对齐字段
    supports_tools = fields.BooleanField(default=False, description="是否支持 tools")
    supports_image_output = fields.BooleanField(default=False, description="是否支持图片输出")
    supports_json_schema_output = fields.BooleanField(default=False, description="是否支持 JSON Schema 输出")
    supports_json_object_output = fields.BooleanField(default=False, description="是否支持 JSON Object 输出")
    default_structured_output_mode = fields.CharField(
        max_length=16,
        default="tool",
        description="默认结构化输出模式(tool/native/prompted)",
    )
    native_output_requires_schema_in_instructions = fields.BooleanField(
        default=False,
        description="native 输出模式是否要求把 schema 写入 instructions",
    )

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

    def __str__(self) -> str:
        return f"{self.name} ({self.type.value})"


class ChatCapabilityPackage(BaseModel):
    owner_account_id = fields.BigIntField(null=True, description="所属账户ID，空表示全局")
    kind = fields.CharEnumField(ChatCapabilityKindEnum, description="能力包类型")
    category = fields.CharEnumField(
        ChatCapabilityCategoryEnum,
        default=ChatCapabilityCategoryEnum.domain.value,
        description="能力分类",
    )
    runtime_kind = fields.CharEnumField(
        ChatCapabilityRuntimeKindEnum,
        default=ChatCapabilityRuntimeKindEnum.local_toolset.value,
        description="运行时承载类型",
    )
    capability_key = fields.CharField(max_length=128, description="能力包唯一键")
    name = fields.CharField(max_length=128, description="能力包名称")
    description = fields.TextField(default="", description="能力包描述")
    manifest = fields.JSONField(default=dict, description="能力包 manifest")
    visible_to_agents = fields.JSONField(default=list, description="允许暴露给哪些 agent")
    requires_deps = fields.JSONField(default=list, description="依赖字段声明")
    is_enabled = fields.BooleanField(default=True, description="是否启用")
    metadata = fields.JSONField(default=dict, description="附加元数据")
    version = fields.IntField(default=1, description="版本号")

    class Meta:  # type: ignore
        table = "chat_capability_package"
        table_description = "聊天能力包定义表"
        app = _KBConnectionName
        unique_together = [("owner_account_id", "kind", "capability_key", "deleted_at")]
        indexes = [
            ("owner_account_id", "kind", "is_enabled"),
            ("kind", "capability_key", "is_enabled"),
            ("category", "runtime_kind", "is_enabled"),
            ("created_at",),
        ]
        ordering = ["-id"]


class ChatAgentProfile(BaseModel):
    owner_account_id = fields.BigIntField(null=True, description="所属账户ID，空表示全局")
    agent_key = fields.CharField(max_length=128, description="agent 唯一键")
    role = fields.CharEnumField(ChatAgentRoleEnum, description="agent 角色")
    name = fields.CharField(max_length=128, description="agent 名称")
    description = fields.TextField(default="", description="agent 描述")
    system_prompt = fields.TextField(default="", description="agent 系统提示词")
    llm_model_config_id = fields.BigIntField(null=True, description="默认模型配置ID")
    default_resource_config = fields.JSONField(default=dict, description="默认资源选择")
    capability_keys = fields.JSONField(default=list, description="默认 capability keys")
    metadata = fields.JSONField(default=dict, description="附加元数据")
    is_enabled = fields.BooleanField(default=True, description="是否启用")
    version = fields.IntField(default=1, description="版本号")

    class Meta:  # type: ignore
        table = "chat_agent_profile"
        table_description = "聊天 agent 定义表"
        app = _KBConnectionName
        unique_together = [("owner_account_id", "agent_key", "deleted_at")]
        indexes = [
            ("owner_account_id", "role", "is_enabled"),
            ("agent_key", "is_enabled"),
            ("created_at",),
        ]
        ordering = ["-id"]


class ChatAgentMount(BaseModel):
    source_agent = fields.ForeignKeyField(
        f"{_KBConnectionName}.ChatAgentProfile",
        related_name="outgoing_mounts",
        on_delete=fields.CASCADE,
        description="源 agent",
    )
    mounted_agent = fields.ForeignKeyField(
        f"{_KBConnectionName}.ChatAgentProfile",
        related_name="incoming_mounts",
        on_delete=fields.CASCADE,
        description="被挂载 agent",
    )
    mode = fields.CharEnumField(ChatAgentMountModeEnum, description="挂载模式")
    purpose = fields.TextField(default="", description="挂载目的")
    trigger_tags = fields.JSONField(default=list, description="触发标签")
    pass_message_history = fields.BooleanField(default=False, description="是否透传历史消息")
    pass_deps_fields = fields.JSONField(default=list, description="透传依赖字段")
    output_contract = fields.CharField(max_length=255, null=True, description="输出契约")
    mounted_as_capability = fields.CharField(max_length=128, null=True, description="映射成的 capability key")
    metadata = fields.JSONField(default=dict, description="附加元数据")
    is_enabled = fields.BooleanField(default=True, description="是否启用")

    class Meta:  # type: ignore
        table = "chat_agent_mount"
        table_description = "聊天 agent 挂载关系表"
        app = _KBConnectionName
        unique_together = [
            ("source_agent_id", "mounted_agent_id", "mode", "mounted_as_capability", "deleted_at"),
        ]
        indexes = [
            ("source_agent_id", "is_enabled"),
            ("mounted_agent_id", "is_enabled"),
            ("mode", "is_enabled"),
        ]
        ordering = ["-id"]


class ChatConversation(BaseModel):
    user_id = fields.BigIntField(description="所属账户ID", null=True)
    agent_key = fields.CharField(max_length=128, description="默认 agent key", default="orchestrator.default")
    title = fields.CharField(max_length=255, description="会话标题", default="新会话")
    default_resource_config = fields.JSONField(default=dict, description="默认资源选择")

    class Meta:  # type: ignore
        table = "chat_conversation"
        table_description = "聊天会话表"
        app = _KBConnectionName
        indexes = [
            ("user_id", "agent_key"),
            ("agent_key",),
        ]
        ordering = ["-id"]


class ChatTurn(BaseModel):
    conversation = fields.ForeignKeyField(
        f"{_KBConnectionName}.ChatConversation",
        related_name="turns",
        on_delete=fields.CASCADE,
        description="所属会话",
    )
    seq = fields.IntField(description="会话内顺序号")
    agent_key = fields.CharField(max_length=128, description="执行 turn 的 agent key", default="orchestrator.default")
    status = fields.CharEnumField(
        ChatTurnStatusEnum,
        default=ChatTurnStatusEnum.running.value,
        description="turn状态",
    )
    trigger = fields.CharEnumField(
        ChatTurnTriggerEnum,
        default=ChatTurnTriggerEnum.user.value,
        description="触发方式",
    )
    request_id = fields.UUIDField(null=True, description="客户端请求ID")
    input_root_data_id = fields.BigIntField(null=True, description="输入根数据ID")
    output_root_data_id = fields.BigIntField(null=True, description="输出根数据ID")
    resource_selection = fields.JSONField(default=dict, description="资源选择")
    planner_mode = fields.CharField(max_length=32, null=True, description="规划模式")
    planner_summary = fields.CharField(max_length=1000, null=True, description="规划摘要")
    candidate_snapshot = fields.JSONField(default=list, description="候选能力快照")
    execution_plan = fields.JSONField(default=dict, description="执行计划快照")
    started_at = fields.DatetimeField(null=True, description="开始时间")
    finished_at = fields.DatetimeField(null=True, description="结束时间")
    usage = fields.JSONField(default=dict, description="模型/检索使用统计")

    class Meta:  # type: ignore
        table = "chat_turn"
        table_description = "聊天轮次表"
        app = _KBConnectionName
        unique_together = [("conversation_id", "seq")]
        indexes = [
            ("conversation_id", "status"),
            ("conversation_id", "request_id"),
            ("agent_key",),
        ]
        ordering = ["-id"]


class ChatStep(BaseModel):
    conversation = fields.ForeignKeyField(
        f"{_KBConnectionName}.ChatConversation",
        related_name="steps",
        on_delete=fields.CASCADE,
        description="所属会话",
    )
    turn = fields.ForeignKeyField(
        f"{_KBConnectionName}.ChatTurn",
        related_name="steps",
        on_delete=fields.CASCADE,
        description="所属turn",
    )
    parent_step = fields.ForeignKeyField(
        f"{_KBConnectionName}.ChatStep",
        related_name="children",
        null=True,
        on_delete=fields.SET_NULL,
        description="父步骤",
    )
    kind = fields.CharEnumField(
        ChatStepKindEnum,
        default=ChatStepKindEnum.system.value,
        description="步骤类型",
    )
    capability_key = fields.CharField(max_length=128, null=True, description="所属能力 key")
    operation_key = fields.CharField(max_length=128, null=True, description="计划操作 key")
    name = fields.CharField(max_length=100, description="步骤名称")
    status = fields.CharEnumField(
        ChatStepStatusEnum,
        default=ChatStepStatusEnum.running.value,
        description="步骤状态",
    )
    sequence = fields.IntField(default=0, description="回放顺序")
    metadata = fields.JSONField(default=dict, description="步骤元数据")
    started_at = fields.DatetimeField(null=True, description="开始时间")
    finished_at = fields.DatetimeField(null=True, description="结束时间")

    class Meta:  # type: ignore
        table = "chat_step"
        table_description = "聊天步骤表"
        app = _KBConnectionName
        indexes = [
            ("conversation_id", "turn_id", "sequence"),
            ("turn_id", "sequence"),
            ("parent_step_id",),
            ("status",),
        ]
        ordering = ["id"]


class ChatData(BaseModel):
    conversation = fields.ForeignKeyField(
        f"{_KBConnectionName}.ChatConversation",
        related_name="data_items",
        on_delete=fields.CASCADE,
        description="所属会话",
    )
    turn = fields.ForeignKeyField(
        f"{_KBConnectionName}.ChatTurn",
        related_name="data_items",
        on_delete=fields.CASCADE,
        description="所属turn",
    )
    step = fields.ForeignKeyField(
        f"{_KBConnectionName}.ChatStep",
        related_name="data_items",
        on_delete=fields.CASCADE,
        description="所属step",
    )
    kind = fields.CharEnumField(
        ChatDataKindEnum,
        default=ChatDataKindEnum.output.value,
        description="数据类别",
    )
    payload_type = fields.CharField(max_length=64, description="载荷类型")
    payload = fields.JSONField(default=dict, description="数据载荷")

    class Meta:  # type: ignore
        table = "chat_data"
        table_description = "聊天数据工件表"
        app = _KBConnectionName
        indexes = [
            ("conversation_id", "turn_id", "kind"),
            ("step_id",),
            ("turn_id", "payload_type"),
        ]
        ordering = ["id"]
