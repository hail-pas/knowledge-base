from json import load
from posixpath import split

from core.types import IntEnum, StrEnum


class StatusEnum(StrEnum):
    """启用状态"""

    enable = ("enable", "启用")
    disable = ("disable", "禁用")


class PermissionTypeEnum(StrEnum):
    """权限类型"""

    api = ("api", "API")


class SystemResourceTypeEnum(StrEnum):
    """系统资源类型"""

    menu = ("menu", "菜单")
    button = ("button", "按钮")
    api = ("api", "接口")


class SystemResourceSubTypeEnum(StrEnum):
    """系统资源子类型"""

    add_tab = ("add_tab", "选项卡")
    dialog = ("dialog", "弹窗")
    ajax = ("ajax", "Ajax请求")
    link = ("link", "链接")


class TokenSceneTypeEnum(StrEnum):
    """token场景"""

    general = ("General", "通用")
    web = ("Web", "网页端")
    ios = ("Ios", "Ios")
    android = ("Android", "Android")
    wmp = ("WMP", "微信小程序")
    unknown = ("Unknown", "未知")


class SendCodeScene(StrEnum):
    """发送短信场景"""

    login = ("login", "登录")
    reset_password = ("reset_password", "重置密码")
    change_account_phone = ("change_account_phone", "修改账户手机号")


# =============================================================================
# RAG 平台 - 文件源相关枚举
# =============================================================================


class FileSourceTypeEnum(StrEnum):
    """文件源类型"""

    local_file = ("local_file", "本地文件")
    s3 = ("s3", "S3 Compatible")
    minio = ("minio", "MinIO")
    aliyun_oss = ("aliyun_oss", "阿里云 OSS")
    sharepoint = ("sharepoint", "SharePoint")
    api = ("api", "API 接口")


class DocumentStatusEnum(StrEnum):
    """文件状态"""

    pending = ("pending", "待处理")
    processing = ("processing", "处理中")
    success = ("success", "成功")
    failure = ("failure", "失败")


class WorkflowConfigFormatEnum(StrEnum):
    """工作流配置格式"""

    yaml = ("yaml", "YAML格式")
    json = ("json", "JSON格式")
    dict = ("dict", "Python Dict")


class WorkflowStatusEnum(StrEnum):
    """工作流状态"""

    pending = ("pending", "待处理")
    running = ("running", "运行中")
    completed = ("completed", "已完成")
    failed = ("failed", "失败")
    canceled = ("canceled", "已取消")


class ActivityStatusEnum(StrEnum):
    pending = ("pending", "待执行")
    running = ("running", "运行中")  # activity 实际执行时设置
    completed = ("completed", "已完成")
    failed = ("failed", "失败")
    canceled = ("canceled", "已取消")
    retrying = ("retrying", "重试中")


# =============================================================================
# RAG 平台 - Embedding 模型相关枚举
# =============================================================================


class EmbeddingModelTypeEnum(StrEnum):
    """Embedding 模型类型"""

    openai = ("openai", "OpenAI")


# =============================================================================
# RAG 平台 - Indexing 相关枚举
# =============================================================================


# =============================================================================
# RAG 平台 - LLM 模型相关枚举
# =============================================================================


class LLMModelTypeEnum(StrEnum):
    """LLM 模型类型"""

    openai = ("openai", "OpenAI")
    azure_openai = ("azure_openai", "Azure OpenAI")
    deepseek = ("deepseek", "DeepSeek")
    anthropic = ("anthropic", "Anthropic")


class IndexingBackendTypeEnum(StrEnum):
    """Indexing 后端类型"""

    elasticsearch = ("elasticsearch", "Elasticsearch")
    milvus = ("milvus", "Milvus")


class IndexingTypeEnum(StrEnum):
    """索引类型"""

    sparse = ("sparse", "稀疏索引")
    dense = ("dense", "稠密索引")
    hybrid = ("hybrid", "混合索引")


class IndexingStatusEnum(StrEnum):
    """索引状态"""

    creating = ("creating", "创建中")
    active = ("active", "活跃")
    inactive = ("inactive", "未活跃")
    error = ("error", "错误")
    deleting = ("deleting", "删除中")


# =============================================================================
# Chat 模块相关枚举
# =============================================================================


class ChatTurnStatusEnum(StrEnum):
    running = ("running", "运行中")
    completed = ("completed", "已完成")
    failed = ("failed", "失败")
    canceled = ("canceled", "已取消")


class ChatTurnTriggerEnum(StrEnum):
    user = ("user", "用户触发")


class ChatStepStatusEnum(StrEnum):
    running = ("running", "运行中")
    completed = ("completed", "已完成")
    failed = ("failed", "失败")
    canceled = ("canceled", "已取消")


class ChatStepKindEnum(StrEnum):
    system = ("system", "系统步骤")
    retrieval = ("retrieval", "检索步骤")
    llm = ("llm", "模型步骤")
    tool = ("tool", "工具步骤")


class ChatDataKindEnum(StrEnum):
    input = ("input", "输入")
    output = ("output", "输出")


class ChatCapabilityKindEnum(StrEnum):
    skill = ("skill", "流程型能力")
    extension = ("extension", "扩展能力")
    sub_agent = ("sub_agent", "子代理能力")


class ChatCapabilityCategoryEnum(StrEnum):
    core = ("core", "核心能力")
    domain = ("domain", "领域能力")
    infra = ("infra", "基础设施能力")
    agent = ("agent", "代理能力")
    guarded = ("guarded", "受控能力")


class ChatCapabilityRuntimeKindEnum(StrEnum):
    local_toolset = ("local_toolset", "本地工具集")
    mcp_toolset = ("mcp_toolset", "MCP 工具集")
    agent_delegate = ("agent_delegate", "代理委派")
    agent_handoff = ("agent_handoff", "代理切换")


class ChatAgentRoleEnum(StrEnum):
    orchestrator = ("orchestrator", "编排代理")
    specialist = ("specialist", "专家代理")


class ChatAgentMountModeEnum(StrEnum):
    delegate = ("delegate", "委派")
