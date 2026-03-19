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


class ChatConversationStatusEnum(StrEnum):
    active = ("active", "活跃")
    archived = ("archived", "已归档")


class ChatTurnStatusEnum(StrEnum):
    pending = ("pending", "待处理")
    accepted = ("accepted", "已受理")
    running = ("running", "运行中")
    streaming = ("streaming", "流式输出中")
    completed = ("completed", "已完成")
    failed = ("failed", "失败")
    canceled = ("canceled", "已取消")
    rolled_back = ("rolled_back", "已回退")


class ChatTurnTriggerEnum(StrEnum):
    user = ("user", "用户触发")
    system = ("system", "系统触发")
    retry = ("retry", "重试")
    resume = ("resume", "恢复")
    rollback = ("rollback", "回退")


class ChatStepStatusEnum(StrEnum):
    pending = ("pending", "待处理")
    running = ("running", "运行中")
    streaming = ("streaming", "流式输出中")
    completed = ("completed", "已完成")
    failed = ("failed", "失败")
    canceled = ("canceled", "已取消")


class ChatStepKindEnum(StrEnum):
    system = ("system", "系统步骤")
    retrieval = ("retrieval", "检索步骤")
    llm = ("llm", "模型步骤")
    tool = ("tool", "工具步骤")
    guardrail = ("guardrail", "护栏步骤")
    response = ("response", "响应步骤")


class ChatDataKindEnum(StrEnum):
    input = ("input", "输入")
    output = ("output", "输出")
    intermediate = ("intermediate", "中间数据")
    reference = ("reference", "引用数据")
    control = ("control", "控制数据")


class ChatCapabilityKindEnum(StrEnum):
    skill = ("skill", "流程型能力")
    extension = ("extension", "扩展能力")
    sub_agent = ("sub_agent", "子代理能力")
