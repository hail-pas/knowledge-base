from core.types import StrEnum


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


class ClientSceneEnum(StrEnum):
    """资源适用的客户端场景"""

    general = ("General", "通用")
    web = ("Web", "网页端")
    ios = ("Ios", "iOS")
    android = ("Android", "Android")
    wmp = ("WMP", "微信小程序")
    unknown = ("Unknown", "未知")


# Backward-compatible name used by existing menu seed data.
TokenSceneTypeEnum = ClientSceneEnum


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
    running = ("running", "运行中")
    completed = ("completed", "已完成")
    failed = ("failed", "失败")
    canceled = ("canceled", "已取消")
    retrying = ("retrying", "重试中")


class EmbeddingModelTypeEnum(StrEnum):
    """Embedding 模型类型"""

    openai = ("openai", "OpenAI")


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
