from core.types import StrEnum


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
