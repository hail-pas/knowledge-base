"""
Embedding 模块自定义异常类
"""


class EmbeddingError(Exception):
    """Embedding 模块基础异常类"""
    pass


class EmbeddingConfigError(EmbeddingError):
    """Embedding 配置错误"""
    pass


class EmbeddingModelNotFoundError(EmbeddingError):
    """Embedding 模型未找到错误"""
    pass


class EmbeddingAPIError(EmbeddingError):
    """Embedding API 调用错误"""

    def __init__(self, message: str, status_code: int | None = None, response_text: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


class EmbeddingBatchError(EmbeddingError):
    """Embedding 批处理错误"""

    def __init__(self, message: str, failed_indices: list | None = None):
        super().__init__(message)
        self.failed_indices = failed_indices or []


class EmbeddingTimeoutError(EmbeddingError):
    """Embedding 超时错误"""
    pass
