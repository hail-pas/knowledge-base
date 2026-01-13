"""
Indexing 模块自定义异常类

尽量保留各后端的原生错误，仅在必要时转换为自定义异常。
"""


class IndexingError(Exception):
    """Indexing 模块基础异常类"""
    pass


class IndexingConfigError(IndexingError):
    """Indexing 配置错误

    当后端配置无效或缺少必需参数时抛出
    """
    pass


class IndexingBackendError(IndexingError):
    """Indexing 后端错误

    后端服务连接或操作失败时抛出。
    此异常可能包装后端的原生异常，通过 __cause__ 访问原始错误。
    """
    pass


class IndexingQueryError(IndexingError):
    """Indexing 查询错误

    当查询语法错误、查询参数无效或查询执行失败时抛出
    """
    pass


class IndexingDocumentError(IndexingError):
    """Indexing 文档操作错误

    当文档插入、更新、删除操作失败时抛出
    """
    pass


class IndexingIndexError(IndexingError):
    """Indexing 索引操作错误

    当索引创建、删除、schema 操作失败时抛出
    """
    pass


class IndexingModelValidationError(IndexingError):
    """Indexing 模型验证错误

    当 IndexModel 数据验证失败时抛出
    """
    pass
