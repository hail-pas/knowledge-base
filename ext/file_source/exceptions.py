"""
FileSource Exceptions - 文件源异常类定义
"""


class FileSourceError(Exception):
    """文件源基础异常类"""

    pass


class FileSourceConfigError(FileSourceError):
    """文件源配置错误

    当配置无效、未启用或缺少必要参数时抛出
    """

    pass


class FileSourceTypeError(FileSourceError):
    """文件源类型错误

    当尝试创建不支持的文件源类型时抛出
    """

    pass


class FileSourceNotFoundError(FileSourceError):
    """文件未找到错误

    当尝试访问不存在的文件时抛出
    """

    pass


class FileUploadError(FileSourceError):
    """文件上传错误

    当文件上传失败时抛出
    """

    pass


class FileDownloadError(FileSourceError):
    """文件下载错误

    当文件下载失败时抛出
    """

    pass


class FileDeleteError(FileSourceError):
    """文件删除错误

    当文件删除失败时抛出
    """

    pass


class FilePermissionError(FileSourceError):
    """文件权限错误

    当没有足够的权限访问文件时抛出
    """

    pass


class FileConnectionError(FileSourceError):
    """文件源连接错误

    当无法连接到文件源时抛出
    """

    pass
