from pydantic import BaseModel
from ext.file_source.base import FileSourceAdapter
from ext.file_source.providers.local import LocalAdapter
from ext.file_source.providers.s3 import S3Adapter
from ext.file_source.providers.aliyun_oss import AliyunOSSAdapter
from ext.file_source.providers.sharepoint import SharePointAdapter
from ext.ext_tortoise.models.knowledge_base import FileSource

class FileSourceConfig(BaseModel):
    access_key_id: str
    access_key_secret: str
    endpoint: str
    region: str
    bucket_name: str


class FileSourceAdapterFactory:
    """文件源工厂

    负责根据文件源类型创建对应的适配器实例。
    """

    _adapters: dict[str, type[FileSourceAdapter]] = {}

    @classmethod
    def register(cls, type: str, adapter: type[FileSourceAdapter]) -> None:
        """注册新的文件源类型

        Args:
            type: 文件源类型标识
            adapter: 适配器类
        """
        cls._adapters[type] = adapter

    @classmethod
    def create(cls, source: FileSource) -> FileSourceAdapter:
        """创建适配器实例

        Args:
            source: 文件源配置实例

        Returns:
            适配器实例

        Raises:
            ValueError: 不支持的文件源类型
        """
        adapter_cls = cls._adapters.get(source.type.value)
        if not adapter_cls:
            raise ValueError(f"不支持的文件源类型: {source.type}")
        return adapter_cls(source.config)


# 注册内置适配器（导入时自动注册）
FileSourceAdapterFactory.register("local_file", LocalAdapter)
FileSourceAdapterFactory.register("s3", S3Adapter)
FileSourceAdapterFactory.register("aliyun_oss", AliyunOSSAdapter)
FileSourceAdapterFactory.register("sharepoint", SharePointAdapter)
# 后续扩展：
# FileSourceFactory.register("api", APIAdapter)
