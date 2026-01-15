from core.types import StrEnum


class TagsEnum(StrEnum):
    """Tags"""

    root = ("Root", "根目录")
    config = ("config", "配置")
    collection = ("collection", "文件集合")
    document = ("document", "文件记录")
    file_source = ("file_source", "文件源配置")
