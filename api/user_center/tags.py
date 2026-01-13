from core.types import StrEnum


class TagsEnum(StrEnum):
    """Tags"""

    root = ("Root", "根目录")
    authorization = ("Authorization", "授权相关")
    account = ("Account", "账户信息管理")
    role = ("Role", "角色管理")
    resource = ("Resource", "资源管理")
    # >> 新增tag
    other = ("Other", "其他")
    # >> 新增tag
