from core.types import StrEnum


class TagsEnum(StrEnum):
    """Tags"""

    root = ("Root", "根目录")
    config = ("config", "配置")
    collection = ("collection", "文件集合")
    document = ("document", "文件记录")
    file_source = ("file_source", "文件源配置")
    chat = ("chat", "聊天")
    role = ("role", "角色管理")
    resource = ("resource", "菜单与资源管理")
    permission = ("permission", "接口权限管理")
    user_role = ("user-role", "用户角色分配")
