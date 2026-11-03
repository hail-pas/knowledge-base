from pydantic import BaseModel, ConfigDict

class PermissionList(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    code: str
    label: str
    permission_type: str
    is_deprecated: bool


class PermissionDeprecationUpdate(BaseModel):
    code: str
    is_deprecated: bool
