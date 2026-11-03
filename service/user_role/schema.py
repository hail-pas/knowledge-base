from pydantic import BaseModel, Field


class UserRoleReplace(BaseModel):
    role_ids: list[int] = Field(default_factory=list, description="需要分配给该 Pero 用户的角色ID")
