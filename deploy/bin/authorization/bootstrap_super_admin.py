"""Bootstrap a first local RBAC super-admin assignment for a Pero UUID.

This command does not create or modify a Pero account. It only maps the supplied
external UUID to a local super-admin role.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from uuid import UUID

sys.path.append(str(Path(__file__).parent.parent.parent.parent))  # type: ignore

from core.context import ctx
from ext.ext_tortoise.models.authorization import Role, UserRole


async def assign_super_admin(user_id: UUID, role_label: str) -> None:
    async with ctx():
        role = await Role.filter(label=role_label, deleted_at=0).first()
        if role is None:
            role = await Role.create(
                label=role_label,
                remark="Knowledge Base local RBAC super-admin role",
                is_super_admin=True,
            )
        if not role.is_super_admin:
            role.is_super_admin = True
            await role.save(update_fields=["is_super_admin"])
        assignment = await UserRole.filter(
            user_id=user_id,
            role_id=role.id,
            deleted_at=0,
        ).first()
        if assignment is None:
            await UserRole.create(user_id=user_id, role=role)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assign a local super-admin role to a Pero UUID")
    parser.add_argument("--user-id", required=True, help="UUID returned as Pero OIDC sub")
    parser.add_argument("--role-label", default="Knowledge Base 超级管理员")
    args = parser.parse_args()
    asyncio.run(assign_super_admin(UUID(args.user_id), args.role_label))
