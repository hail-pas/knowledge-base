import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))  # type: ignore

import asyncio
import argparse

from core.context import ctx
from util.encrypt import PasswordUtil
from ext.ext_tortoise.models.user_center import Role, Account


async def create_super_admin_role():
    return await Role.create(label="Super Admin", remark="Super Admin")


async def create_super_admin(role: Role, username: str, phone: str, email: str, password: str):
    return await Account.create(
        username=username,
        phone=phone,
        email=email,
        password=PasswordUtil.get_password_hash(password),
        role=role,
        is_staff=True,
        is_super_admin=True,
        remark="System created",
        role_id=role.id,
    )


async def main(username: str, phone: str, email: str, password: str) -> None:
    async with ctx():
        role = await create_super_admin_role()
        await create_super_admin(role, username, phone, email, password)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create super admin user")
    parser.add_argument("--username", type=str, default="admin", help="Username for super admin (default: admin)")
    parser.add_argument(
        "--phone",
        type=str,
        default="13800138000",
        help="Phone number for super admin (default: 13800138000)",
    )
    parser.add_argument(
        "--email",
        type=str,
        default="admin@example.com",
        help="Email for super admin (default: admin@example.com)",
    )
    parser.add_argument(
        "--password",
        type=str,
        default="password",
        help="Password for super admin (default: password)",
    )

    args = parser.parse_args()

    asyncio.run(main(args.username, args.phone, args.email, args.password))

    print(">>>>>> Success")
