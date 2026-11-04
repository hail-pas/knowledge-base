from types import SimpleNamespace
from uuid import UUID

import pytest

from ext.ext_tortoise.enums import (
    PermissionTypeEnum,
    SystemResourceSubTypeEnum,
    SystemResourceTypeEnum,
    TokenSceneTypeEnum,
)
from ext.ext_tortoise.models.authorization import Permission, Resource, Role, UserRole
from service.depend import (
    _get_user_permission_codes as resolve_permission_codes,
    required_permission_code,
)


@pytest.mark.asyncio
async def test_permissions_are_resolved_from_external_uuid_role_mapping():
    """Pero UUID -> local role -> resource -> permission must grant the API code."""

    user_id = UUID("00000000-0000-0000-0000-000000000101")
    role = await Role.create(label="rbac-test-role")
    resource = await Resource.create(
        code="rbac-test-resource",
        label="RBAC test resource",
        resource_type=SystemResourceTypeEnum.menu,
        sub_resource_type=SystemResourceSubTypeEnum.link,
        scene=TokenSceneTypeEnum.web,
    )
    permission = await Permission.create(
        code="KnowledgeBase:GET:/v1/rbac-test",
        label="RBAC test endpoint",
        permission_type=PermissionTypeEnum.api,
    )
    await resource.permissions.add(permission)
    await role.resources.add(resource)
    await UserRole.create(user_id=user_id, role=role)

    permissions = await resolve_permission_codes(user_id)

    assert permission.code in permissions


@pytest.mark.asyncio
async def test_super_admin_role_grants_wildcard_permission():
    """Super-admin is a local Role capability, not a Pero user profile flag."""

    user_id = UUID("00000000-0000-0000-0000-000000000102")
    role = await Role.create(label="rbac-test-super-admin", is_super_admin=True)
    await UserRole.create(user_id=user_id, role=role)

    assert await resolve_permission_codes(user_id) == {"*"}


def test_required_permission_code_uses_mounted_app_route_path():
    request = SimpleNamespace(
        scope={
            "route": SimpleNamespace(path="/v1/rbac-test"),
            "root_path": "/knowledge",
        },
        method="GET",
        app=SimpleNamespace(code="KnowledgeBase"),
        url=SimpleNamespace(path="/knowledge/v1/rbac-test"),
    )

    assert required_permission_code(request) == "KnowledgeBase:GET:/v1/rbac-test"

    request.method = "HEAD"
    assert required_permission_code(request) == "KnowledgeBase:GET:/v1/rbac-test"
