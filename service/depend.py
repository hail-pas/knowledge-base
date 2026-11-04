"""External identity validation and local RBAC authorization.

Pero owns user accounts and access tokens. Knowledge Base stores only role,
permission and resource definitions plus UUID-to-role assignments.
"""

from dataclasses import dataclass
from typing import Annotated, Any
from uuid import UUID

import httpx
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config.main import local_configs
from core.types import ApiException

_bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class CurrentUser:
    """The only identity attribute Knowledge Base needs."""

    id: UUID


async def _fetch_userinfo(access_token: str) -> dict[str, Any]:
    """Validate the access token through Pero's OIDC UserInfo endpoint."""

    config = local_configs.user_center
    url = f"{str(config.base_url).rstrip('/')}{config.userinfo_path}"
    try:
        response = await local_configs.extensions.httpx.instance.get(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=config.timeout_seconds,
        )
    except httpx.TimeoutException as exc:
        raise ApiException("用户中心认证超时，请稍后重试", code=503) from exc
    except httpx.HTTPError as exc:
        raise ApiException("用户中心认证服务暂不可用", code=503) from exc

    if response.status_code == 401:
        raise ApiException("访问令牌无效或已过期", code=401)
    if response.status_code == 403:
        raise ApiException("访问令牌缺少 openid scope", code=403)
    if response.status_code >= 500:
        raise ApiException("用户中心认证服务暂不可用", code=503)
    if response.status_code != 200:
        raise ApiException("用户中心无法验证当前访问令牌", code=401)

    try:
        payload = response.json()
    except ValueError as exc:
        raise ApiException("用户中心返回了无效的身份信息", code=502) from exc
    if not isinstance(payload, dict):
        raise ApiException("用户中心返回了无效的身份信息", code=502)
    return payload


async def require_user(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
) -> CurrentUser:
    """Resolve the current Pero UUID and attach it to request scope."""

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise ApiException("缺少有效的 Bearer Token", code=401)

    claims = await _fetch_userinfo(credentials.credentials)
    try:
        user_id = UUID(str(claims.get("sub")))
    except (TypeError, ValueError) as exc:
        raise ApiException("用户中心未返回有效的 UUID subject", code=401) from exc

    user = CurrentUser(id=user_id)
    request.scope["user"] = user
    request.scope["user_id"] = user_id
    return user


async def _get_user_permission_codes(user_id: UUID) -> set[str]:
    """Resolve API permission codes through user-role-resource-permission links."""

    from ext.ext_tortoise.models.authorization import Permission, Resource, Role, UserRole

    role_ids = await UserRole.filter(user_id=user_id, deleted_at=0).values_list("role_id", flat=True)
    if not role_ids:
        return set()

    roles = await Role.filter(id__in=role_ids, deleted_at=0)
    active_role_ids = [role.id for role in roles]
    if not active_role_ids:
        return set()
    if any(role.is_super_admin for role in roles):
        return {"*"}

    resource_ids = await Resource.filter(
        roles__id__in=active_role_ids,
        enabled=True,
        assignable=True,
    ).values_list("id", flat=True)
    if not resource_ids:
        return set()

    codes = await Permission.filter(
        resources__id__in=resource_ids,
        is_deprecated=False,
    ).distinct().values_list("code", flat=True)
    return set(codes)


async def is_super_admin_user(user_id: UUID) -> bool:
    """Check the local super-admin role flag for a verified Pero UUID."""

    from ext.ext_tortoise.models.authorization import Role, UserRole

    role_ids = await UserRole.filter(user_id=user_id, deleted_at=0).values_list("role_id", flat=True)
    if not role_ids:
        return False
    return await Role.filter(id__in=role_ids, deleted_at=0, is_super_admin=True).exists()


async def ensure_super_admin(user_id: UUID) -> None:
    """Require the root RBAC role for authorization-model mutations."""
    if not await is_super_admin_user(user_id):
        raise ApiException("只有系统超级管理员可以管理 RBAC 配置", code=403)


class ApiPermissionCheck:
    """Require a valid Pero identity and a matching locally assigned permission."""

    async def __call__(
        self,
        request: Request,
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
    ) -> CurrentUser:
        user = request.scope.get("user")
        if not isinstance(user, CurrentUser):
            user = await require_user(request, credentials)

        route = request.scope.get("route")
        route_path = getattr(route, "path", request.url.path)
        root_path = request.scope.get("root_path", "")
        required_code = f"{request.app.code}:{request.method}:{root_path}{route_path}"

        permissions = await _get_user_permission_codes(user.id)
        if "*" in permissions or f"{request.app.code}:*" in permissions or required_code in permissions:
            return user

        raise ApiException("当前用户没有访问该接口的权限", code=403)


api_permission_check = ApiPermissionCheck()
