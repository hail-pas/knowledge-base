"""Authentication boundary for the external Pero user center.

Knowledge Base never creates or manages local users. It validates the incoming
access token through Pero's OIDC UserInfo endpoint and uses the returned
`sub` UUID solely as the owner key for knowledge-base data.
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
    """Minimal external identity required for data ownership."""

    id: UUID


async def _fetch_userinfo(access_token: str) -> dict[str, Any]:
    """Ask Pero to validate the token and return the OIDC subject."""

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
    """Validate Bearer token with Pero and attach its UUID subject to request scope."""

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise ApiException("缺少有效的 Bearer Token", code=401)

    claims = await _fetch_userinfo(credentials.credentials)
    subject = claims.get("sub")
    try:
        user_id = UUID(str(subject))
    except (TypeError, ValueError) as exc:
        raise ApiException("用户中心未返回有效的 UUID subject", code=401) from exc

    user = CurrentUser(id=user_id)
    request.scope["user"] = user
    request.scope["user_id"] = user_id
    return user
