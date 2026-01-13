import time
from uuid import UUID
from typing import Annotated

from loguru import logger
from fastapi import Header, Depends, Request
from cachetools import TTLCache
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.security.utils import get_authorization_scheme_param

from core.types import ApiException, RequestHeaderKeyEnum
from config.main import local_configs
from util.encrypt import HashUtil
from core.response import ResponseCodeEnum
from ext.ext_tortoise.models.user_center import Account


class TheBearer(HTTPBearer):
    async def __call__(
        self: "TheBearer",
        request: Request,  # WebSocket
    ) -> HTTPAuthorizationCredentials:  # _authorization: Annotated[Optional[str], Depends(oauth2_scheme)]
        authorization: str | None = request.headers.get("Authorization")
        if not authorization:
            raise ApiException(
                code=ResponseCodeEnum.unauthorized,
                message="授权头部缺失",
            )
        scheme, credentials = get_authorization_scheme_param(authorization)
        if not (authorization and scheme and credentials):
            raise ApiException(
                code=ResponseCodeEnum.unauthorized,
                message="授权头无效",
            )
        if scheme != "Bearer" and self.auto_error:
            raise ApiException(
                code=ResponseCodeEnum.unauthorized,
                message="授权头类型错误",
            )
        return HTTPAuthorizationCredentials(
            scheme=scheme,
            credentials=credentials,
        )


auth_schema = TheBearer()

_account_cache = TTLCache(maxsize=256, ttl=60)


async def _get_account_by_id(account_id: UUID) -> Account:
    if account_id in _account_cache:
        return _account_cache[account_id]
    acc = await Account.get_or_none(id=account_id, deleted_at=0)
    if not acc:
        raise ApiException(
            code=ResponseCodeEnum.unauthorized,
            message="Invalid Account",
        )
    _account_cache[account_id] = acc
    return acc


async def _validate_jwt_token(request: Request, token: HTTPAuthorizationCredentials) -> Account:
    async with local_configs.extensions.redis.instance as r:
        token_identifier = await r.get(
            keys.UserCenterKey.Token2AccountKey.format(  # type: ignore
                token=token.credentials,
            ),
        )

        if not token_identifier:
            logger.warning("token缓存失效")
            raise ApiException(
                code=ResponseCodeEnum.unauthorized.value,
                message="登录失效或已在其他地方登录",
            )

        account_id, scene = token_identifier.split(":")

        if request.headers.get(RequestHeaderKeyEnum.front_scene.value) and scene != request.headers.get(
            RequestHeaderKeyEnum.front_scene.value,
        ):
            logger.warning("token场景不匹配")
            raise ApiException(
                code=ResponseCodeEnum.unauthorized.value,
                message="token异常使用",
            )

        account = await _get_account_by_id(account_id)

        if not account:
            logger.warning("token账户不存在")
            raise ApiException(
                code=ResponseCodeEnum.unauthorized.value,
                message="授权头无效",
            )

        # set scope
        request.scope["user"] = account
        request.scope["scene"] = scene
        request.scope["is_staff"] = account.is_staff
        request.scope["is_super_admin"] = account.is_super_admin
        return account


class TokenRequired:
    async def __call__(
        self,
        request: Request,  # WebSocket
        token: Annotated[HTTPAuthorizationCredentials, Depends(auth_schema)],
    ) -> Account:
        return await _validate_jwt_token(
            request,
            token,
        )


token_required = TokenRequired()


class ApiPermissionCheck:
    def __init__(
        self,
    ) -> None:
        pass

    async def __call__(
        self,
        request: Request,
        token: Annotated[HTTPAuthorizationCredentials, Depends(auth_schema)],
    ) -> Account:
        account: Account | None = request.scope.get("user")  # type: ignore
        if not account:
            account = await token_required(request, token)

        if account.is_super_admin:
            return account

        method = request.method
        root_path: str = request.scope["root_path"]
        path: str = request.scope["route"].path

        if await account.has_permission(
            [
                "*",
                f"{request.app.code}:*",
                f"{request.app.code}:{method}:{root_path}{path}",
            ],
        ):
            return account

        raise ApiException(
            code=ResponseCodeEnum.forbidden.value,
            message=ResponseCodeEnum.forbidden.label,
        )


api_permission_check = ApiPermissionCheck()


class SuperAdminRequired:
    def __init__(
        self,
    ) -> None:
        pass

    async def __call__(
        self,
        request: Request,
        token: Annotated[HTTPAuthorizationCredentials, Depends(auth_schema)],
    ) -> Account:
        account: Account | None = request.scope.get("user")  # type: ignore
        if not account:
            account = await token_required(request, token)

        if not account.is_super_admin:
            raise ApiException(
                code=ResponseCodeEnum.forbidden.value,
                message=ResponseCodeEnum.forbidden.label,
            )
        return account


super_admin_required = SuperAdminRequired()


class StaffAdminRequired:
    def __init__(
        self,
    ) -> None:
        pass

    async def __call__(
        self,
        request: Request,
        token: Annotated[HTTPAuthorizationCredentials, Depends(auth_schema)],
    ):
        account: Account | None = request.scope.get("user")  # type: ignore
        if not account:
            account = await token_required(request, token)

        if not account.is_staff:
            raise ApiException(
                code=ResponseCodeEnum.forbidden.value,
                message=ResponseCodeEnum.forbidden.label,
            )
        return account


staff_admin_required = StaffAdminRequired()


class ApiKeyPermissionCheck:
    """外部api key权限校验"""

    async def __call__(
        self,
        request: Request,
        x_api_key: str = Header(
            description="ApiKey",
        ),
        x_timestamp: int = Header(
            description="请求时间戳, 秒级时间戳, 允许误差+30s",
        ),
        x_sign: str = Header(
            description='签名, 生成: hmac_sha256(secret_key, "api_key&timestamp")',
        ),
    ) -> bool:
        if abs(int(time.time()) - int(x_timestamp)) > 30:
            raise ApiException(
                message="请求时间戳过期",
                code=ResponseCodeEnum.unauthorized.value,
            )

        redis_api_secret_key = keys.RedisCacheKey.ApiSecretKey.format(  # type: ignore
            api_key=x_api_key,
        )
        redis_perm_key = keys.RedisCacheKey.ApiKeyPermissionSet.format(  # type: ignore
            api_key=x_api_key,
        )

        async with local_configs.extensions.redis.instance as r:
            async with r.pipeline() as pipe:
                pipe.get(redis_api_secret_key)
                pipe.smismember(
                    name=redis_perm_key,
                    values=[
                        f"{request.app.code}:*",
                        f'{request.app.code}:{request.method}:{request.scope["root_path"]}{request.scope["route"].path}',
                    ],
                )
                secret_key, is_ok = await pipe.execute()
        if not secret_key:
            raise ApiException(
                message="无效的ApiKey",
                code=ResponseCodeEnum.unauthorized.value,
            )

        result_sign = HashUtil.hmac_sha256_encode(
            k=secret_key,
            s=f"{x_api_key}&{x_timestamp}",
        )
        if result_sign != x_sign:
            raise ApiException(
                message="签名错误",
                code=ResponseCodeEnum.unauthorized.value,
            )

        # 如果请求的接口，不在权限集合里
        if not any(is_ok):
            raise ApiException(
                message="禁止访问",
                code=ResponseCodeEnum.forbidden.value,
            )
        request.scope["scene"] = "ApiCall"
        request.scope["is_staff"] = False
        request.scope["is_super_admin"] = False
        request.scope["is_trial"] = False

        return True


external_api_key_permission_check = ApiKeyPermissionCheck()
