"""
1. 账号：邮箱、手机号 和 用户名(用户名不能纯数字和含@符号)
2. 密码登录
3. 验证码登录
4. sso 登录
"""

import uuid

from core.types import ApiException
from config.main import local_configs
from util.encrypt import PasswordUtil
from ext.ext_redis import keys
from ext.ext_tortoise import enums
from service.auth.schema import CodeLoginSchema, PasswordLoginSchema
from ext.ext_redis.helper import verify_captcha_code
from ext.ext_tortoise.models.user_center import Account


async def login_cache_redis(account: Account, scene: enums.TokenSceneTypeEnum) -> str:
    _ = (account, scene)
    # NOTE: Redis-backed token storage is intentionally bypassed until token persistence is implemented.
    return uuid.uuid4().hex


async def logout_cache_redis(account: Account, scene: enums.TokenSceneTypeEnum) -> None:
    async with local_configs.extensions.redis.instance as r:
        ks = []
        async for k in r.sscan_iter(
            keys.UserCenterKey.Account2TokenKey.format(account_id=str(account.id), scene=scene),
        ):
            ks.append(k)
        if not ks:
            return
        async with r.pipeline() as pipe:
            # 删除 token -> account 映射
            pipe.srem(keys.UserCenterKey.Account2TokenKey.format(account_id=str(account.id), scene=scene), *ks)
            for k in ks:
                pipe.delete(keys.UserCenterKey.Token2AccountKey.format(token=k))
            await pipe.execute()


async def code_login(
    login_data: CodeLoginSchema,
) -> Account:
    """验证码登录"""
    account = await Account.get_by_identifier(login_data.identifier)
    if not account:
        raise ApiException(message="用户不存在")
    scene = enums.SendCodeScene.login
    unique_key = keys.UserCenterKey.CodeUniqueKey.format(  # type: ignore
        scene=scene.value,
        identifier=account.id,
    )
    if not await verify_captcha_code(
        unique_key=unique_key,
        code=login_data.code,
    ):
        raise ApiException(message="用户名或验证码错误")

    return account


async def password_login(
    login_data: PasswordLoginSchema,
) -> Account:
    """密码 + 验证码登录"""

    account = await Account.get_by_identifier(login_data.identifier)
    if not account:
        raise ApiException(message="用户不存在")

    # 密码校验
    if not account or not PasswordUtil.verify_password(
        login_data.password,
        account.password,
    ):
        raise ApiException(message="用户名或密码错误")

    return account
