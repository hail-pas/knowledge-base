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
from ext.ext_redis.helper import verify_captcha_code
from api.service.auth.schema import CodeLoginSchema, PasswordLoginSchema
from ext.ext_tortoise.models.user_center import Account


async def login_cache_redis(account: Account, scene: enums.TokenSceneTypeEnum) -> str:
    token = uuid.uuid4().hex
    return token
    # account_info = AccountRedisInfo.model_validate(account).model_dump()
    perms = await account.get_permission_codes()
    async with local_configs.extensions.redis.instance as r:
        async with r.pipeline() as pipe:
            # 设置 token -> account 映射
            pipe.set(
                keys.UserCenterKey.Token2AccountKey.format(token=token),
                value=f"{str(account.id)}:{scene.value}",
                ex=local_configs.server.token_expire_seconds,
            )
            # 设置 account -> token 映射
            pipe.sadd(keys.UserCenterKey.Account2TokenKey.format(account_id=str(account.id), scene=scene), token)
            # 设置 account -> token 过期时间
            pipe.expire(
                keys.UserCenterKey.Account2TokenKey.format(account_id=str(account.id), scene=scene),
                local_configs.server.token_expire_seconds,
            )

            # 权限
            pipe.delete(
                keys.UserCenterKey.AccountApiPermissionSet.format(  # type: ignore
                    uuid=str(account.id),
                ),
            )
            if perms:
                # 刷新接口权限
                pipe.sadd(
                    keys.UserCenterKey.AccountApiPermissionSet.format(  # type: ignore
                        uuid=str(account.id),
                    ),
                    *perms,  # type: ignore
                )
            await pipe.execute()
    return token


async def logout_cache_redis(account: Account, scene: enums.TokenSceneTypeEnum):
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
