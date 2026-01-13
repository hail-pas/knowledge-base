import uuid
import random

from PIL import Image
from fastapi import Body, Query, APIRouter
from pydantic import Field, BaseModel
from captcha.image import ImageCaptcha, random_color  # type: ignore
from fastapi.responses import StreamingResponse

from core.types import ApiException
from core.response import Resp
from ext.ext_redis import keys
from constant.regex import EMAIL_REGEX, PHONE_REGEX_CN
from ext.ext_tortoise import enums
from ext.ext_redis.helper import generate_captcha_code
from ext.ext_tortoise.models.user_center import Account

router = APIRouter()


class CaptchaCodeResponse(BaseModel):
    unique_key: str = Field(description="验证码唯一标识")


@router.get(
    "/captcha/image",
    summary="图片验证码",
    description="图片验证码, unique_key附带在响应头中",
    response_class=StreamingResponse,
)
async def captcha_image(
    scene: enums.SendCodeScene = Query(description="场景"),
) -> StreamingResponse:
    identifier = uuid.uuid4()

    class CustomImageCaptcha(ImageCaptcha):
        def generate_image(self, chars: str) -> Image:  # type: ignore
            """Generate the image of the given characters.

            :param chars: text to be generated.
            """
            background = random_color(238, 255)
            color = random_color(10, 200, random.randint(220, 255))
            im = self.create_captcha_image(chars, color, background)
            self.create_noise_dots(im, color, 1, 20)
            self.create_noise_curve(im, color)
            # im = im.filter(ImageFilter.SMOOTH)
            return im  # type: ignore

    image = CustomImageCaptcha(height=80, width=180, font_sizes=(60,))
    unique_key = keys.UserCenterKey.CodeUniqueKey.format(  # type: ignore
        scene=scene.value,
        identifier=identifier,
    )

    # account = await Account.filter(
    #     **{filter_key: identifier},
    #     deleted_at=0,
    # ).first()
    # if account:
    #     unique_key = keys.UserCenterKey.CodeUniqueKey.format(  # type: ignore
    #         scene=scene.value,
    #         identifier=account.id,
    #     )

    code = await generate_captcha_code(
        unique_key=unique_key,
        length=4,
        all_digits=True,
        expire_seconds=1 * 60,
        # excludes=["o", "0", "l"],
    )

    return StreamingResponse(
        content=image.generate(chars=code),
        media_type="image/jpeg",
        headers={"x-unique-key": unique_key},
    )


@router.post(
    "/captcha/code",
    summary="发送验证码",
    description="发送验证码, phone/email + scene组成unique_key",
)
async def captcha_code(
    identifier: str = Body(
        description="唯一标识, 手机号或邮箱",
        max_length=11,
        min_length=11,
    ),
    scene: enums.SendCodeScene = Body(description="场景"),
    extra: dict = Body(default_factory=dict, description="额外参数: change_account_phone需要传 {'account_id': ''}"),
) -> Resp[CaptchaCodeResponse]:
    filter_key = ""
    if len(identifier) == 11 and PHONE_REGEX_CN.match(identifier):
        filter_key = "phone"
    elif EMAIL_REGEX.match(identifier):
        filter_key = "email"
    else:
        raise ApiException(message="ParameterErrorMsg", code=422)

    expire_seconds = 5 * 60

    # 需要先用户存在的情况
    match scene:
        case enums.SendCodeScene.login:
            account = await Account.filter(
                **{filter_key: identifier},
                deleted_at=0,
            ).first()
            if account:
                unique_key = keys.UserCenterKey.CodeUniqueKey.format(  # type: ignore
                    scene=scene.value,
                    identifier=account.id,
                )
            else:
                return Resp.fail(message="账号未注册")
            if filter_key == "email":
                return Resp.fail(message="暂不支持邮箱验证码")

        case enums.SendCodeScene.reset_password:
            account = await Account.filter(
                **{filter_key: identifier},
                deleted_at=0,
            ).first()
            if not account:
                return Resp.fail(message="账号未注册")

            unique_key = keys.UserCenterKey.CodeUniqueKey.format(  # type: ignore
                scene=scene.value,
                identifier=str(account.id),
            )

        case enums.SendCodeScene.change_account_phone:
            if filter_key != "phone":
                return Resp.fail(message="暂不支持邮箱修改试用账号")

            if not extra:
                return Resp.fail(message="缺少参数")
            account_id = extra.get("account_id")
            if not account_id:
                return Resp.fail(message="缺少AccountID参数")
            try:
                account_id = uuid.UUID(account_id)
            except ValueError:
                return Resp.fail(message="AccountID参数错误")
            account = await Account.filter(
                id=account_id,
            ).first()
            if not account:
                return Resp.fail(message="未找到账号信息")
            unique_key = keys.UserCenterKey.CodeUniqueKey.format(  # type: ignore
                scene=scene.value,
                identifier=f"{extra['account_id']}:{identifier}",
            )

        case _:
            raise ApiException(message="ParameterErrorMsg", code=422)

    code = await generate_captcha_code(
        unique_key=unique_key,
        all_digits=True,
        length=6,
        expire_seconds=expire_seconds,
    )

    # tpl_config = getattr(get_sms_config(), scene.value)
    # if not tpl_config:
    #     raise ApiException(message="未找到对应场景的模版配置")

    # resp = await sms_tpl_single_send_v2(
    #     data=SmsTplSingleSendV2Schema(
    #         mobile=identifier,
    #         tpl_value={"code": code},
    #         **tpl_config.model_dump(by_alias=True),
    #     ),
    # )
    # if not resp.success:
    #     if resp.code == 53:
    #         return Resp.fail(message="获取验证码太频繁, 请30s后再试")
    #     return Resp.fail(
    #         message=(resp.data.detail if hasattr(resp.data, "detail") else resp.message) or "获取验证码失败，未知错误",
    #     )
    return Resp[CaptchaCodeResponse](data=CaptchaCodeResponse(unique_key=unique_key))
