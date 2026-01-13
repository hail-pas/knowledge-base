import random
import string
import asyncio
from typing import Any
from collections.abc import Callable, Hashable, Iterable, Coroutine


def generate_random_string(
    length: int,
    all_digits: bool = False,
    excludes: list[str] | None = None,
) -> str:
    """生成任意长度字符串."""
    if excludes is None:
        excludes = []
    all_char = string.digits if all_digits else string.ascii_letters + string.digits
    if excludes:
        for char in excludes:
            all_char = all_char.replace(char, "")
    # return "".join(random.sample(all_char, length))
    return "".join(random.SystemRandom().choice(all_char) for _ in range(length))


def await_in_sync(to_await: Coroutine) -> Any:  # ruff: noqa
    """
    同步环境执行异步
    """
    async_response = []

    async def run_and_capture_result() -> None:
        r = await to_await
        async_response.append(r)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    coroutine = run_and_capture_result()
    loop.run_until_complete(coroutine)
    return async_response[0]


def filter_dict(
    dict_obj: dict,
    callback: Callable[[Hashable, Any], bool],
) -> dict:
    """适用于字典的filter."""
    new_dict = {}
    for key, value in dict_obj.items():
        if callback(key, value):
            new_dict[key] = value
    return new_dict


def flatten_list(element: Iterable) -> list[Any]:
    """Iterable 递归展开成一级列表."""
    flat_list = []

    def _flatten_list(e: Any) -> None:
        if type(e) in [list, set, tuple]:
            for item in e:
                _flatten_list(item)
        else:
            flat_list.append(e)

    _flatten_list(element)

    return flat_list
