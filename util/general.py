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


def truncate_content(content: str | None, truncate: bool = True, max_length: int = 100) -> str:
    """Truncate content for logging

    Args:
        content: Content to truncate
        truncate: Whether to truncate
        max_length: Maximum length before truncation

    Returns:
        Truncated or original content
    """
    if not content:
        return ""

    if not truncate or len(content) <= max_length:
        return content

    return content[:max_length] + f"... (truncated, total {len(content)} chars)"


def format_dict_for_log(data: dict[str, Any] | None, max_items: int = 10, max_value_length: int = 100) -> str:
    """Format dictionary for logging

    Args:
        data: Dictionary to format
        max_items: Maximum number of items to show
        max_value_length: Maximum length for values

    Returns:
        Formatted string representation
    """
    if not data:
        return "{}"

    items = list(data.items())[:max_items]
    formatted = {}

    for key, value in items:
        if isinstance(value, str):
            formatted[key] = truncate_content(value, True, max_value_length)
        elif isinstance(value, dict):
            formatted[key] = format_dict_for_log(value, max_items, max_value_length)
        elif isinstance(value, list):
            formatted[key] = f"[{len(value)} items]"
        else:
            formatted[key] = str(value)[:max_value_length]

    if len(data) > max_items:
        return f"{formatted} ... and {len(data) - max_items} more items"

    return str(formatted)
