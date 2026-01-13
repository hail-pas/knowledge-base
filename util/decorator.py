import sys
import threading
from typing import TypeVar, ParamSpec, Concatenate
from functools import wraps
from collections.abc import Callable

P = ParamSpec("P")
T = TypeVar("T")


def singleton(key_generator: Callable[P, str] | None = None):  # type: ignore
    def deco(cls):  # type: ignore
        instances = {}

        def get_instance(*args, **kwargs):  # type: ignore
            key = key_generator(*args, **kwargs) if key_generator else cls  # type: ignore
            if key not in instances:
                instances[key] = cls(*args, **kwargs)
            return instances[key]  # type: ignore

        return get_instance  # type: ignore

    return deco  # type: ignore


def time_limit(
    timeout: int | float,
) -> Callable[[Callable[Concatenate[P], T]], Callable[Concatenate[P], T]]:
    """A decorator to limit a function to `timeout` seconds, raising `TimeoutError`
    if it takes longer.
        >>> import time
        >>> def meaning_of_life():
        ...     time.sleep(.2)
        ...     return 42
        >>>
        >>> time_limit(.1)(meaning_of_life)()
        Traceback (most recent call last):
            ...
        RuntimeError: took too long
        >>> time_limit(1)(meaning_of_life)()
        42
    _Caveat:_ The function isn't stopped after `timeout` seconds but continues
    executing in a separate thread. (There seems to be no way to kill a thread.)
    inspired by <http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/473878>.
    """

    def _1(function: Callable[Concatenate[P], T]) -> Callable[Concatenate[P], T | None]:
        @wraps(function)
        def _2(*args: P.args, **kw: P.kwargs) -> T | None:
            class Dispatch(threading.Thread):
                def __init__(self) -> None:
                    threading.Thread.__init__(self)
                    self.result = None
                    self.error = None

                    self.setDaemon(True)
                    self.start()

                def run(self) -> None:
                    try:
                        self.result = function(*args, **kw)  # type: ignore
                    except Exception:
                        self.error = sys.exc_info()  # type: ignore

            c = Dispatch()
            c.join(timeout)
            if c.is_alive():
                raise RuntimeError("took too long")
            if c.error:
                raise c.error[1]  # type: ignore
            return c.result

        return _2

    return _1  # type: ignore
