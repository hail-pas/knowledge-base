import atexit
import pdb  # noqa
import cProfile
import importlib

from IPython import start_ipython
from traitlets.config import Config


def sync_clear_ctx():
    import asyncio
    from core.context import clear_ctx
    loop = asyncio.get_event_loop()
    loop.run_until_complete(clear_ctx())

atexit.register(sync_clear_ctx)

if __name__ == "__main__":

    # models = {cls.__name__: cls for cls in BaseModel.__subclasses__()}
    main = importlib.import_module("__main__")
    ctx = main.__dict__
    ctx.update(
        {
            # **models,
            "ipdb": pdb,
            "cProfile": cProfile,
        },
    )
    conf = Config()
    conf.InteractiveShellApp.exec_lines = [
        "print('System Ready!')",
        "from core.context import init_ctx",
        "await init_ctx()",
    ]
    # DEBUG=10, INFO=20, WARN=30
    conf.InteractiveShellApp.log_level = 30
    conf.TerminalInteractiveShell.loop_runner = "asyncio"
    conf.TerminalInteractiveShell.colors = "neutral"
    conf.TerminalInteractiveShell.autoawait = True
    start_ipython(argv=[], user_ns=ctx, config=conf)
