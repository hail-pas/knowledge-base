"""
Celery Worker 入口文件

启动 worker 时使用此文件作为入口，确保所有任务都被正确注册

启动命令：
    uv run celery -A ext.ext_celery.worker worker -l info
"""
import asyncio
from ext.ext_celery.app import celery_app

# 导入所有任务模块，确保它们被注册到 Celery
from ext.workflow import tasks as workflow_tasks
from ext.workflow import file_process_tasks

# 初始化数据库连接
from core.context import init_ctx, clear_ctx

# 在 worker 启动时初始化数据库
from celery.signals import worker_init, worker_ready, worker_shutdown


# 这些导入会触发任务注册
_ = workflow_tasks
_ = file_process_tasks


@worker_init.connect
def worker_init_handler(**kwargs):
    """Worker 初始化时调用"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(init_ctx())
    finally:
        loop.close()


@worker_ready.connect
def worker_ready_handler(**kwargs):
    """Worker 准备就绪时调用"""
    print("Worker is ready! Database connection initialized.")


@worker_shutdown.connect
def worker_shutdown_handler(**kwargs):
    """Worker 关闭时调用"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(clear_ctx())
    finally:
        loop.close()


__all__ = ['celery_app']
