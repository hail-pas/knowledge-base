import asyncio
from config.main import local_configs
from loguru import logger

# 初始化数据库连接
from core.context import init_ctx, clear_ctx

# 在 worker 启动时初始化数据库
from celery.signals import worker_init, worker_ready, worker_shutdown

celery_app = local_configs.extensions.celery.instance


def get_celery_app():
    """获取 Celery 应用实例

    Returns:
        Celery: Celery 应用实例
    """
    return celery_app



@worker_init.connect
def worker_init_handler(**kwargs):
    """Worker 初始化时调用"""
    # 创建并持久化事件循环，不关闭它
    # 数据库连接池需要在整个 worker 生命周期中使用同一个事件循环
    loop = asyncio.get_event_loop()
    loop.run_until_complete(init_ctx())


@worker_ready.connect
def worker_ready_handler(**kwargs):
    """Worker 准备就绪时调用"""
    logger.info("Worker is ready! Database connection initialized.")


@worker_shutdown.connect
def worker_shutdown_handler(**kwargs):
    """Worker 关闭时调用"""
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(clear_ctx())
    finally:
        loop.close()
