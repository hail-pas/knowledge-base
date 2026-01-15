"""
Celery Worker 入口文件

启动 worker 时使用此文件作为入口，确保所有任务都被正确注册

启动命令：
    uv run celery -A ext.ext_celery.worker worker -l info
"""
from ext.ext_celery.app import celery_app

# 导入所有任务模块，确保它们被注册到 Celery
from ext.workflow import tasks as workflow_tasks
from ext.workflow import file_process_tasks


# 这些导入会触发任务注册
_ = workflow_tasks
_ = file_process_tasks


__all__ = ['celery_app']
