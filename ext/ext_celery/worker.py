"""
Celery Worker 入口文件

启动 worker 时使用此文件作为入口，确保所有任务都被正确注册

启动命令：
    uv run celery -A ext.ext_celery.worker worker -l info
"""

# 导入 workflow 模块，确保所有任务都被注册到 Celery
from ext.workflow import demo_tasks  # Import demo tasks
from ext.workflow import WorkflowScheduler
from ext.ext_celery.app import celery_app
from service.workflow.document import tasks

"""
uv run celery -A ext.ext_celery.worker worker -Q workflow_entry -c 4
uv run celery -A ext.ext_celery.worker worker -Q workflow_handoff -c 1
uv run celery -A ext.ext_celery.worker worker -c 8
"""

# 这些导入会触发任务注册
_ = WorkflowScheduler
_ = demo_tasks
_ = tasks


__all__ = ["celery_app"]
