"""
Workflow 模块

提供通用工作流调度系统，基于 Celery 任务和 DAG 调度

核心组件：
- WorkflowManager: 工作流管理器，提供 CRUD 和状态管理
- ActivityTaskTemplate: 任务模板基类，用户只需实现 execute 方法
- activity_task: 任务装饰器，将任务类转换为 Celery 任务
- 调度任务: schedule_workflow_start, schedule_workflow_resume, schedule_activity_handoff
- 文件处理工作流: 完整的文件处理示例

使用方式：
    from ext.workflow import WorkflowManager, ActivityTaskTemplate, activity_task
    from ext.workflow.tasks import schedule_workflow_start

    # 定义自定义任务
    class MyTask(ActivityTaskTemplate):
        async def execute(self) -> Dict[str, Any]:
            return {"result": "success"}

    # 注册任务
    my_task = activity_task(MyTask)

    # 启动工作流
    workflow_uid = schedule_workflow_start(
        config={"task1": {"execute_params": {"task_name": my_task.name}, "depends_on": []}},
        config_format="dict"
    )
"""

# 核心组件
from .manager import WorkflowManager
from .template import ActivityTaskTemplate, activity_task

# 调度任务（支持直接调用和 Celery apply_async）
from .tasks import (
    schedule_workflow_start,
    schedule_workflow_resume,
    schedule_activity_handoff,
)

# 文件处理工作流（示例实现）
from .file_process_tasks import (
    # 任务类
    FetchFileTask,
    LoadFileTask,
    ReplaceContentTask,
    SplitTextTask,
    IndexIntoMilvusTask,
    IndexIntoEsTask,
    SummaryTask,
    # 任务函数
    fetch_file_task,
    load_file_task,
    replace_content_task,
    split_text_task,
    index_into_milvus_task,
    index_into_es_task,
    summary_task,
    # 工作流配置
    FILE_PROCESS_WORKFLOW,
    # 任务注册表
    TASK_REGISTRY,
    get_task_by_name,
)

__all__ = [
    # 核心组件
    "WorkflowManager",
    "ActivityTaskTemplate",
    "activity_task",
    # 调度任务
    "schedule_workflow_start",
    "schedule_workflow_resume",
    "schedule_activity_handoff",
    # 文件处理任务类
    "FetchFileTask",
    "LoadFileTask",
    "ReplaceContentTask",
    "SplitTextTask",
    "IndexIntoMilvusTask",
    "IndexIntoEsTask",
    "SummaryTask",
    # 文件处理任务函数
    "fetch_file_task",
    "load_file_task",
    "replace_content_task",
    "split_text_task",
    "index_into_milvus_task",
    "index_into_es_task",
    "summary_task",
    # 工作流配置
    "FILE_PROCESS_WORKFLOW",
    # 任务注册表
    "TASK_REGISTRY",
    "get_task_by_name",
]
