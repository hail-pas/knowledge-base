"""
文件处理工作流模块

提供文件处理相关的 Celery 任务和工作流定义
支持文件获取、加载、分割、索引和摘要生成等操作
"""

from .tasks import (
    # Pydantic Models
    FileInput,
    FileOutput,
    LoadOutput,
    SplitOutput,
    IndexOutput,
    SummaryOutput,
    # Celery Tasks
    fetch_file_task,
    load_file_task,
    split_text_task,
    index_into_milvus_task,
    index_into_es_task,
    summary_task,
    # Task Registry
    TASK_REGISTRY,
    get_task_by_name,
)

__all__ = [
    # Pydantic Models
    "FileInput",
    "FileOutput",
    "LoadOutput",
    "SplitOutput",
    "IndexOutput",
    "SummaryOutput",
    # Celery Tasks
    "fetch_file_task",
    "load_file_task",
    "split_text_task",
    "index_into_milvus_task",
    "index_into_es_task",
    "summary_task",
    # Task Registry
    "TASK_REGISTRY",
    "get_task_by_name",
]

# Default workflow configuration for file processing
DEFAULT_FILE_PROCESS_WORKFLOW = {
    "fetch_file": {
        "options": {},
        "execute_params": {},
        "depends_on": [],
        "input": {}
    },
    "load_file": {
        "options": {},
        "execute_params": {},
        "depends_on": ["fetch_file"],
        "input": {}
    },
    "split_text": {
        "options": {},
        "execute_params": {},
        "depends_on": ["load_file"],
        "input": {}
    },
    "index_into_milvus": {
        "options": {},
        "execute_params": {},
        "depends_on": ["split_text"],
        "input": {}
    },
    "index_into_es": {
        "options": {},
        "execute_params": {},
        "depends_on": ["split_text"],
        "input": {}
    },
    "summary": {
        "options": {},
        "execute_params": {},
        "depends_on": ["load_file"],
        "input": {}
    },
}
