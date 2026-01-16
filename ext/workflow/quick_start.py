#!/usr/bin/env python3
"""
快速启动脚本

快速启动并运行一个简单的工作流示例
无需配置，开箱即用
"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.context import ctx
from ext.workflow import WorkflowManager
from ext.workflow.tasks import schedule_workflow_start, schedule_workflow_resume
from ext.ext_tortoise.models.knowledge_base import Workflow
from ext.ext_tortoise.enums import WorkflowStatusEnum
from loguru import logger


def setup_logger():
    """配置简单的日志"""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>",
        level="INFO",
    )


def create_temp_file(content: str) -> str:
    """创建临时文件"""
    fd, path = tempfile.mkstemp(suffix=".txt", text=True)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write(content)
    return path


async def quick_start():
    """快速启动工作流"""
    setup_logger()

    logger.info("=" * 60)
    logger.info("Workflow Quick Start")
    logger.info("=" * 60)

    # 1. 创建临时文件
    sample_text = """Hello, Workflow System!
This is a quick demonstration of the workflow engine.
The system will process this file through multiple tasks:
1. Fetch file
2. Load and analyze it
3. Generate a summary
"""
    file_path = create_temp_file(sample_text)
    logger.success(f"✓ Created sample file: {file_path}")

    # 2. 配置简单的串行工作流
    workflow_config = {
        "fetch_file": {
            "input": {"file_path": file_path},
            "execute_params": {"task_name": "workflow_activity.FetchFileTask"},
            "depends_on": []
        },
        "load_file": {
            "execute_params": {"task_name": "workflow_activity.LoadFileTask"},
            "depends_on": ["fetch_file"]
        },
        "replace_content": {
            "execute_params": {"task_name": "workflow_activity.ReplaceContentTask"},
            "depends_on": ["load_file"],
            "input": {"replace_rules": []}
        },
        "summary": {
            "execute_params": {"task_name": "workflow_activity.SummaryTask"},
            "depends_on": ["replace_content"],
            "input": {"max_length": 100}
        }
    }

    logger.success("✓ Workflow configured (4 tasks)")
    logger.info("  Task flow: fetch_file → load_file → replace_content → summary")

    # 3. 启动工作流
    logger.info("\n" + "=" * 60)
    logger.info("Starting Workflow")
    logger.info("=" * 60)

    logger.info("\n🚀 Starting workflow...")
    logger.info("Note: Celery Worker must be running in another terminal")
    logger.info("If workflow stalls, check if worker is running:")
    logger.info("  uv run celery -A ext.ext_celery.worker worker -l info\n")
    try:

        import uuid
        workflow_uid = uuid.uuid4()

        # 创建工作流记录
        workflow = await Workflow.create(
            uid=workflow_uid,
            config=workflow_config,
            config_format="dict",
            status=WorkflowStatusEnum.pending.value,
        )

        await schedule_workflow_start(
            workflow_uid=workflow_uid,
            config=workflow_config,
            config_format="dict",
            initial_inputs={},
            use_async=False
        )

        logger.success(f"✓ Workflow started: {workflow_uid}")

        # workflow_uid = await schedule_workflow_resume(workflow_uid, use_async=False)
    except Exception as e:
        logger.error(f"\n✗ Failed to start workflow: {e}")
        logger.error("\n" + "=" * 60)
        logger.error("Troubleshooting")
        logger.error("=" * 60)
        logger.error("\n1. Check if Celery Worker is running:")
        logger.error("   Open another terminal and run:")
        logger.error("   uv run celery -A ext.ext_celery.worker worker -l info")
        logger.error("\n2. Check Redis connection:")
        logger.error("   redis-cli ping")
        logger.error("\n3. Check PostgreSQL connection:")
        logger.error("   pg_isready")
        logger.error("\n4. View detailed logs:")
        logger.error("   tail -f workflow_quick_start.log")
        logger.error("\n" + "=" * 60)
        return

    # 4. 等待完成
    logger.info("\n⏳ Waiting for workflow to complete...")
    logger.info("Press Ctrl+C to stop waiting\n")

    max_wait = 60  # 最多等待 60 秒
    for i in range(max_wait):
        await asyncio.sleep(1)

        workflow = await WorkflowManager.get_workflow_by_uid(workflow_uid)
        if not workflow:
            logger.error("✗ Workflow not found")
            return

        if workflow.status.value in ["completed", "failed"]:
            break

        # 每 5 秒显示一次进度
        if (i + 1) % 5 == 0:
            activities = await WorkflowManager.get_activities_by_workflow(workflow_uid)
            completed = sum(1 for a in activities if a.status.value == "completed")
            total = len(activities)
            logger.info(f"  Progress: {completed}/{total} tasks completed")

    # 5. 显示结果
    logger.info("\n" + "=" * 60)
    logger.info("Workflow Result")
    logger.info("=" * 60)

    workflow = await WorkflowManager.get_workflow_by_uid(workflow_uid)
    logger.info(f"\nStatus: {workflow.status.value}")

    if workflow.status.value == "completed":
        logger.success("✅ Workflow completed successfully!")

        # 显示每个任务的结果
        activities = await WorkflowManager.get_activities_by_workflow(workflow_uid)
        logger.info("\nTask Results:")

        for activity in activities:
            if activity.output:
                logger.info(f"\n  📋 {activity.name}:")
                output = activity.output
                for key, value in output.items():
                    if key == "metadata":
                        continue
                    if isinstance(value, str) and len(value) > 100:
                        logger.info(f"     {key}: {value[:100]}...")
                    else:
                        logger.info(f"     {key}: {value}")
    else:
        logger.error("✗ Workflow failed!")

        # 显示错误信息
        activities = await WorkflowManager.get_activities_by_workflow(workflow_uid)
        for activity in activities:
            if activity.status.value == "failed" and activity.error_message:
                logger.error(f"\n  ❌ {activity.name}: {activity.error_message}")

    # 6. 清理
    logger.info("\n" + "=" * 60)
    logger.info("Cleanup")
    logger.info("=" * 60)

    try:
        os.unlink(file_path)
        logger.success("✓ Removed temporary file")
    except:
        pass

    logger.info("\n" + "=" * 60)
    logger.success("🎉 Quick Start Complete!")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("  1. Run full demo: python ext/workflow/demo.py")
    logger.info("  2. Read documentation: ext/workflow/README.md")
    logger.info("  3. Create your own custom tasks")
    logger.info("\n")


async def main():
    async with ctx():
        await quick_start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Quick start failed: {e}")
        sys.exit(1)
