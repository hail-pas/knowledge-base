"""
Quick start script

Quickly start and run a simple workflow example.
No configuration required, works out of the box.
"""

import os
import sys
import asyncio
import argparse
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger

from core.context import ctx
from ext.workflow import demo_tasks  # Import demo tasks to register them
from ext.workflow import WorkflowManager, schedule_workflow
from ext.ext_tortoise.enums import WorkflowStatusEnum


def setup_logger() -> None:
    """Configure simple logger"""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>",
        level="INFO",
    )


def create_temp_file(content: str) -> str:
    """Create temporary file"""
    fd, path = tempfile.mkstemp(suffix=".txt", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path


async def quick_start(execute_mode: str = "direct") -> None:
    """Quick start workflow

    Args:
        execute_mode: Execution mode (direct/celery)
    """
    setup_logger()

    logger.info("=" * 60)
    logger.info("Workflow Quick Start")
    logger.info("=" * 60)

    sample_text = """Hello, Workflow System!
This is a quick demonstration of the workflow engine.
The system will process this file through multiple tasks:
1. Fetch file
2. Load and analyze it
3. Generate a summary
"""
    file_path = create_temp_file(sample_text)
    logger.success(f"✓ Created sample file: {file_path}")

    workflow_config = {
        "fetch_file": {
            "input": {"file_path": file_path},
            "execute_params": {"task_name": "workflow_activity.FetchFileTask"},
            "depends_on": [],
        },
        "load_file": {
            "execute_params": {"task_name": "workflow_activity.LoadFileTask"},
            "depends_on": ["fetch_file"],
        },
        "replace_content": {
            "execute_params": {"task_name": "workflow_activity.ReplaceContentTask"},
            "depends_on": ["load_file"],
            "input": {"replace_rules": []},
        },
        "summary": {
            "execute_params": {"task_name": "workflow_activity.SummaryTask"},
            "depends_on": ["replace_content"],
            "input": {"max_length": 100},
        },
    }

    logger.success("✓ Workflow configured (4 tasks)")
    logger.info("  Task flow: fetch_file → load_file → replace_content → summary")

    logger.info("\n" + "=" * 60)
    logger.info("Starting Workflow")
    logger.info("=" * 60)

    logger.info(f"\n🚀 Starting workflow in {execute_mode} mode...")

    if execute_mode == "celery":
        logger.info("Note: Celery Worker must be running in another terminal")
        logger.info("If workflow stalls, check if worker is running:")
        logger.info("  uv run celery -A ext.ext_celery.worker worker -l info\n")
    else:
        logger.info("Note: Tasks will execute directly in the same process.\n")

    try:
        workflow_uid = await schedule_workflow(
            config=workflow_config,
            config_format="dict",
            initial_inputs={},
            execute_mode=execute_mode,  # type: ignore
        )

        logger.success(f"✓ Workflow started: {workflow_uid}")
        logger.info(f"  Mode: {execute_mode}")
        if execute_mode == "celery":
            logger.info("  Execution: Tasks will be processed by Celery worker")
        else:
            logger.info("  Execution: Tasks will execute directly and wait for completion")
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

    logger.info("\n⏳ Waiting for workflow to complete...")
    logger.info("Press Ctrl+C to stop waiting\n")

    max_wait = 60
    for i in range(max_wait):
        await asyncio.sleep(1)

        import uuid

        workflow = await WorkflowManager.get_workflow_by_uid(uuid.UUID(workflow_uid))
        if not workflow:
            logger.error("✗ Workflow not found")
            return

        if workflow.status in ["completed", "failed"]:
            break

        if (i + 1) % 5 == 0:
            activities = await WorkflowManager.get_activities_by_workflow(workflow.uid)
            completed = sum(1 for a in activities if a.status.value == "completed")
            total = len(activities)
            logger.info(f"  Progress: {completed}/{total} tasks completed")

    logger.info("\n" + "=" * 60)
    logger.info("Workflow Result")
    logger.info("=" * 60)

    import uuid

    workflow = await WorkflowManager.get_workflow_by_uid(uuid.UUID(workflow_uid))
    logger.info(f"\nStatus: {workflow.status.value}")

    if workflow.status.value == "completed":
        logger.success("✅ Workflow completed successfully!")

        activities = await WorkflowManager.get_activities_by_workflow(workflow.uid)
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

        activities = await WorkflowManager.get_activities_by_workflow(workflow.uid)
        for activity in activities:
            if activity.status.value == "failed" and activity.error_message:
                logger.error(f"\n  ❌ {activity.name}: {activity.error_message}")

    logger.info("\n" + "=" * 60)
    logger.info("Cleanup")
    logger.info("=" * 60)

    try:
        os.unlink(file_path)
        logger.success("✓ Removed temporary file")
    except OSError:
        pass

    logger.info("\n" + "=" * 60)
    logger.success("🎉 Quick Start Complete!")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("  1. Run full demo: python ext/workflow/demo.py")
    logger.info("  2. Read documentation: ext/workflow/README.md")
    logger.info("  3. Create your own custom tasks")
    logger.info("\n")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Quick start workflow demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["direct", "celery"],
        default="direct",
        help=(
            "Execution mode: direct (execute in process) or celery "
            "(execute via Celery worker:\n"
            "uv run celery -A ext.ext_celery.worker worker -Q workflow_handoff -c 1\n"
            "uv run celery -A ext.ext_celery.worker worker -c 8)"
        ),
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    async with ctx():
        await quick_start(execute_mode=args.mode)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Quick start failed: {e}")
        sys.exit(1)
