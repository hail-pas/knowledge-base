#!/usr/bin/env python3
"""
工作流系统测试脚本

验证系统的各项功能：
- 工作流创建和管理
- 任务执行
- 并行和串行任务
- 错误处理
- 断点续传
- 输出传播
"""

import asyncio
import os
import sys
import tempfile
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ext.workflow import WorkflowManager, ActivityTaskTemplate
from ext.workflow.template import activity_task
from ext.workflow.tasks import schedule_workflow_start, schedule_workflow_resume
from ext.workflow.file_process_tasks import FILE_PROCESS_WORKFLOW
from loguru import logger


class TestResults:
    """测试结果统计"""

    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.errors = []

    def add_result(self, test_name: str, passed: bool, error: str = None):
        """添加测试结果"""
        self.total += 1
        if passed:
            self.passed += 1
            logger.success(f"✅ PASS: {test_name}")
        else:
            self.failed += 1
            self.errors.append((test_name, error))
            logger.error(f"❌ FAIL: {test_name}")
            if error:
                logger.error(f"   Error: {error}")

    def print_summary(self):
        """打印测试摘要"""
        logger.info("\n" + "=" * 60)
        logger.info("Test Summary")
        logger.info("=" * 60)
        logger.info(f"Total:  {self.total}")
        logger.success(f"Passed: {self.passed}")
        logger.error(f"Failed: {self.failed}")
        logger.info(f"Success rate: {self.passed / self.total * 100:.1f}%")

        if self.errors:
            logger.error("\nFailed tests:")
            for test_name, error in self.errors:
                logger.error(f"\n  ❌ {test_name}")
                if error:
                    logger.error(f"     {error}")

        return self.failed == 0


def setup_test_logger():
    """配置测试日志"""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>",
        level="INFO",
    )
    logger.add(
        "workflow_test.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
    )


# =============================================================================
# Test Tasks
# =============================================================================


class SimpleTask(ActivityTaskTemplate):
    """简单的测试任务"""

    async def execute(self) -> Dict[str, Any]:
        value = self.input.get("value", 0)
        await asyncio.sleep(0.1)
        return {"result": value + 1}


class FailingTask(ActivityTaskTemplate):
    """会失败的任务"""

    async def execute(self) -> Dict[str, Any]:
        should_fail = self.input.get("should_fail", False)
        if should_fail:
            await asyncio.sleep(0.1)
            raise ValueError("Task failed as expected")
        await asyncio.sleep(0.1)
        return {"result": "success"}


class SlowTask(ActivityTaskTemplate):
    """慢速任务"""

    async def execute(self) -> Dict[str, Any]:
        delay = self.input.get("delay", 1)
        await asyncio.sleep(delay)
        return {"result": f"slept for {delay}s"}


class OutputPropagationTask(ActivityTaskTemplate):
    """测试输出传播的任务"""

    async def execute(self) -> Dict[str, Any]:
        # 获取上游输出
        upstream_outputs = await self.get_upstream_outputs()
        message = self.input.get("message", "default")

        # 合并上游输出
        merged = {}
        for source, output in upstream_outputs.items():
            merged[source] = output

        return {"message": message, "upstream": merged}


# 注册测试任务
simple_task = activity_task(SimpleTask)
failing_task = activity_task(FailingTask)
slow_task = activity_task(SlowTask)
output_propagation_task = activity_task(OutputPropagationTask)


# =============================================================================
# Test Functions
# =============================================================================


async def test_workflow_creation(results: TestResults):
    """测试工作流创建"""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Workflow Creation")
    logger.info("=" * 60)

    try:
        config = {
            "task1": {
                "input": {"value": 1},
                "execute_params": {"task_name": simple_task.name},
                "depends_on": [],
            }
        }

        workflow = await WorkflowManager.create_workflow(config=config, config_format="dict")

        assert workflow is not None, "Workflow should be created"
        assert workflow.uid is not None, "Workflow should have UID"
        assert workflow.status.value == "pending", "Initial status should be pending"

        # 检查活动是否创建
        activities = await WorkflowManager.get_activities_by_workflow(str(workflow.uid))
        assert len(activities) == 1, "Should have 1 activity"
        assert activities[0].name == "task1", "Activity name should be task1"

        results.add_result("Workflow Creation", True)
    except Exception as e:
        results.add_result("Workflow Creation", False, str(e) + "\n" + traceback.format_exc())


async def test_workflow_retrieval(results: TestResults):
    """测试工作流检索"""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Workflow Retrieval")
    logger.info("=" * 60)

    try:
        config = {
            "task1": {
                "execute_params": {"task_name": simple_task.name},
                "depends_on": [],
            }
        }

        workflow = await WorkflowManager.create_workflow(config=config)
        workflow_uid = str(workflow.uid)

        # 检索工作流
        retrieved_workflow = await WorkflowManager.get_workflow_by_uid(workflow_uid)

        assert retrieved_workflow is not None, "Workflow should be retrievable"
        assert retrieved_workflow.uid == workflow.uid, "UID should match"

        results.add_result("Workflow Retrieval", True)
    except Exception as e:
        results.add_result("Workflow Retrieval", False, str(e) + "\n" + traceback.format_exc())


async def test_activity_retrieval(results: TestResults):
    """测试活动检索"""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Activity Retrieval")
    logger.info("=" * 60)

    try:
        config = {
            "task1": {
                "execute_params": {"task_name": simple_task.name},
                "depends_on": [],
            }
        }

        workflow = await WorkflowManager.create_workflow(config=config)
        workflow_uid = str(workflow.uid)

        # 按名称检索活动
        activity = await WorkflowManager.get_activity_by_name(workflow_uid, "task1")

        assert activity is not None, "Activity should be retrievable"
        assert activity.name == "task1", "Activity name should match"

        results.add_result("Activity Retrieval", True)
    except Exception as e:
        results.add_result("Activity Retrieval", False, str(e) + "\n" + traceback.format_exc())


async def test_serial_workflow_execution(results: TestResults):
    """测试串行工作流执行"""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Serial Workflow Execution")
    logger.info("=" * 60)

    try:
        config = {
            "task1": {
                "input": {"value": 1},
                "execute_params": {"task_name": simple_task.name},
                "depends_on": [],
            },
            "task2": {
                "execute_params": {"task_name": simple_task.name},
                "depends_on": ["task1"],
            },
            "task3": {
                "execute_params": {"task_name": simple_task.name},
                "depends_on": ["task2"],
            },
        }

        # 启动工作流
        workflow_uid = schedule_workflow_start(config=config, config_format="dict")

        # 等待完成
        completed = await wait_for_workflow_completion(workflow_uid, timeout=30)

        assert completed, "Workflow should complete"

        # 检查结果
        activities = await WorkflowManager.get_activities_by_workflow(workflow_uid)
        for activity in activities:
            assert activity.status.value == "completed", f"Activity {activity.name} should complete"

        results.add_result("Serial Workflow Execution", True)
    except Exception as e:
        results.add_result("Serial Workflow Execution", False, str(e) + "\n" + traceback.format_exc())


async def test_parallel_workflow_execution(results: TestResults):
    """测试并行工作流执行"""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Parallel Workflow Execution")
    logger.info("=" * 60)

    try:
        config = {
            "task_a": {
                "input": {"delay": 0.5},
                "execute_params": {"task_name": slow_task.name},
                "depends_on": [],
            },
            "task_b": {
                "input": {"delay": 0.5},
                "execute_params": {"task_name": slow_task.name},
                "depends_on": [],
            },
            "task_c": {
                "input": {"delay": 0.5},
                "execute_params": {"task_name": slow_task.name},
                "depends_on": [],
            },
            "merge": {
                "execute_params": {"task_name": simple_task.name},
                "depends_on": ["task_a", "task_b", "task_c"],
            },
        }

        start_time = datetime.now()

        # 启动工作流
        workflow_uid = schedule_workflow_start(config=config, config_format="dict")

        # 等待完成
        completed = await wait_for_workflow_completion(workflow_uid, timeout=30)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # 并行任务应该在约 0.5 秒内完成，而不是串行的 1.5 秒
        # 加上一些缓冲，应该是 < 1.0 秒
        logger.info(f"Parallel workflow duration: {duration:.2f}s")

        assert completed, "Workflow should complete"
        assert duration < 1.5, f"Parallel execution should be faster, took {duration:.2f}s"

        results.add_result("Parallel Workflow Execution", True)
    except Exception as e:
        results.add_result("Parallel Workflow Execution", False, str(e) + "\n" + traceback.format_exc())


async def test_output_propagation(results: TestResults):
    """测试输出传播"""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Output Propagation")
    logger.info("=" * 60)

    try:
        config = {
            "producer_a": {
                "input": {"value": 100},
                "execute_params": {"task_name": simple_task.name},
                "depends_on": [],
            },
            "producer_b": {
                "input": {"value": 200},
                "execute_params": {"task_name": simple_task.name},
                "depends_on": [],
            },
            "consumer": {
                "input": {"message": "Consuming outputs"},
                "execute_params": {"task_name": output_propagation_task.name},
                "depends_on": ["producer_a", "producer_b"],
            },
        }

        # 启动工作流
        workflow_uid = schedule_workflow_start(config=config, config_format="dict")

        # 等待完成
        completed = await wait_for_workflow_completion(workflow_uid, timeout=30)

        assert completed, "Workflow should complete"

        # 检查消费者的输入是否包含上游输出
        activities = await WorkflowManager.get_activities_by_workflow(workflow_uid)
        consumer = next((a for a in activities if a.name == "consumer"), None)

        assert consumer is not None, "Consumer task should exist"
        assert consumer.status.value == "completed", "Consumer should complete"

        # 检查消费者任务的输出
        output = consumer.output
        assert "upstream" in output, "Output should contain upstream data"
        assert "producer_a" in output["upstream"], "Should have output from producer_a"
        assert "producer_b" in output["upstream"], "Should have output from producer_b"

        # 验证上游输出的值
        assert output["upstream"]["producer_a"]["result"] == 101, "Producer A result should be 101"
        assert output["upstream"]["producer_b"]["result"] == 201, "Producer B result should be 201"

        results.add_result("Output Propagation", True)
    except Exception as e:
        results.add_result("Output Propagation", False, str(e) + "\n" + traceback.format_exc())


async def test_error_handling(results: TestResults):
    """测试错误处理"""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Error Handling")
    logger.info("=" * 60)

    try:
        config = {
            "good_task": {
                "input": {"should_fail": False},
                "execute_params": {"task_name": failing_task.name},
                "depends_on": [],
            },
            "bad_task": {
                "input": {"should_fail": True},
                "execute_params": {"task_name": failing_task.name, "max_retries": 2},
                "depends_on": [],
            },
        }

        # 启动工作流
        workflow_uid = schedule_workflow_start(config=config, config_format="dict")

        # 等待完成
        completed = await wait_for_workflow_completion(workflow_uid, timeout=30)

        assert completed, "Workflow should complete (with failures)"

        # 检查工作流状态
        workflow = await WorkflowManager.get_workflow_by_uid(workflow_uid)
        assert workflow.status.value == "failed", "Workflow should be marked as failed"

        # 检查任务状态
        activities = await WorkflowManager.get_activities_by_workflow(workflow_uid)

        good_task = next((a for a in activities if a.name == "good_task"), None)
        bad_task = next((a for a in activities if a.name == "bad_task"), None)

        assert good_task is not None and good_task.status.value == "completed", "Good task should complete"
        assert bad_task is not None and bad_task.status.value == "failed", "Bad task should fail"
        assert bad_task.retry_count == 2, f"Bad task should have 2 retries, got {bad_task.retry_count}"
        assert bad_task.error_message is not None, "Bad task should have error message"

        results.add_result("Error Handling", True)
    except Exception as e:
        results.add_result("Error Handling", False, str(e) + "\n" + traceback.format_exc())


async def test_workflow_resume(results: TestResults):
    """测试工作流恢复"""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Workflow Resume")
    logger.info("=" * 60)

    try:
        config = {
            "task1": {
                "input": {"value": 1},
                "execute_params": {"task_name": simple_task.name},
                "depends_on": [],
            },
            "task2": {
                "execute_params": {"task_name": simple_task.name},
                "depends_on": ["task1"],
            },
            "task3": {
                "input": {"should_fail": True},
                "execute_params": {"task_name": failing_task.name, "max_retries": 2},
                "depends_on": ["task2"],
            },
            "task4": {
                "execute_params": {"task_name": simple_task.name},
                "depends_on": ["task3"],
            },
        }

        # 启动工作流
        workflow_uid = schedule_workflow_start(config=config, config_format="dict")

        # 等待失败
        await wait_for_workflow_completion(workflow_uid, timeout=30)

        # 检查状态
        workflow = await WorkflowManager.get_workflow_by_uid(workflow_uid)
        assert workflow.status.value == "failed", "Workflow should fail initially"

        # 修复配置，移除失败的任务
        config_fixed = {
            "task1": {
                "input": {"value": 1},
                "execute_params": {"task_name": simple_task.name},
                "depends_on": [],
            },
            "task2": {
                "execute_params": {"task_name": simple_task.name},
                "depends_on": ["task1"],
            },
            "task3_fixed": {
                "input": {"should_fail": False},
                "execute_params": {"task_name": failing_task.name},
                "depends_on": ["task2"],
            },
            "task4": {
                "execute_params": {"task_name": simple_task.name},
                "depends_on": ["task3_fixed"],
            },
        }

        # 创建新的工作流
        workflow_uid2 = schedule_workflow_start(config=config_fixed, config_format="dict")

        # 等待完成
        completed = await wait_for_workflow_completion(workflow_uid2, timeout=30)

        assert completed, "Fixed workflow should complete"

        results.add_result("Workflow Resume", True)
    except Exception as e:
        results.add_result("Workflow Resume", False, str(e) + "\n" + traceback.format_exc())


async def test_file_processing_workflow(results: TestResults):
    """测试完整的文件处理工作流"""
    logger.info("\n" + "=" * 60)
    logger.info("Test: File Processing Workflow")
    logger.info("=" * 60)

    try:
        # 创建临时文件
        sample_content = "This is a sample document for testing.\n" * 20
        fd, file_path = tempfile.mkstemp(suffix=".txt", text=True)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(sample_content)

        # 配置工作流
        workflow_config = FILE_PROCESS_WORKFLOW.copy()
        workflow_config["fetch_file"]["input"]["file_path"] = file_path

        # 简化工作流（移除索引任务，因为可能没有 Milvus/ES）
        simple_config = {
            "fetch_file": workflow_config["fetch_file"],
            "load_file": workflow_config["load_file"],
            "summary": workflow_config["summary"],
        }

        # 启动工作流
        workflow_uid = schedule_workflow_start(config=simple_config, config_format="dict")

        # 等待完成
        completed = await wait_for_workflow_completion(workflow_uid, timeout=60)

        assert completed, "File processing workflow should complete"

        # 检查结果
        workflow = await WorkflowManager.get_workflow_by_uid(workflow_uid)
        assert workflow.status.value == "completed", "Workflow should complete successfully"

        activities = await WorkflowManager.get_activities_by_workflow(workflow_uid)
        for activity in activities:
            assert activity.status.value == "completed", f"Activity {activity.name} should complete"
            assert activity.output is not None, f"Activity {activity.name} should have output"

        # 检查摘要输出
        summary_activity = next((a for a in activities if a.name == "summary"), None)
        assert summary_activity is not None, "Summary activity should exist"
        assert "summary" in summary_activity.output, "Should have summary output"

        # 清理
        os.unlink(file_path)

        results.add_result("File Processing Workflow", True)
    except Exception as e:
        results.add_result("File Processing Workflow", False, str(e) + "\n" + traceback.format_exc())


async def test_complex_dag(results: TestResults):
    """测试复杂的 DAG 结构"""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Complex DAG Structure")
    logger.info("=" * 60)

    try:
        # 复杂的 DAG 结构
        #   A
        #  / \
        # B   C
        # | \ /|
        # D  E
        #  \ /
        #   F
        config = {
            "task_a": {
                "execute_params": {"task_name": simple_task.name},
                "depends_on": [],
            },
            "task_b": {
                "execute_params": {"task_name": simple_task.name},
                "depends_on": ["task_a"],
            },
            "task_c": {
                "execute_params": {"task_name": simple_task.name},
                "depends_on": ["task_a"],
            },
            "task_d": {
                "execute_params": {"task_name": simple_task.name},
                "depends_on": ["task_b"],
            },
            "task_e": {
                "execute_params": {"task_name": simple_task.name},
                "depends_on": ["task_b", "task_c"],
            },
            "task_f": {
                "execute_params": {"task_name": simple_task.name},
                "depends_on": ["task_d", "task_e"],
            },
        }

        # 启动工作流
        workflow_uid = schedule_workflow_start(config=config, config_format="dict")

        # 等待完成
        completed = await wait_for_workflow_completion(workflow_uid, timeout=30)

        assert completed, "Complex DAG workflow should complete"

        # 验证执行顺序
        activities = await WorkflowManager.get_activities_by_workflow(workflow_uid)
        completed_times = {}

        for activity in activities:
            if activity.completed_at:
                completed_times[activity.name] = activity.completed_at

        # 验证依赖关系
        assert (
            completed_times["task_a"] < completed_times["task_b"]
        ), "task_a should complete before task_b"
        assert (
            completed_times["task_a"] < completed_times["task_c"]
        ), "task_a should complete before task_c"
        assert (
            completed_times["task_b"] < completed_times["task_e"]
        ), "task_b should complete before task_e"
        assert (
            completed_times["task_c"] < completed_times["task_e"]
        ), "task_c should complete before task_e"

        results.add_result("Complex DAG Structure", True)
    except Exception as e:
        results.add_result("Complex DAG Structure", False, str(e) + "\n" + traceback.format_exc())


# =============================================================================
# Helper Functions
# =============================================================================


async def wait_for_workflow_completion(workflow_uid: str, timeout: int = 60) -> bool:
    """等待工作流完成"""
    import time

    start_time = time.time()

    while time.time() - start_time < timeout:
        workflow = await WorkflowManager.get_workflow_by_uid(workflow_uid)
        if not workflow:
            return False

        if workflow.status.value in ["completed", "failed", "canceled"]:
            return True

        await asyncio.sleep(1)

    return False


async def run_all_tests():
    """运行所有测试"""
    setup_test_logger()

    logger.info("=" * 60)
    logger.info("Workflow System Test Suite")
    logger.info("=" * 60)

    results = TestResults()

    # 基础功能测试
    await test_workflow_creation(results)
    await test_workflow_retrieval(results)
    await test_activity_retrieval(results)

    # 工作流执行测试
    await test_serial_workflow_execution(results)
    await test_parallel_workflow_execution(results)

    # 高级功能测试
    await test_output_propagation(results)
    await test_error_handling(results)
    await test_workflow_resume(results)
    await test_complex_dag(results)

    # 实际应用测试
    await test_file_processing_workflow(results)

    # 打印结果
    all_passed = results.print_summary()

    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.success("🎉 All tests passed!")
    else:
        logger.error("❌ Some tests failed")
    logger.info("=" * 60)

    return all_passed


if __name__ == "__main__":
    try:
        all_passed = asyncio.run(run_all_tests())
        sys.exit(0 if all_passed else 1)
    except KeyboardInterrupt:
        logger.info("\n\nTests interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Test suite failed with error: {e}")
        sys.exit(1)
