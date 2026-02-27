"""测试 Workflow 结构验证逻辑"""

import pytest
from service.collection.helper import WorkflowTemplateValidator


class TestWorkflowStructureValidation:
    """测试 workflow_template 的结构验证"""

    def test_valid_single_activity_root(self):
        """测试合法的单个活动（根节点）"""
        workflow = {
            "parse_document": {
                "input": {"document_id": 0},
                "execute_params": {"task_name": "workflow_document.DocumentParseTask"},
                "depends_on": [],
            }
        }

        # 应该通过验证
        WorkflowTemplateValidator.validate(workflow)

    def test_valid_workflow_subset(self):
        """测试合法的活动子集（从根节点开始）"""
        workflow = {
            "parse_document": {
                "input": {"document_id": 0},
                "execute_params": {"task_name": "workflow_document.DocumentParseTask"},
                "depends_on": [],
            },
            "chunk_document": {
                "input": {"document_id": 0, "strategy": "auto"},
                "execute_params": {"task_name": "workflow_document.DocumentChunkTask"},
                "depends_on": ["parse_document"],
            },
        }

        # 应该通过验证
        WorkflowTemplateValidator.validate(workflow)

    def test_valid_full_workflow(self):
        """测试完整的合法 workflow"""
        workflow = {
            "parse_document": {
                "input": {"document_id": 0},
                "execute_params": {"task_name": "workflow_document.DocumentParseTask"},
                "depends_on": [],
            },
            "chunk_document": {
                "input": {"document_id": 0, "strategy": "auto"},
                "execute_params": {"task_name": "workflow_document.DocumentChunkTask"},
                "depends_on": ["parse_document"],
            },
            "index_chunks": {
                "input": {"document_id": 0, "batch_size": 100},
                "execute_params": {"task_name": "workflow_document.IndexChunkTask"},
                "depends_on": ["chunk_document"],
            },
        }

        # 应该通过验证
        WorkflowTemplateValidator.validate(workflow)

    def test_invalid_first_activity_with_deps(self):
        """测试非法：第一个活动有依赖"""
        workflow = {
            "chunk_document": {
                "input": {"document_id": 0, "strategy": "auto"},
                "execute_params": {"task_name": "workflow_document.DocumentChunkTask"},
                "depends_on": ["parse_document"],  # ❌ 第一个活动有依赖
            },
            "parse_document": {
                "input": {"document_id": 0},
                "execute_params": {"task_name": "workflow_document.DocumentParseTask"},
                "depends_on": [],
            },
        }

        # 应该抛出 ValueError
        with pytest.raises(ValueError, match="第一个活动.*必须是根节点.*不能有依赖"):
            WorkflowTemplateValidator.validate(workflow)

    def test_invalid_single_activity_with_deps(self):
        """测试非法：单个活动有依赖"""
        workflow = {
            "chunk_document": {
                "input": {"document_id": 0, "strategy": "auto"},
                "execute_params": {"task_name": "workflow_document.DocumentChunkTask"},
                "depends_on": ["parse_document"],  # ❌ 单个活动但有依赖
            }
        }

        # 应该抛出 ValueError（依赖的活动不存在）
        with pytest.raises(ValueError, match="依赖的活动.*不存在"):
            WorkflowTemplateValidator.validate(workflow)

    def test_invalid_dependency_order(self):
        """测试非法：依赖顺序错误（依赖在后面）"""
        workflow = {
            "parse_document": {
                "input": {"document_id": 0},
                "execute_params": {"task_name": "workflow_document.DocumentParseTask"},
                "depends_on": [],
            },
            "index_chunks": {
                "input": {"document_id": 0, "batch_size": 100},
                "execute_params": {"task_name": "workflow_document.IndexChunkTask"},
                "depends_on": ["chunk_document"],  # ❌ 依赖在后面
            },
            "chunk_document": {
                "input": {"document_id": 0, "strategy": "auto"},
                "execute_params": {"task_name": "workflow_document.DocumentChunkTask"},
                "depends_on": ["parse_document"],
            },
        }

        # 应该抛出 ValueError
        with pytest.raises(ValueError, match="依赖的.*必须在它之前定义"):
            WorkflowTemplateValidator.validate(workflow)

    def test_valid_parallel_activities(self):
        """测试合法的并行活动（都依赖同一个前置活动）"""
        workflow = {
            "parse_document": {
                "input": {"document_id": 0},
                "execute_params": {"task_name": "workflow_document.DocumentParseTask"},
                "depends_on": [],
            },
            "generate_tags": {
                "input": {"document_id": 0, "max_tags": 10},
                "execute_params": {"task_name": "workflow_document.GenerateTagsTask"},
                "depends_on": ["parse_document"],
            },
            "generate_faq": {
                "input": {"document_id": 0, "max_faq": 5},
                "execute_params": {"task_name": "workflow_document.GenerateFAQTask"},
                "depends_on": ["parse_document"],  # ✅ 并行活动
            },
        }

        # 应该通过验证
        WorkflowTemplateValidator.validate(workflow)

    def test_invalid_missing_dependency(self):
        """测试非法：依赖的活动不存在"""
        workflow = {
            "parse_document": {
                "input": {"document_id": 0},
                "execute_params": {"task_name": "workflow_document.DocumentParseTask"},
                "depends_on": [],
            },
            "chunk_document": {
                "input": {"document_id": 0, "strategy": "auto"},
                "execute_params": {"task_name": "workflow_document.DocumentChunkTask"},
                "depends_on": ["non_existent_activity"],  # ❌ 依赖不存在
            },
        }

        # 应该抛出 ValueError
        with pytest.raises(ValueError, match="依赖的活动.*不存在"):
            WorkflowTemplateValidator.validate(workflow)

    def test_valid_complex_dag(self):
        """测试复杂的合法 DAG"""
        workflow = {
            "parse_document": {
                "input": {"document_id": 0},
                "execute_params": {"task_name": "workflow_document.DocumentParseTask"},
                "depends_on": [],
            },
            "chunk_document": {
                "input": {"document_id": 0, "strategy": "auto"},
                "execute_params": {"task_name": "workflow_document.DocumentChunkTask"},
                "depends_on": ["parse_document"],
            },
            "generate_tags": {
                "input": {"document_id": 0, "max_tags": 10},
                "execute_params": {"task_name": "workflow_document.GenerateTagsTask"},
                "depends_on": ["parse_document"],
            },
            "index_chunks": {
                "input": {"document_id": 0, "batch_size": 100},
                "execute_params": {"task_name": "workflow_document.IndexChunkTask"},
                "depends_on": ["chunk_document"],
            },
        }

        # 应该通过验证（parse → [chunk, tags] → index）
        WorkflowTemplateValidator.validate(workflow)
