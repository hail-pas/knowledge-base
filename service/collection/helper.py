from typing import List

from ext.ext_tortoise.models.knowledge_base import Collection, Document, EmbeddingModelConfig
from ext.ext_tortoise.enums import DocumentStatusEnum
from ext.indexing.models import CollectionIndexModelHelper

from service.workflow.document.schemas import (
    DocumentParseTaskInput,
    DocumentChunkTaskInput,
    IndexChunkTaskInput,
    GenerateTagsTaskInput,
    GenerateFAQTaskInput,
)


class WorkflowTemplateValidator:
    """Validates workflow_template structure for Collection"""

    VALID_TASKS = {
        "workflow_document.DocumentParseTask": DocumentParseTaskInput,
        "workflow_document.DocumentChunkTask": DocumentChunkTaskInput,
        "workflow_document.IndexChunkTask": IndexChunkTaskInput,
        "workflow_document.GenerateTagsTask": GenerateTagsTaskInput,
        "workflow_document.GenerateFAQTask": GenerateFAQTaskInput,
    }

    @classmethod
    def validate(cls, workflow_template: dict) -> None:
        """
        Validates workflow template:
        1. All activity names exist
        2. All execute_params.task_name are valid
        3. All inputs match their schemas
        4. All depends_on references valid activities
        5. Activities can be subset but must start from beginning with correct dependencies
        """

        if not workflow_template or not isinstance(workflow_template, dict):
            raise ValueError("workflow_template 不能为空且必须是字典格式")

        activity_names = list(workflow_template.keys())
        if not activity_names:
            raise ValueError("workflow_template 至少需要一个活动")

        for activity_name, activity_config in workflow_template.items():
            if not isinstance(activity_config, dict):
                raise ValueError(f"活动 '{activity_name}' 的配置必须是字典")

            cls._validate_execute_params(activity_name, activity_config)
            cls._validate_input(activity_name, activity_config)
            cls._validate_depends_on(activity_name, activity_config, activity_names)

        cls._validate_workflow_structure(workflow_template)

    @classmethod
    def _validate_execute_params(cls, activity_name: str, activity_config: dict) -> None:
        execute_params = activity_config.get("execute_params", {})
        if not isinstance(execute_params, dict):
            raise ValueError(f"活动 '{activity_name}' 的 execute_params 必须是字典")

        task_name = execute_params.get("task_name")
        if not task_name or not isinstance(task_name, str):
            raise ValueError(f"活动 '{activity_name}' 必须提供有效的 task_name")

        if task_name not in cls.VALID_TASKS:
            valid_tasks = ", ".join(cls.VALID_TASKS.keys())
            raise ValueError(f"活动 '{activity_name}' 的 task_name '{task_name}' 无效，必须是: {valid_tasks}")

    @classmethod
    def _validate_input(cls, activity_name: str, activity_config: dict) -> None:
        execute_params = activity_config.get("execute_params", {})
        task_name = execute_params.get("task_name")
        input_data = activity_config.get("input", {})

        if not isinstance(input_data, dict):
            raise ValueError(f"活动 '{activity_name}' 的 input 必须是字典")

        input_schema_class = cls.VALID_TASKS.get(task_name)
        if input_schema_class:
            try:
                input_schema_class(**input_data)
            except Exception as e:
                raise ValueError(f"活动 '{activity_name}' 的 input 验证失败: {str(e)}")

    @classmethod
    def _validate_depends_on(cls, activity_name: str, activity_config: dict, activity_names: List[str]) -> None:
        depends_on = activity_config.get("depends_on", [])
        if not isinstance(depends_on, list):
            raise ValueError(f"活动 '{activity_name}' 的 depends_on 必须是列表")

        for dep in depends_on:
            if dep not in activity_names:
                raise ValueError(f"活动 '{activity_name}' 依赖的活动 '{dep}' 不存在于 workflow_template 中")

    @classmethod
    def _validate_workflow_structure(cls, workflow_template: dict) -> None:
        """
        Validates that activities form a valid DAG starting from the beginning.
        Allows subset but must maintain correct dependencies.

        Rules:
        1. First activity must be root (depends_on: [])
        2. All dependencies must appear before the activity that depends on them
        3. No circular dependencies
        """
        activity_names = list(workflow_template.keys())

        if len(activity_names) == 1:
            first_activity = activity_names[0]
            first_config = workflow_template[first_activity]
            depends_on = first_config.get("depends_on", [])
            if depends_on and depends_on != []:
                raise ValueError(f"第一个活动 '{first_activity}' 不能有依赖")
            return

        first_activity = activity_names[0]
        first_config = workflow_template[first_activity]
        depends_on = first_config.get("depends_on", [])
        if depends_on and depends_on != []:
            raise ValueError(f"第一个活动 '{first_activity}' 必须是根节点，不能有依赖")

        activity_index = {name: i for i, name in enumerate(activity_names)}

        for activity_name, activity_config in workflow_template.items():
            depends_on = activity_config.get("depends_on", [])
            for dep in depends_on:
                if dep not in activity_index:
                    continue

                if activity_index[dep] >= activity_index[activity_name]:
                    raise ValueError(
                        f"活动 '{activity_name}' 依赖的 '{dep}' 必须在它之前定义 "
                        f"('{dep}' 位置: {activity_index[dep]}, '{activity_name}' 位置: {activity_index[activity_name]})"
                    )


class CollectionService:
    def __init__(self, collection: Collection):
        self.collection = collection
        self.collection_index_helper = CollectionIndexModelHelper(collection)

    async def switch_embedding_model(self, new_embedding_model_config: EmbeddingModelConfig):
        """
        Switch embedding model for collection with rollback support.

        This method performs:
        1. Rebuild all indexes for documents in this collection
        2. Atomic transaction with rollback on failure
        3. Handle partial failures gracefully

        # TODO: Implement full logic:
        - Get all documents in collection
        - For each document, rebuild chunks with new embedding model
        - Update dense_model and faq_model indexes
        - If any failure, rollback to old embedding model
        """
        old_config = await self.collection.embedding_model_config

        self.collection.embedding_model_config = new_embedding_model_config
        await self.collection.save()

        try:
            pass
        except Exception as e:
            await self.collection.embedding_model_config.set(old_config)
            await self.collection.save()
            raise e

    async def can_delete_collection(self) -> bool:
        """
        Check if collection can be deleted.
        Returns True if all documents are in final status (success/failure).
        """
        documents = await Document.filter(collection_id=self.collection.id, deleted_at=0).values_list(
            "status", flat=True
        )

        final_statuses = {
            DocumentStatusEnum.success.value,
            DocumentStatusEnum.failure.value,
        }

        return all(status in final_statuses for status in documents)
