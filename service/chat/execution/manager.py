from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Any, Callable, Protocol, Awaitable
from datetime import UTC, datetime

from loguru import logger
from pydantic_ai import Agent

import ext.embedding.providers  # noqa: F401
from ext.embedding import EmbeddingModelFactory
from ext.indexing.types import (
    FilterClause,
    DenseSearchClause,
    HybridSearchClause,
    SparseSearchClause,
)
from ext.indexing.models import CollectionIndexModelHelper
from ext.ext_tortoise.enums import (
    ChatDataKindEnum,
    ChatStepKindEnum,
    ChatStepStatusEnum,
    ChatTurnStatusEnum,
)
from service.document.schema import DocumentList, DocumentChunkList
from service.llm_model.factory import LLMModelFactory
from service.chat.domain.schema import (
    ChatEvent,
    TextBlock,
    StrictModel,
    ChatRoleEnum,
    UsagePayload,
    EventNameEnum,
    MCPCallConfig,
    StepIOPayload,
    ChatDataSchema,
    RetrievalBlock,
    ToolCallConfig,
    WarningPayload,
    ChatHistoryItem,
    StepIOPhaseEnum,
    DataEventPayload,
    FunctionToolSpec,
    StepEventPayload,
    TurnEventPayload,
    LLMResponseConfig,
    ResourceSelection,
    SelectionModeEnum,
    StepMetricPayload,
    CapabilityKindEnum,
    ChatActionKindEnum,
    FunctionCallConfig,
    SubAgentCallConfig,
    SystemPromptConfig,
    ChatContextEnvelope,
    ChatPayloadTypeEnum,
    ChatWarningCodeEnum,
    IntentResultPayload,
    MessageDeltaPayload,
    ActionTerminalOutput,
    MessageBundlePayload,
    PromptContextPayload,
    RetrievalListPayload,
    CapabilityPlanPayload,
    ExtensionEventPayload,
    FunctionResultPayload,
    IntentDetectionConfig,
    ExtensionEventLevelEnum,
    ExtensionEventStageEnum,
    IntentRecognitionResult,
    ToolExecutionPolicyEnum,
    FunctionExecutionSummary,
    KnowledgeRetrievalConfig,
    FunctionCallResultModeEnum,
)
from service.chat.runtime.session import ChatSessionContext
from service.chat.store.repository import ChatRepository
from service.chat.runtime.prompting import ChatPromptBuilder
from service.chat.execution.registry import (
    ExecutionAction,
    ExecutionActionRegistry,
    create_default_action_registry,
)
from service.chat.runtime.function_tools import (
    FunctionToolRegistry,
    create_default_function_tool_registry,
)
from ext.ext_tortoise.models.knowledge_base import Document, Collection, DocumentChunk

EventSender = Callable[[ChatEvent[Any]], Awaitable[None]]


class ExtensionEventSender(Protocol):
    async def __call__(
        self,
        stage: ExtensionEventStageEnum,
        message: str,
        data: dict[str, Any] | None = None,
        level: ExtensionEventLevelEnum = ExtensionEventLevelEnum.info,
    ) -> None: ...


class _LLMFunctionToolPlan(StrictModel):
    selected_tool_names: list[str]
    summary: str = ""


class _KnowledgeRetrievalExecutionResult(StrictModel):
    retrievals: list[RetrievalBlock] = []
    requested_collection_ids: list[int] = []
    searched_collection_ids: list[int] = []
    missing_collection_ids: list[int] = []
    inaccessible_collection_ids: list[int] = []
    failed_collection_ids: list[int] = []


class ChatExecutionManager:
    def __init__(
        self,
        repository: ChatRepository,
        action_registry: ExecutionActionRegistry | None = None,
        function_tool_registry: FunctionToolRegistry | None = None,
        prompt_builder: ChatPromptBuilder | None = None,
    ) -> None:
        self.repository = repository
        self.action_registry = action_registry or create_default_action_registry()
        self.function_tool_registry = function_tool_registry or create_default_function_tool_registry()
        self.prompt_builder = prompt_builder or ChatPromptBuilder()

    def action_step_metadata(self, action: ExecutionAction) -> dict[str, Any]:
        return {
            "action_id": action.action_id,
            "action_kind": action.kind.value,
            "action_name": action.name,
            "action_source": action.source,
            **action.metadata,
        }

    def normalize_payload_type(self, payload_type: ChatPayloadTypeEnum | str) -> ChatPayloadTypeEnum:
        return payload_type if isinstance(payload_type, ChatPayloadTypeEnum) else ChatPayloadTypeEnum(payload_type)

    def is_required_action(self, action: ExecutionAction) -> bool:
        return bool(action.metadata.get("required") or action.metadata.get("capability_required"))

    def should_emit_extension_intermediate(self, action: ExecutionAction) -> bool:
        capability_kind = action.metadata.get("capability_kind")
        return bool(action.metadata.get("emit_intermediate_events")) and (
            capability_kind == CapabilityKindEnum.extension.value or action.source == "capability:extension"
        )

    async def emit_extension_stage(
        self,
        action: ExecutionAction,
        *,
        turn,
        parent_step,
        emit: Callable[..., Awaitable[None]],
        seq: int,
        step_sequence: int,
        stage_index: int,
        stage: ExtensionEventStageEnum,
        message: str,
        data: dict[str, Any] | None = None,
        level: ExtensionEventLevelEnum = ExtensionEventLevelEnum.info,
    ) -> tuple[int, int]:
        if not self.should_emit_extension_intermediate(action):
            return seq, stage_index
        stage_step = await self.repository.create_step(
            turn_id=turn.id,
            parent_step_id=parent_step.id,
            root_step_id=parent_step.id,
            name=f"extension:{stage.value}",
            kind=ChatStepKindEnum.system,
            sequence=step_sequence + stage_index,
            metadata={
                **self.action_step_metadata(action),
                "extension_stage": stage.value,
                "extension_level": level.value,
            },
        )
        await emit(
            EventNameEnum.step_created.value,
            StepEventPayload(step=await self.repository.summarize_step(stage_step)),
            seq=seq,
            step_id=stage_step.id,
        )
        seq += 1
        await self.repository.update_step(
            stage_step,
            status=ChatStepStatusEnum.running,
            started_at=datetime.now(UTC),
        )
        await emit(
            EventNameEnum.step_started.value,
            StepEventPayload(step=await self.repository.summarize_step(stage_step)),
            seq=seq,
            step_id=stage_step.id,
        )
        seq += 1
        payload = ExtensionEventPayload(
            extension_key=action.metadata.get("capability_key"),
            capability_id=action.metadata.get("capability_id"),
            action_id=action.action_id,
            action_name=action.name,
            stage=stage,
            level=level,
            message=message,
            data=data or {},
        )
        stage_data = await self.repository.create_data(
            turn_id=turn.id,
            step_id=stage_step.id,
            kind=ChatDataKindEnum.intermediate,
            payload_type=ChatPayloadTypeEnum.extension_event,
            payload=payload.model_dump(mode="json"),
            is_final=False,
            metadata={
                **self.action_step_metadata(action),
                "extension_stage": stage.value,
                "extension_level": level.value,
            },
        )
        await self.repository.update_step(
            stage_step,
            status=ChatStepStatusEnum.completed,
            finished_at=datetime.now(UTC),
            output_data_ids=[stage_data.id],
            metrics=StepMetricPayload(latency_ms=0).model_dump(mode="json"),
        )
        await emit(
            EventNameEnum.data_created.value,
            DataEventPayload[ExtensionEventPayload](
                data=ChatDataSchema(
                    id=stage_data.id,
                    turn_id=turn.id,
                    step_id=stage_step.id,
                    kind=str(stage_data.kind),
                    payload_type=self.normalize_payload_type(stage_data.payload_type),
                    role=stage_data.role,
                    is_final=stage_data.is_final,
                    is_visible=stage_data.is_visible,
                    payload=payload,
                    refs=[],
                    metadata=stage_data.metadata or {},
                ),
            ),
            seq=seq,
            step_id=stage_step.id,
            data_id=stage_data.id,
        )
        seq += 1
        await emit(
            EventNameEnum.step_completed.value,
            StepEventPayload(step=await self.repository.summarize_step(stage_step)),
            seq=seq,
            step_id=stage_step.id,
        )
        return seq + 1, stage_index + 1

    async def emit_step_io_data(
        self,
        step,
        *,
        turn_id: int,
        action: ExecutionAction,
        emit: Callable[..., Awaitable[None]],
        seq: int,
        phase: StepIOPhaseEnum,
        message: str,
        data: dict[str, Any],
    ) -> tuple[Any, int]:
        payload = StepIOPayload(
            phase=phase,
            action_id=action.action_id,
            action_name=action.name,
            action_kind=action.kind,
            message=message,
            data=data,
        )
        step_data = await self.repository.create_data(
            turn_id=turn_id,
            step_id=step.id,
            kind=ChatDataKindEnum.intermediate,
            payload_type=ChatPayloadTypeEnum.step_io,
            payload=payload.model_dump(mode="json"),
            is_final=phase == StepIOPhaseEnum.output,
            metadata={
                **self.action_step_metadata(action),
                "step_io_phase": phase.value,
            },
        )
        data_ids = (
            list(step.input_data_ids or []) if phase == StepIOPhaseEnum.input else list(step.output_data_ids or [])
        )
        data_ids.append(step_data.id)
        await self.repository.update_step(
            step,
            input_data_ids=data_ids if phase == StepIOPhaseEnum.input else None,
            output_data_ids=data_ids if phase == StepIOPhaseEnum.output else None,
        )
        await emit(
            EventNameEnum.data_created.value,
            DataEventPayload[StepIOPayload](
                data=ChatDataSchema(
                    id=step_data.id,
                    turn_id=turn_id,
                    step_id=step.id,
                    kind=str(step_data.kind),
                    payload_type=self.normalize_payload_type(step_data.payload_type),
                    role=step_data.role,
                    is_final=step_data.is_final,
                    is_visible=step_data.is_visible,
                    payload=payload,
                    refs=[],
                    metadata=step_data.metadata or {},
                ),
            ),
            seq=seq,
            step_id=step.id,
            data_id=step_data.id,
        )
        return step, seq + 1

    async def create_action_step(
        self,
        action: ExecutionAction,
        *,
        turn,
        root_step,
        emit: Callable[..., Awaitable[None]],
        seq: int,
        step_sequence: int,
    ) -> tuple[Any, int]:
        step = await self.repository.create_step(
            turn_id=turn.id,
            parent_step_id=root_step.id,
            root_step_id=root_step.id,
            name=action.name,
            kind=action.step_kind,
            sequence=step_sequence,
            metadata=self.action_step_metadata(action),
        )
        await emit(
            EventNameEnum.step_created.value,
            StepEventPayload(step=await self.repository.summarize_step(step)),
            seq=seq,
            step_id=step.id,
        )
        return step, seq + 1

    async def execute_action(
        self,
        action: ExecutionAction,
        *,
        turn,
        root_step,
        session_context: ChatSessionContext,
        cancel_event: asyncio.Event,
        send_event: EventSender,
        emit: Callable[..., Awaitable[None]],
        seq: int,
        step_sequence: int,
        ensure_not_canceled: Callable[[asyncio.Event], Awaitable[None]],
    ) -> int:
        session_context.set_state("active_action_id", action.action_id)
        if action.kind == ChatActionKindEnum.knowledge_retrieval:
            return await self.execute_retrieval_action(
                action,
                turn=turn,
                root_step=root_step,
                session_context=session_context,
                cancel_event=cancel_event,
                emit=emit,
                seq=seq,
                step_sequence=step_sequence,
                ensure_not_canceled=ensure_not_canceled,
            )
        if action.kind == ChatActionKindEnum.intent_detection:
            return await self.execute_intent_action(
                action,
                turn=turn,
                root_step=root_step,
                session_context=session_context,
                emit=emit,
                seq=seq,
                step_sequence=step_sequence,
            )
        if action.kind == ChatActionKindEnum.system_prompt:
            return await self.execute_system_prompt_action(
                action,
                turn=turn,
                root_step=root_step,
                session_context=session_context,
                emit=emit,
                seq=seq,
                step_sequence=step_sequence,
            )
        if action.kind == ChatActionKindEnum.function_call:
            return await self.execute_function_action(
                action,
                turn=turn,
                root_step=root_step,
                session_context=session_context,
                emit=emit,
                seq=seq,
                step_sequence=step_sequence,
            )
        if action.kind == ChatActionKindEnum.tool_call:
            return await self.execute_tool_action(
                action,
                turn=turn,
                root_step=root_step,
                emit=emit,
                seq=seq,
                step_sequence=step_sequence,
            )
        if action.kind == ChatActionKindEnum.mcp_call:
            return await self.execute_mcp_action(
                action,
                turn=turn,
                root_step=root_step,
                emit=emit,
                seq=seq,
                step_sequence=step_sequence,
            )
        if action.kind == ChatActionKindEnum.sub_agent_call:
            return await self.execute_sub_agent_action(
                action,
                turn=turn,
                root_step=root_step,
                session_context=session_context,
                cancel_event=cancel_event,
                emit=emit,
                seq=seq,
                step_sequence=step_sequence,
                ensure_not_canceled=ensure_not_canceled,
            )
        if action.kind == ChatActionKindEnum.llm_response:
            return await self.execute_llm_action(
                action,
                turn=turn,
                root_step=root_step,
                session_context=session_context,
                cancel_event=cancel_event,
                send_event=send_event,
                emit=emit,
                seq=seq,
                step_sequence=step_sequence,
                ensure_not_canceled=ensure_not_canceled,
            )
        return seq

    async def execute_retrieval_action(
        self,
        action: ExecutionAction,
        *,
        turn,
        root_step,
        session_context: ChatSessionContext,
        cancel_event: asyncio.Event,
        emit: Callable[..., Awaitable[None]],
        seq: int,
        step_sequence: int,
        ensure_not_canceled: Callable[[asyncio.Event], Awaitable[None]],
    ) -> int:
        assert isinstance(action.config, KnowledgeRetrievalConfig)
        collection_ids = list(action.config.collection_ids)
        if not collection_ids:
            if self.is_required_action(action):
                retrieval_step, seq = await self.create_action_step(
                    action,
                    turn=turn,
                    root_step=root_step,
                    emit=emit,
                    seq=seq,
                    step_sequence=step_sequence,
                )
                error_message = f"required retrieval action `{action.name}` 缺少 collection_ids 配置"
                await self.repository.update_step(
                    retrieval_step,
                    status=ChatStepStatusEnum.failed,
                    started_at=datetime.now(UTC),
                    finished_at=datetime.now(UTC),
                    metrics=StepMetricPayload(latency_ms=0).model_dump(mode="json"),
                    error_message=error_message,
                )
                await emit(
                    EventNameEnum.step_failed.value,
                    StepEventPayload(step=await self.repository.summarize_step(retrieval_step)),
                    seq=seq,
                    step_id=retrieval_step.id,
                )
                raise ValueError(error_message)
            await emit(
                EventNameEnum.warning.value,
                WarningPayload(
                    message="未选择知识库，跳过检索",
                    code=ChatWarningCodeEnum.knowledge_retrieval_skipped,
                ),
                seq=seq,
                step_id=root_step.id,
            )
            return seq + 1

        retrieval_step, seq = await self.create_action_step(
            action,
            turn=turn,
            root_step=root_step,
            emit=emit,
            seq=seq,
            step_sequence=step_sequence,
        )
        started = perf_counter()
        await self.repository.update_step(
            retrieval_step,
            status=ChatStepStatusEnum.running,
            started_at=datetime.now(UTC),
        )
        retrieval_step, seq = await self.emit_step_io_data(
            retrieval_step,
            turn_id=turn.id,
            action=action,
            emit=emit,
            seq=seq,
            phase=StepIOPhaseEnum.input,
            message="知识库检索输入",
            data={
                "query": session_context.query,
                "collection_ids": collection_ids,
                "top_k": action.config.top_k,
                "required": self.is_required_action(action),
            },
        )
        await emit(
            EventNameEnum.step_started.value,
            StepEventPayload(step=await self.repository.summarize_step(retrieval_step)),
            seq=seq,
            step_id=retrieval_step.id,
        )
        seq += 1

        if self.is_required_action(action):
            session_context.artifacts.add_instruction(
                "本轮已启用必走的知识库检索。回答必须优先依据检索结果；若检索证据不足或未命中，明确说明，不要直接凭常识补全。",
            )
        stage_index = 1

        async def emit_extension_progress(
            stage: ExtensionEventStageEnum,
            message: str,
            data: dict[str, Any] | None = None,
            level: ExtensionEventLevelEnum = ExtensionEventLevelEnum.info,
        ) -> None:
            nonlocal seq, stage_index
            seq, stage_index = await self.emit_extension_stage(
                action,
                turn=turn,
                parent_step=retrieval_step,
                emit=emit,
                seq=seq,
                step_sequence=step_sequence,
                stage_index=stage_index,
                stage=stage,
                message=message,
                data=data,
                level=level,
            )

        await emit_extension_progress(
            ExtensionEventStageEnum.retrieving,
            "正在检索知识库",
            data={"collection_ids": collection_ids, "top_k": action.config.top_k},
        )

        logger.info(
            "KB_RETRIEVAL_STARTED turn_id={} step_id={} action_id={} required={} collections={} top_k={} query={!r}",
            turn.id,
            retrieval_step.id,
            action.action_id,
            self.is_required_action(action),
            collection_ids,
            action.config.top_k,
            session_context.query[:200],
        )
        logger.info(
            "Knowledge retrieval triggered: turn_id={}, action_id={}, action_name={}, "
            "required={}, collection_ids={}, top_k={}, query={!r}",
            turn.id,
            action.action_id,
            action.name,
            self.is_required_action(action),
            collection_ids,
            action.config.top_k,
            session_context.query[:200],
        )
        raw_retrieval_result = await self.retrieve_context(
            query=session_context.query,
            collection_ids=collection_ids,
            top_k=action.config.top_k,
            cancel_event=cancel_event,
            session_context=session_context,
            ensure_not_canceled=ensure_not_canceled,
            extension_event_callback=(
                emit_extension_progress if self.should_emit_extension_intermediate(action) else None
            ),
        )
        retrieval_result = self.normalize_retrieval_execution_result(
            raw_retrieval_result,
            collection_ids=collection_ids,
        )
        retrieval_results = retrieval_result.retrievals
        logger.info(
            "KB_RETRIEVAL_FINISHED turn_id={} step_id={} action_id={} "
            "searched={} hits={} missing={} inaccessible={} failed={}",
            turn.id,
            retrieval_step.id,
            action.action_id,
            retrieval_result.searched_collection_ids,
            len(retrieval_results),
            retrieval_result.missing_collection_ids,
            retrieval_result.inaccessible_collection_ids,
            retrieval_result.failed_collection_ids,
        )
        logger.info(
            "Knowledge retrieval finished: turn_id={}, action_id={}, requested={}, searched={}, "
            "hits={}, missing={}, inaccessible={}, failed={}",
            turn.id,
            action.action_id,
            retrieval_result.requested_collection_ids,
            retrieval_result.searched_collection_ids,
            len(retrieval_results),
            retrieval_result.missing_collection_ids,
            retrieval_result.inaccessible_collection_ids,
            retrieval_result.failed_collection_ids,
        )
        session_context.artifacts.add_retrieval_context(
            action,
            retrievals=retrieval_results,
            title=f"{action.name} 检索结果",
        )
        if not retrieval_result.searched_collection_ids:
            retrieval_step, seq = await self.emit_step_io_data(
                retrieval_step,
                turn_id=turn.id,
                action=action,
                emit=emit,
                seq=seq,
                phase=StepIOPhaseEnum.output,
                message="知识库检索未实际执行",
                data={
                    "requested_collection_ids": retrieval_result.requested_collection_ids,
                    "searched_collection_ids": retrieval_result.searched_collection_ids,
                    "missing_collection_ids": retrieval_result.missing_collection_ids,
                    "inaccessible_collection_ids": retrieval_result.inaccessible_collection_ids,
                    "failed_collection_ids": retrieval_result.failed_collection_ids,
                    "retrieval_hit_count": len(retrieval_results),
                },
            )
            await emit_extension_progress(
                ExtensionEventStageEnum.unavailable,
                "知识库检索未实际执行",
                data={
                    "requested_collection_ids": retrieval_result.requested_collection_ids,
                    "missing_collection_ids": retrieval_result.missing_collection_ids,
                    "inaccessible_collection_ids": retrieval_result.inaccessible_collection_ids,
                    "failed_collection_ids": retrieval_result.failed_collection_ids,
                },
                level=ExtensionEventLevelEnum.warning,
            )
            message = self.build_retrieval_diagnostics_message(
                prefix=f"知识库检索未实际执行: {action.name}",
                result=retrieval_result,
            )
            await self.repository.update_step(
                retrieval_step,
                metrics=StepMetricPayload(
                    latency_ms=int((perf_counter() - started) * 1000),
                    retrieval_hit_count=0,
                ).model_dump(mode="json"),
                status=ChatStepStatusEnum.failed if self.is_required_action(action) else ChatStepStatusEnum.completed,
                finished_at=datetime.now(UTC),
                error_message=message if self.is_required_action(action) else None,
            )
            await emit(
                EventNameEnum.warning.value,
                WarningPayload(
                    message=message,
                    code=ChatWarningCodeEnum.knowledge_retrieval_unavailable,
                ),
                seq=seq,
                step_id=retrieval_step.id,
            )
            seq += 1
            if self.is_required_action(action):
                await emit(
                    EventNameEnum.step_failed.value,
                    StepEventPayload(step=await self.repository.summarize_step(retrieval_step)),
                    seq=seq,
                    step_id=retrieval_step.id,
                )
                raise ValueError(message)
            await emit(
                EventNameEnum.step_completed.value,
                StepEventPayload(step=await self.repository.summarize_step(retrieval_step)),
                seq=seq,
                step_id=retrieval_step.id,
            )
            return seq + 1
        if self.is_required_action(action) and not retrieval_results:
            session_context.artifacts.add_instruction(
                "当前必走知识库检索未返回有效命中。请明确告知未检索到足够依据，不要编造答案。",
            )
        if not retrieval_results:
            retrieval_step, seq = await self.emit_step_io_data(
                retrieval_step,
                turn_id=turn.id,
                action=action,
                emit=emit,
                seq=seq,
                phase=StepIOPhaseEnum.output,
                message="知识库检索已执行但无命中",
                data={
                    "requested_collection_ids": retrieval_result.requested_collection_ids,
                    "searched_collection_ids": retrieval_result.searched_collection_ids,
                    "retrieval_hit_count": len(retrieval_results),
                },
            )
            await emit_extension_progress(
                ExtensionEventStageEnum.no_hit,
                "知识库检索已执行但没有命中结果",
                data={
                    "requested_collection_ids": retrieval_result.requested_collection_ids,
                    "searched_collection_ids": retrieval_result.searched_collection_ids,
                },
                level=ExtensionEventLevelEnum.warning,
            )
            message = self.build_retrieval_diagnostics_message(
                prefix=f"知识库检索已执行但无命中: {action.name}",
                result=retrieval_result,
            )
            await emit(
                EventNameEnum.warning.value,
                WarningPayload(
                    message=message,
                    code=ChatWarningCodeEnum.knowledge_retrieval_no_hit,
                ),
                seq=seq,
                step_id=retrieval_step.id,
            )
            seq += 1
        else:
            retrieval_step, seq = await self.emit_step_io_data(
                retrieval_step,
                turn_id=turn.id,
                action=action,
                emit=emit,
                seq=seq,
                phase=StepIOPhaseEnum.output,
                message="知识库检索完成",
                data={
                    "requested_collection_ids": retrieval_result.requested_collection_ids,
                    "searched_collection_ids": retrieval_result.searched_collection_ids,
                    "retrieval_hit_count": len(retrieval_results),
                },
            )
            await emit_extension_progress(
                ExtensionEventStageEnum.completed,
                "知识库检索完成",
                data={
                    "requested_collection_ids": retrieval_result.requested_collection_ids,
                    "searched_collection_ids": retrieval_result.searched_collection_ids,
                    "retrieval_hit_count": len(retrieval_results),
                },
            )
        if retrieval_results:
            retrieval_data = await self.repository.create_data(
                turn_id=turn.id,
                step_id=retrieval_step.id,
                kind=ChatDataKindEnum.reference,
                payload_type=ChatPayloadTypeEnum.retrieval_hit_list,
                payload={"items": [item.model_dump(mode="json") for item in retrieval_results]},
                is_final=True,
                metadata=self.action_step_metadata(action),
            )
            await self.repository.update_step(
                retrieval_step,
                output_data_ids=[*list(retrieval_step.output_data_ids or []), retrieval_data.id],
                metrics=StepMetricPayload(
                    latency_ms=int((perf_counter() - started) * 1000),
                    retrieval_hit_count=len(retrieval_results),
                ).model_dump(mode="json"),
                status=ChatStepStatusEnum.completed,
                finished_at=datetime.now(UTC),
            )
            await emit(
                EventNameEnum.data_created.value,
                DataEventPayload[RetrievalListPayload](
                    data=ChatDataSchema(
                        id=retrieval_data.id,
                        turn_id=turn.id,
                        step_id=retrieval_step.id,
                        kind=str(retrieval_data.kind),
                        payload_type=self.normalize_payload_type(retrieval_data.payload_type),
                        role=retrieval_data.role,
                        is_final=retrieval_data.is_final,
                        is_visible=retrieval_data.is_visible,
                        payload=RetrievalListPayload(items=retrieval_results),
                        refs=[],
                        metadata=retrieval_data.metadata or {},
                    ),
                ),
                seq=seq,
                step_id=retrieval_step.id,
                data_id=retrieval_data.id,
            )
            seq += 1
        else:
            await self.repository.update_step(
                retrieval_step,
                metrics=StepMetricPayload(
                    latency_ms=int((perf_counter() - started) * 1000),
                    retrieval_hit_count=0,
                ).model_dump(mode="json"),
                status=ChatStepStatusEnum.completed,
                finished_at=datetime.now(UTC),
            )
        await emit(
            EventNameEnum.step_completed.value,
            StepEventPayload(step=await self.repository.summarize_step(retrieval_step)),
            seq=seq,
            step_id=retrieval_step.id,
        )
        return seq + 1

    def normalize_retrieval_execution_result(
        self,
        value: list[RetrievalBlock] | dict[str, Any] | _KnowledgeRetrievalExecutionResult,
        *,
        collection_ids: list[int],
    ) -> _KnowledgeRetrievalExecutionResult:
        if isinstance(value, _KnowledgeRetrievalExecutionResult):
            return value
        if isinstance(value, list):
            return _KnowledgeRetrievalExecutionResult(
                retrievals=value,
                requested_collection_ids=list(collection_ids),
                searched_collection_ids=list(collection_ids),
            )
        return _KnowledgeRetrievalExecutionResult.model_validate(
            {
                "requested_collection_ids": list(collection_ids),
                **value,
            },
        )

    def build_retrieval_diagnostics_message(
        self,
        *,
        prefix: str,
        result: _KnowledgeRetrievalExecutionResult,
    ) -> str:
        parts = [prefix]
        parts.append(f"requested={result.requested_collection_ids}")
        if result.searched_collection_ids:
            parts.append(f"searched={result.searched_collection_ids}")
        if result.missing_collection_ids:
            parts.append(f"missing={result.missing_collection_ids}")
        if result.inaccessible_collection_ids:
            parts.append(f"inaccessible={result.inaccessible_collection_ids}")
        if result.failed_collection_ids:
            parts.append(f"failed={result.failed_collection_ids}")
        parts.append(f"hits={len(result.retrievals)}")
        return "; ".join(parts)

    async def execute_intent_action(
        self,
        action: ExecutionAction,
        *,
        turn,
        root_step,
        session_context: ChatSessionContext,
        emit: Callable[..., Awaitable[None]],
        seq: int,
        step_sequence: int,
    ) -> int:
        assert isinstance(action.config, IntentDetectionConfig)
        intent_step, seq = await self.create_action_step(
            action,
            turn=turn,
            root_step=root_step,
            emit=emit,
            seq=seq,
            step_sequence=step_sequence,
        )
        started = perf_counter()
        await self.repository.update_step(intent_step, status=ChatStepStatusEnum.running, started_at=datetime.now(UTC))
        await emit(
            EventNameEnum.step_started.value,
            StepEventPayload(step=await self.repository.summarize_step(intent_step)),
            seq=seq,
            step_id=intent_step.id,
        )
        seq += 1

        result, matched_rule = self.detect_intent(session_context.query, action.config)
        session_context.artifacts.set_intent_result(result)
        if action.config.add_to_context:
            session_context.artifacts.add_json_context(
                action,
                data=result.model_dump(mode="json"),
                title=f"{action.name} 意图识别结果",
            )
        if action.config.add_instruction and matched_rule is not None:
            for instruction in matched_rule.instructions:
                session_context.artifacts.add_instruction(instruction)
        if action.config.set_terminal_response and matched_rule is not None and matched_rule.terminal_response:
            session_context.artifacts.set_terminal_output(
                action,
                payload=MessageBundlePayload(
                    role=ChatRoleEnum.assistant,
                    blocks=[TextBlock(text=matched_rule.terminal_response)],
                ),
                metadata={
                    **self.action_step_metadata(action),
                    "intent": result.intent,
                    "terminal_source": "intent_detection",
                },
            )

        intent_payload = IntentResultPayload(result=result)
        intent_data = await self.repository.create_data(
            turn_id=turn.id,
            step_id=intent_step.id,
            kind=ChatDataKindEnum.intermediate,
            payload_type=ChatPayloadTypeEnum.intent_result,
            payload=intent_payload.model_dump(mode="json"),
            is_final=True,
            metadata={
                **self.action_step_metadata(action),
                "intent": result.intent,
                "matched_keywords": result.matched_keywords,
            },
        )
        await self.repository.update_step(
            intent_step,
            status=ChatStepStatusEnum.completed,
            finished_at=datetime.now(UTC),
            output_data_ids=[intent_data.id],
            metrics=StepMetricPayload(latency_ms=int((perf_counter() - started) * 1000)).model_dump(mode="json"),
        )
        await emit(
            EventNameEnum.data_created.value,
            DataEventPayload[IntentResultPayload](
                data=ChatDataSchema(
                    id=intent_data.id,
                    turn_id=turn.id,
                    step_id=intent_step.id,
                    kind=str(intent_data.kind),
                    payload_type=self.normalize_payload_type(intent_data.payload_type),
                    role=intent_data.role,
                    is_final=intent_data.is_final,
                    is_visible=intent_data.is_visible,
                    payload=intent_payload,
                    refs=[],
                    metadata=intent_data.metadata or {},
                ),
            ),
            seq=seq,
            step_id=intent_step.id,
            data_id=intent_data.id,
        )
        seq += 1
        await emit(
            EventNameEnum.step_completed.value,
            StepEventPayload(step=await self.repository.summarize_step(intent_step)),
            seq=seq,
            step_id=intent_step.id,
        )
        return seq + 1

    async def execute_system_prompt_action(
        self,
        action: ExecutionAction,
        *,
        turn,
        root_step,
        session_context: ChatSessionContext,
        emit: Callable[..., Awaitable[None]],
        seq: int,
        step_sequence: int,
    ) -> int:
        assert isinstance(action.config, SystemPromptConfig)
        prompt_step, seq = await self.create_action_step(
            action,
            turn=turn,
            root_step=root_step,
            emit=emit,
            seq=seq,
            step_sequence=step_sequence,
        )
        await self.repository.update_step(prompt_step, status=ChatStepStatusEnum.running, started_at=datetime.now(UTC))
        await emit(
            EventNameEnum.step_started.value,
            StepEventPayload(step=await self.repository.summarize_step(prompt_step)),
            seq=seq,
            step_id=prompt_step.id,
        )
        seq += 1

        session_context.apply_system_prompt_config(action, action.config)
        prompt_payload = session_context.artifacts.prompt_context or PromptContextPayload(
            template_key=action.config.template_key,
        )
        prompt_data = await self.repository.create_data(
            turn_id=turn.id,
            step_id=prompt_step.id,
            kind=ChatDataKindEnum.control,
            payload_type=ChatPayloadTypeEnum.prompt_context,
            payload=prompt_payload.model_dump(mode="json"),
            is_final=True,
            metadata=self.action_step_metadata(action),
        )
        await self.repository.update_step(
            prompt_step,
            status=ChatStepStatusEnum.completed,
            finished_at=datetime.now(UTC),
            output_data_ids=[prompt_data.id],
            metrics=StepMetricPayload(latency_ms=0).model_dump(mode="json"),
        )
        await emit(
            EventNameEnum.data_created.value,
            DataEventPayload[PromptContextPayload](
                data=ChatDataSchema(
                    id=prompt_data.id,
                    turn_id=turn.id,
                    step_id=prompt_step.id,
                    kind=str(prompt_data.kind),
                    payload_type=self.normalize_payload_type(prompt_data.payload_type),
                    role=prompt_data.role,
                    is_final=prompt_data.is_final,
                    is_visible=prompt_data.is_visible,
                    payload=prompt_payload,
                    refs=[],
                    metadata=prompt_data.metadata or {},
                ),
            ),
            seq=seq,
            step_id=prompt_step.id,
            data_id=prompt_data.id,
        )
        seq += 1
        await emit(
            EventNameEnum.step_completed.value,
            StepEventPayload(step=await self.repository.summarize_step(prompt_step)),
            seq=seq,
            step_id=prompt_step.id,
        )
        return seq + 1

    async def execute_function_action(
        self,
        action: ExecutionAction,
        *,
        turn,
        root_step,
        session_context: ChatSessionContext,
        emit: Callable[..., Awaitable[None]],
        seq: int,
        step_sequence: int,
    ) -> int:
        assert isinstance(action.config, FunctionCallConfig)
        function_step, seq = await self.create_action_step(
            action,
            turn=turn,
            root_step=root_step,
            emit=emit,
            seq=seq,
            step_sequence=step_sequence,
        )
        started = perf_counter()
        await self.repository.update_step(
            function_step,
            status=ChatStepStatusEnum.running,
            started_at=datetime.now(UTC),
        )
        await emit(
            EventNameEnum.step_started.value,
            StepEventPayload(step=await self.repository.summarize_step(function_step)),
            seq=seq,
            step_id=function_step.id,
        )
        seq += 1

        selected_tool_specs, forced_tool_names = await self.select_function_tools(
            action.config,
            session_context=session_context,
        )
        output_data_ids: list[int] = []
        executed = False
        for tool_spec in selected_tool_specs:
            execution = await self.function_tool_registry.execute(
                tool_spec,
                session=session_context,
                force=tool_spec.tool_name in forced_tool_names,
            )
            if execution is None:
                continue
            executed = True
            definition, result = execution
            result_mode = self.function_tool_registry.resolve_result_mode(definition, tool_spec, result)
            session_context.artifacts.add_function_execution(
                FunctionExecutionSummary(
                    tool_name=tool_spec.tool_name,
                    title=tool_spec.title or definition.title,
                    result_mode=result_mode,
                    matched=True,
                    summary=result.summary,
                    metadata=result.metadata,
                ),
            )

            if result_mode == FunctionCallResultModeEnum.context:
                if result.data is not None:
                    session_context.artifacts.add_json_context(
                        action,
                        data=result.data,
                        title=tool_spec.title or definition.title,
                        metadata={"tool_name": tool_spec.tool_name, **result.metadata},
                    )
                elif result.text:
                    session_context.artifacts.add_text_context(
                        action,
                        text=result.text,
                        title=tool_spec.title or definition.title,
                        metadata={"tool_name": tool_spec.tool_name, **result.metadata},
                    )
            elif result_mode == FunctionCallResultModeEnum.terminal:
                terminal_payload = result.terminal_payload or MessageBundlePayload(
                    role=ChatRoleEnum.assistant,
                    blocks=[TextBlock(text=result.text or result.summary or tool_spec.tool_name)],
                )
                session_context.artifacts.set_terminal_output(
                    action,
                    payload=terminal_payload,
                    metadata={**self.action_step_metadata(action), "tool_name": tool_spec.tool_name, **result.metadata},
                )

            function_payload = FunctionResultPayload(
                tool_name=tool_spec.tool_name,
                title=tool_spec.title or definition.title,
                result_mode=result_mode,
                summary=result.summary,
                content_text=result.text,
                content_data=result.data,
                terminal=result_mode == FunctionCallResultModeEnum.terminal,
                metadata=result.metadata,
            )
            function_data = await self.repository.create_data(
                turn_id=turn.id,
                step_id=function_step.id,
                kind=ChatDataKindEnum.intermediate,
                payload_type=ChatPayloadTypeEnum.function_result,
                payload=function_payload.model_dump(mode="json"),
                is_final=True,
                metadata={**self.action_step_metadata(action), "tool_name": tool_spec.tool_name},
            )
            output_data_ids.append(function_data.id)
            await emit(
                EventNameEnum.data_created.value,
                DataEventPayload[FunctionResultPayload](
                    data=ChatDataSchema(
                        id=function_data.id,
                        turn_id=turn.id,
                        step_id=function_step.id,
                        kind=str(function_data.kind),
                        payload_type=self.normalize_payload_type(function_data.payload_type),
                        role=function_data.role,
                        is_final=function_data.is_final,
                        is_visible=function_data.is_visible,
                        payload=function_payload,
                        refs=[],
                        metadata=function_data.metadata or {},
                    ),
                ),
                seq=seq,
                step_id=function_step.id,
                data_id=function_data.id,
            )
            seq += 1
            if result_mode == FunctionCallResultModeEnum.terminal and action.config.stop_after_terminal:
                break

        if not executed:
            message = f"function action `{action.name}` 未匹配到任何可执行函数"
            if action.config.fail_on_no_match:
                await self.repository.update_step(
                    function_step,
                    status=ChatStepStatusEnum.failed,
                    finished_at=datetime.now(UTC),
                    metrics=StepMetricPayload(latency_ms=int((perf_counter() - started) * 1000)).model_dump(
                        mode="json",
                    ),
                    error_message=message,
                )
                await emit(
                    EventNameEnum.step_failed.value,
                    StepEventPayload(step=await self.repository.summarize_step(function_step)),
                    seq=seq,
                    step_id=function_step.id,
                )
                raise ValueError(message)

            await self.repository.update_step(
                function_step,
                status=ChatStepStatusEnum.completed,
                finished_at=datetime.now(UTC),
                metrics=StepMetricPayload(latency_ms=int((perf_counter() - started) * 1000)).model_dump(mode="json"),
            )
            await emit(
                EventNameEnum.warning.value,
                WarningPayload(message=message, code=ChatWarningCodeEnum.function_call_skipped),
                seq=seq,
                step_id=function_step.id,
            )
            seq += 1
            await emit(
                EventNameEnum.step_completed.value,
                StepEventPayload(step=await self.repository.summarize_step(function_step)),
                seq=seq,
                step_id=function_step.id,
            )
            return seq + 1

        await self.repository.update_step(
            function_step,
            status=ChatStepStatusEnum.completed,
            finished_at=datetime.now(UTC),
            output_data_ids=output_data_ids or None,
            metrics=StepMetricPayload(latency_ms=int((perf_counter() - started) * 1000)).model_dump(mode="json"),
        )
        await emit(
            EventNameEnum.step_completed.value,
            StepEventPayload(step=await self.repository.summarize_step(function_step)),
            seq=seq,
            step_id=function_step.id,
        )
        return seq + 1

    async def select_function_tools(
        self,
        config: FunctionCallConfig,
        *,
        session_context: ChatSessionContext,
    ) -> tuple[list[FunctionToolSpec], set[str]]:
        scored_specs = self.score_function_tools(config, session_context=session_context)
        selected_specs = self.select_function_tools_heuristically(config, scored_specs=scored_specs)
        forced_tool_names: set[str] = set()

        if config.selection_mode in (SelectionModeEnum.llm, SelectionModeEnum.hybrid):
            llm_plan = await self.plan_function_tools_with_llm(
                config,
                session_context=session_context,
                scored_specs=scored_specs,
            )
            if llm_plan is not None and llm_plan.selected_tool_names:
                selected_by_name = {item.tool_name: item for item, _score in scored_specs}
                llm_selected_specs: list[FunctionToolSpec] = []
                seen_names: set[str] = set()
                for tool_name in llm_plan.selected_tool_names:
                    if tool_name in seen_names:
                        continue
                    tool_spec = selected_by_name.get(tool_name)
                    if tool_spec is None:
                        continue
                    llm_selected_specs.append(tool_spec)
                    seen_names.add(tool_name)
                    if len(llm_selected_specs) >= config.max_selected_tools:
                        break
                if llm_selected_specs:
                    return llm_selected_specs, {item.tool_name for item in llm_selected_specs}
            if config.selection_mode == SelectionModeEnum.llm:
                return selected_specs, forced_tool_names

        return selected_specs, forced_tool_names

    def score_function_tools(
        self,
        config: FunctionCallConfig,
        *,
        session_context: ChatSessionContext,
    ) -> list[tuple[FunctionToolSpec, float]]:
        scored_specs: list[tuple[FunctionToolSpec, float]] = []
        order_map = {tool_spec.tool_name: index for index, tool_spec in enumerate(config.tools)}
        for tool_spec in config.tools:
            definition = self.function_tool_registry.get(tool_spec.tool_name)
            if definition is None:
                continue
            score = self.function_tool_registry.match_score(
                tool_spec.tool_name,
                session_context.query,
                session_context,
            )
            scored_specs.append((tool_spec, score))
        return sorted(
            scored_specs,
            key=lambda item: (-item[1], order_map.get(item[0].tool_name, 999), item[0].tool_name),
        )

    def select_function_tools_heuristically(
        self,
        config: FunctionCallConfig,
        *,
        scored_specs: list[tuple[FunctionToolSpec, float]],
    ) -> list[FunctionToolSpec]:
        selected_specs: list[FunctionToolSpec] = []
        for tool_spec, score in scored_specs:
            if score <= 0:
                continue
            selected_specs.append(tool_spec)
            if len(selected_specs) >= config.max_selected_tools:
                break
        return selected_specs

    async def plan_function_tools_with_llm(
        self,
        config: FunctionCallConfig,
        *,
        session_context: ChatSessionContext,
        scored_specs: list[tuple[FunctionToolSpec, float]],
    ) -> _LLMFunctionToolPlan | None:
        planner_model_config_id = config.planner_model_config_id
        if planner_model_config_id is None or not scored_specs:
            return None
        try:
            model = await LLMModelFactory.create_by_id(planner_model_config_id)
            agent = Agent(
                model=model,
                output_type=_LLMFunctionToolPlan,
                system_prompt=(
                    "你是 function tool planner。"
                    "从候选函数工具中选择最适合当前用户问题的 0 到多个工具。"
                    "只选择真正有帮助的工具，避免冗余。"
                ),
            )
            prompt = "\n".join(
                [
                    f"用户问题: {session_context.query}",
                    f"最大可选工具数: {config.max_selected_tools}",
                    "候选工具:",
                    *[
                        (
                            f"- name={tool_spec.tool_name}; title={definition.title}; "
                            f"description={definition.description or '无'}; heuristic_score={score:.2f}"
                        )
                        for tool_spec, score in scored_specs
                        if (definition := self.function_tool_registry.get(tool_spec.tool_name)) is not None
                    ],
                ],
            )
            result = await agent.run(
                prompt,
                model_settings={"temperature": 0.0, "max_tokens": 400},
            )
            return result.output
        except Exception:
            logger.exception("Function tool planner LLM fallback to configured selection mode")
            return None

    async def execute_tool_action(
        self,
        action: ExecutionAction,
        *,
        turn,
        root_step,
        emit: Callable[..., Awaitable[None]],
        seq: int,
        step_sequence: int,
    ) -> int:
        tool_step, seq = await self.create_action_step(
            action,
            turn=turn,
            root_step=root_step,
            emit=emit,
            seq=seq,
            step_sequence=step_sequence,
        )
        assert isinstance(action.config, ToolCallConfig)
        started_at = datetime.now(UTC)
        if action.config.policy in (ToolExecutionPolicyEnum.stub, ToolExecutionPolicyEnum.optional):
            await self.repository.update_step(
                tool_step,
                status=ChatStepStatusEnum.completed,
                started_at=started_at,
                finished_at=started_at,
                metrics=StepMetricPayload(latency_ms=0).model_dump(mode="json"),
            )
            message = (
                "tool action 当前使用 stub 策略，未实际执行"
                if action.config.policy == ToolExecutionPolicyEnum.stub
                else "tool action 当前未接入执行器，已按 optional 策略跳过"
            )
            await emit(
                EventNameEnum.warning.value,
                WarningPayload(message=message, code=ChatWarningCodeEnum.tool_call_skipped),
                seq=seq,
                step_id=tool_step.id,
            )
            seq += 1
            await emit(
                EventNameEnum.step_completed.value,
                StepEventPayload(step=await self.repository.summarize_step(tool_step)),
                seq=seq,
                step_id=tool_step.id,
            )
            return seq + 1

        error_message = f"tool action `{action.name}` 要求必须执行，但当前未注册执行器"
        await self.repository.update_step(
            tool_step,
            status=ChatStepStatusEnum.failed,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            metrics=StepMetricPayload(latency_ms=0).model_dump(mode="json"),
            error_message=error_message,
        )
        await emit(
            EventNameEnum.step_failed.value,
            StepEventPayload(step=await self.repository.summarize_step(tool_step)),
            seq=seq,
            step_id=tool_step.id,
        )
        raise ValueError(error_message)

    async def execute_mcp_action(
        self,
        action: ExecutionAction,
        *,
        turn,
        root_step,
        emit: Callable[..., Awaitable[None]],
        seq: int,
        step_sequence: int,
    ) -> int:
        mcp_step, seq = await self.create_action_step(
            action,
            turn=turn,
            root_step=root_step,
            emit=emit,
            seq=seq,
            step_sequence=step_sequence,
        )
        assert isinstance(action.config, MCPCallConfig)
        error_message = f"mcp action `{action.name}` 已启用，但当前未注册执行器"
        await self.repository.update_step(
            mcp_step,
            status=ChatStepStatusEnum.failed,
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
            metrics=StepMetricPayload(latency_ms=0).model_dump(mode="json"),
            error_message=error_message,
        )
        await emit(
            EventNameEnum.step_failed.value,
            StepEventPayload(step=await self.repository.summarize_step(mcp_step)),
            seq=seq,
            step_id=mcp_step.id,
        )
        raise ValueError(error_message)

    async def execute_sub_agent_action(
        self,
        action: ExecutionAction,
        *,
        turn,
        root_step,
        session_context: ChatSessionContext,
        cancel_event: asyncio.Event,
        emit: Callable[..., Awaitable[None]],
        seq: int,
        step_sequence: int,
        ensure_not_canceled: Callable[[asyncio.Event], Awaitable[None]],
    ) -> int:
        assert isinstance(action.config, SubAgentCallConfig)
        sub_agent_step, seq = await self.create_action_step(
            action,
            turn=turn,
            root_step=root_step,
            emit=emit,
            seq=seq,
            step_sequence=step_sequence,
        )
        await self.repository.update_step(
            sub_agent_step,
            status=ChatStepStatusEnum.running,
            started_at=datetime.now(UTC),
        )
        await emit(
            EventNameEnum.step_started.value,
            StepEventPayload(step=await self.repository.summarize_step(sub_agent_step)),
            seq=seq,
            step_id=sub_agent_step.id,
        )
        seq += 1

        context = await self.build_chat_context(session_context=session_context)
        response_text, usage = await self.generate_response(
            query=session_context.query,
            llm_model_config_id=action.config.llm_model_config_id,
            context=context,
            session_context=session_context,
            cancel_event=cancel_event,
            send_delta=lambda _text: asyncio.sleep(0),
            ensure_not_canceled=ensure_not_canceled,
            system_prompt_prefix=action.config.system_prompt,
            extra_instructions=action.config.instructions,
            stream=False,
        )

        session_context.artifacts.add_text_context(
            action,
            text=response_text,
            title=f"{action.name} 子代理结论",
            metadata={"usage": usage.model_dump(mode="json")},
        )
        sub_agent_data = await self.repository.create_data(
            turn_id=turn.id,
            step_id=sub_agent_step.id,
            kind=ChatDataKindEnum.intermediate,
            payload_type=ChatPayloadTypeEnum.message_bundle,
            payload=MessageBundlePayload(
                role=ChatRoleEnum.assistant,
                blocks=[TextBlock(text=response_text)],
            ).model_dump(mode="json"),
            role=ChatRoleEnum.assistant,
            is_final=True,
            metadata={**self.action_step_metadata(action), "usage": usage.model_dump(mode="json")},
        )
        await self.repository.update_step(
            sub_agent_step,
            status=ChatStepStatusEnum.completed,
            finished_at=datetime.now(UTC),
            output_data_ids=[sub_agent_data.id],
            metrics=StepMetricPayload(
                latency_ms=0,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            ).model_dump(mode="json"),
        )
        await emit(
            EventNameEnum.data_created.value,
            DataEventPayload[MessageBundlePayload](
                data=ChatDataSchema(
                    id=sub_agent_data.id,
                    turn_id=turn.id,
                    step_id=sub_agent_step.id,
                    kind=str(sub_agent_data.kind),
                    payload_type=self.normalize_payload_type(sub_agent_data.payload_type),
                    role=sub_agent_data.role,
                    is_final=sub_agent_data.is_final,
                    is_visible=sub_agent_data.is_visible,
                    payload=MessageBundlePayload.model_validate(sub_agent_data.payload),
                    refs=[],
                    metadata=sub_agent_data.metadata or {},
                ),
            ),
            seq=seq,
            step_id=sub_agent_step.id,
            data_id=sub_agent_data.id,
        )
        seq += 1
        await emit(
            EventNameEnum.step_completed.value,
            StepEventPayload(step=await self.repository.summarize_step(sub_agent_step)),
            seq=seq,
            step_id=sub_agent_step.id,
        )
        return seq + 1

    async def execute_llm_action(
        self,
        action: ExecutionAction,
        *,
        turn,
        root_step,
        session_context: ChatSessionContext,
        cancel_event: asyncio.Event,
        send_event: EventSender,
        emit: Callable[..., Awaitable[None]],
        seq: int,
        step_sequence: int,
        ensure_not_canceled: Callable[[asyncio.Event], Awaitable[None]],
    ) -> int:
        llm_step, seq = await self.create_action_step(
            action,
            turn=turn,
            root_step=root_step,
            emit=emit,
            seq=seq,
            step_sequence=step_sequence,
        )
        await self.repository.update_turn(turn, status=ChatTurnStatusEnum.streaming)
        await self.repository.update_step(llm_step, status=ChatStepStatusEnum.streaming, started_at=datetime.now(UTC))
        await emit(
            EventNameEnum.step_started.value,
            StepEventPayload(step=await self.repository.summarize_step(llm_step)),
            seq=seq,
            step_id=llm_step.id,
        )
        seq += 1

        async def emit_message_delta(text: str) -> None:
            nonlocal seq
            event = ChatEvent[MessageDeltaPayload](
                id=f"evt_{turn.id}_{seq}",
                session_id=session_context.ws_public_session_id,
                conversation_id=session_context.conversation_id,
                turn_id=turn.id,
                seq=seq,
                event=EventNameEnum.message_delta.value,
                ts=datetime.now(UTC),
                payload=MessageDeltaPayload(text=text),
            )
            await self.repository.create_event_log(
                conversation_id=session_context.conversation_id,
                turn_id=turn.id,
                ws_session_id=session_context.ws_session_id,
                seq=seq,
                event=event.event,
                payload=event.model_dump(mode="json"),
                step_id=llm_step.id,
            )
            try:
                await send_event(event)
            except Exception:
                logger.warning("Failed to push message delta to websocket, turn_id={}, seq={}", turn.id, seq)
            seq += 1

        if session_context.artifacts.terminal_output is not None:
            return await self.finalize_terminal_output(
                turn=turn,
                llm_step=llm_step,
                action=action,
                terminal_output=session_context.artifacts.terminal_output,
                usage=session_context.artifacts.usage or UsagePayload(),
                seq=seq,
                emit=emit,
            )

        llm_model_config_id = (
            action.config.llm_model_config_id if isinstance(action.config, LLMResponseConfig) else None
        )
        context = await self.build_chat_context(session_context=session_context)
        response_text, usage = await self.generate_response(
            query=session_context.query,
            llm_model_config_id=llm_model_config_id,
            context=context,
            session_context=session_context,
            cancel_event=cancel_event,
            send_delta=emit_message_delta,
            ensure_not_canceled=ensure_not_canceled,
        )
        seq = await self.next_seq(turn.id)

        output_payload = MessageBundlePayload(role=ChatRoleEnum.assistant, blocks=[TextBlock(text=response_text)])
        session_context.artifacts.output_payload = output_payload
        session_context.artifacts.set_usage(usage)
        return await self.finalize_terminal_output(
            turn=turn,
            llm_step=llm_step,
            action=action,
            terminal_output=ActionTerminalOutput(
                action_id=action.action_id,
                action_kind=action.kind,
                action_name=action.name,
                source=action.source,
                payload=output_payload,
                metadata=self.action_step_metadata(action),
            ),
            usage=usage,
            seq=seq,
            emit=emit,
        )

    async def finalize_terminal_output(
        self,
        *,
        turn,
        llm_step,
        action: ExecutionAction,
        terminal_output: ActionTerminalOutput,
        usage: UsagePayload,
        seq: int,
        emit: Callable[..., Awaitable[None]],
        latency_ms: int = 0,
    ) -> int:
        output_payload = terminal_output.payload
        output_data = await self.repository.create_data(
            turn_id=turn.id,
            step_id=llm_step.id,
            kind=ChatDataKindEnum.output,
            payload_type=ChatPayloadTypeEnum.message_bundle,
            payload=output_payload.model_dump(mode="json"),
            role=output_payload.role,
            is_final=True,
            metadata={
                **self.action_step_metadata(action),
                "result_action_id": terminal_output.action_id,
                "result_action_kind": terminal_output.action_kind.value,
                "result_action_name": terminal_output.action_name,
                "result_action_source": terminal_output.source,
                **terminal_output.metadata,
            },
        )
        await self.repository.finalize_turn(
            turn,
            status=ChatTurnStatusEnum.completed,
            output_root_data_id=output_data.id,
            usage=usage,
            finished_at=datetime.now(UTC),
            set_head=True,
        )
        await self.repository.update_step(
            llm_step,
            status=ChatStepStatusEnum.completed,
            finished_at=datetime.now(UTC),
            output_data_ids=[output_data.id],
            metrics=StepMetricPayload(
                latency_ms=latency_ms,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            ).model_dump(mode="json"),
        )
        await emit(
            EventNameEnum.data_completed.value,
            DataEventPayload[MessageBundlePayload](
                data=ChatDataSchema(
                    id=output_data.id,
                    turn_id=turn.id,
                    step_id=llm_step.id,
                    kind=str(output_data.kind),
                    payload_type=self.normalize_payload_type(output_data.payload_type),
                    role=output_data.role,
                    is_final=output_data.is_final,
                    is_visible=output_data.is_visible,
                    payload=output_payload,
                    refs=[],
                    metadata=output_data.metadata or {},
                ),
            ),
            seq=seq,
            step_id=llm_step.id,
            data_id=output_data.id,
        )
        seq += 1
        await emit(
            EventNameEnum.message_completed.value,
            output_payload,
            seq=seq,
            step_id=llm_step.id,
            data_id=output_data.id,
        )
        seq += 1
        await emit(
            EventNameEnum.step_completed.value,
            StepEventPayload(step=await self.repository.summarize_step(llm_step)),
            seq=seq,
            step_id=llm_step.id,
        )
        seq += 1
        await self.repository.create_checkpoint(
            turn_id=turn.id,
            checkpoint_no=1,
            snapshot={"response_text": output_payload.text, "usage": usage.model_dump(mode="json")},
            latest_event_seq=seq - 1,
        )
        return seq

    def detect_intent(self, query: str, config: IntentDetectionConfig) -> tuple[IntentRecognitionResult, Any | None]:
        normalized_query = query.casefold()
        best_rule = None
        best_matches: list[str] = []
        for rule in config.intents:
            matched_keywords = [
                keyword for keyword in rule.keywords if keyword and keyword.casefold() in normalized_query
            ]
            if len(matched_keywords) > len(best_matches):
                best_rule = rule
                best_matches = matched_keywords

        if best_rule is None:
            return (
                IntentRecognitionResult(
                    intent=config.default_intent,
                    confidence=0.15,
                    matched_keywords=[],
                    description="未命中显式意图规则，使用默认意图",
                ),
                None,
            )

        return (
            IntentRecognitionResult(
                intent=best_rule.intent,
                confidence=min(1.0, 0.35 + 0.2 * len(best_matches)),
                matched_keywords=best_matches,
                description=best_rule.description,
            ),
            best_rule,
        )

    async def next_seq(self, turn_id: int) -> int:
        return await self.repository.get_last_event_seq(turn_id) + 1

    async def retrieve_context(
        self,
        *,
        query: str,
        collection_ids: list[int],
        top_k: int,
        cancel_event: asyncio.Event,
        session_context: ChatSessionContext,
        ensure_not_canceled: Callable[[asyncio.Event], Awaitable[None]],
        extension_event_callback: ExtensionEventSender | None = None,
    ) -> _KnowledgeRetrievalExecutionResult:
        results: list[RetrievalBlock] = []
        searched_collection_ids: list[int] = []
        missing_collection_ids: list[int] = []
        inaccessible_collection_ids: list[int] = []
        failed_collection_ids: list[int] = []
        for collection_id in collection_ids:
            await ensure_not_canceled(cancel_event)
            logger.info("Knowledge retrieval collection start: collection_id={}", collection_id)
            if extension_event_callback is not None:
                await extension_event_callback(
                    ExtensionEventStageEnum.collection_start,
                    f"开始检索知识库 collection {collection_id}",
                    {"collection_id": collection_id},
                )
            collection = await Collection.get_or_none(
                id=collection_id,
                deleted_at=0,
            ).prefetch_related(
                "embedding_model_config",
            )
            if not collection:
                missing_collection_ids.append(collection_id)
                logger.warning("Knowledge retrieval collection missing: collection_id={}", collection_id)
                if extension_event_callback is not None:
                    await extension_event_callback(
                        ExtensionEventStageEnum.collection_missing,
                        f"未找到 collection {collection_id}",
                        {"collection_id": collection_id},
                        ExtensionEventLevelEnum.warning,
                    )
                continue
            if not self.can_access_collection(
                collection,
                account_id=session_context.account_id,
                is_staff=session_context.is_staff,
            ):
                inaccessible_collection_ids.append(collection_id)
                logger.warning(
                    "Knowledge retrieval collection inaccessible: collection_id={}, account_id={}, is_staff={}",
                    collection_id,
                    session_context.account_id,
                    session_context.is_staff,
                )
                if extension_event_callback is not None:
                    await extension_event_callback(
                        ExtensionEventStageEnum.collection_inaccessible,
                        f"无权访问 collection {collection_id}",
                        {"collection_id": collection_id},
                        ExtensionEventLevelEnum.warning,
                    )
                continue
            helper = CollectionIndexModelHelper(collection)
            filter_clause = FilterClause(equals={"collection_id": collection.id})
            # sparse = SparseSearchClause(query_text=query, field_name="content", top_k=top_k)
            try:
                if collection.embedding_model_config:
                    embedding_model = await EmbeddingModelFactory.create(collection.embedding_model_config)
                    vector = (await embedding_model.embed_batch([query]))[0]
                    dense_model = await helper.get_dense_model()
                    index_results = await dense_model.search(
                        query_clause=DenseSearchClause(
                            vector=vector,
                            top_k=top_k,
                        ),
                        filter_clause=filter_clause,
                        limit=top_k,
                    )
                else:
                    index_results = await helper.sparse_model.search(
                        query_clause=SparseSearchClause(
                            query_text=query,
                            field_name="content",
                            top_k=top_k,
                        ),
                        filter_clause=filter_clause,
                        limit=top_k,
                    )
            except Exception as e:
                failed_collection_ids.append(collection.id)
                logger.exception(
                    "Knowledge retrieval failed for collection_id={}, query={!r}, error={}",
                    collection.id,
                    query[:200],
                    e,
                )
                if extension_event_callback is not None:
                    await extension_event_callback(
                        ExtensionEventStageEnum.collection_failed,
                        f"collection {collection.id} 检索失败",
                        {"collection_id": collection.id},
                        ExtensionEventLevelEnum.error,
                    )
                continue
            searched_collection_ids.append(collection.id)
            logger.info(
                "Knowledge retrieval collection finished: collection_id={}, hits={}",
                collection.id,
                len(index_results),
            )
            if extension_event_callback is not None:
                await extension_event_callback(
                    ExtensionEventStageEnum.collection_completed,
                    f"collection {collection.id} 检索完成",
                    {"collection_id": collection.id, "hit_count": len(index_results)},
                )

            chunk_ids = [
                getattr(index_item, "db_chunk_id", None)
                for index_item, _ in index_results
                if getattr(index_item, "db_chunk_id", None)
            ]
            document_ids = [
                getattr(index_item, "file_id", None)
                for index_item, _ in index_results
                if getattr(index_item, "file_id", None)
            ]
            chunk_map = (
                {chunk.id: chunk for chunk in await DocumentChunk.filter(id__in=chunk_ids, deleted_at=0)}
                if chunk_ids
                else {}
            )
            document_map = (
                {document.id: document for document in await Document.filter(id__in=document_ids, deleted_at=0)}
                if document_ids
                else {}
            )

            for index_item, score in index_results:
                chunk_id = getattr(index_item, "db_chunk_id", None)
                document_id = getattr(index_item, "file_id", None)
                chunk = chunk_map.get(chunk_id) if isinstance(chunk_id, int) else None
                document = document_map.get(document_id) if isinstance(document_id, int) else None
                snippet = getattr(index_item, "content", None) or getattr(index_item, "answer", None) or ""
                source_id = f"collection:{collection.id}:chunk:{getattr(index_item, 'db_chunk_id', index_item.id)}"
                results.append(
                    RetrievalBlock(
                        source_id=source_id,
                        collection_id=collection.id,  # type: ignore[arg-type]
                        document_id=document.id if document else None,
                        score=float(score),
                        snippet=snippet[:1200],
                        document=self.build_document_list(document),
                        chunk=self.build_document_chunk_list(chunk),
                        metadata={"collection_name": collection.name},
                    ),
                )
        return _KnowledgeRetrievalExecutionResult(
            retrievals=results[:top_k],
            requested_collection_ids=list(collection_ids),
            searched_collection_ids=searched_collection_ids,
            missing_collection_ids=missing_collection_ids,
            inaccessible_collection_ids=inaccessible_collection_ids,
            failed_collection_ids=failed_collection_ids,
        )

    async def generate_response(
        self,
        *,
        query: str,
        llm_model_config_id: int | None,
        context: ChatContextEnvelope,
        session_context: ChatSessionContext,
        cancel_event: asyncio.Event,
        send_delta: Callable[[str], Awaitable[None]],
        ensure_not_canceled: Callable[[asyncio.Event], Awaitable[None]],
        system_prompt_prefix: str | None = None,
        extra_instructions: list[str] | None = None,
        stream: bool = True,
    ) -> tuple[str, UsagePayload]:
        model = await (
            LLMModelFactory.create_by_id(llm_model_config_id)
            if llm_model_config_id
            else LLMModelFactory.create_default()
        )
        prompt = self.prompt_builder.build(query=query, context=context, session=session_context)
        system_prompt = "\n\n".join(
            [
                part.strip()
                for part in [
                    system_prompt_prefix or "",
                    prompt.system_prompt,
                    "\n".join(item.strip() for item in (extra_instructions or []) if item.strip()),
                ]
                if part and part.strip()
            ],
        )
        agent = Agent(model=model, system_prompt=system_prompt)
        if not stream:
            result = await agent.run(
                prompt.user_prompt,
                model_settings={"temperature": 0.2, "max_tokens": 2048},
            )
            usage = result.usage()
            return str(result.output).strip(), UsagePayload(
                requests=usage.requests,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
            )

        response_chunks: list[str] = []
        async with agent.run_stream(
            prompt.user_prompt,
            model_settings={"temperature": 0.2, "max_tokens": 2048},
        ) as result:
            async for chunk in result.stream_text(delta=True, debounce_by=None):
                await ensure_not_canceled(cancel_event)
                if not chunk:
                    continue
                response_chunks.append(chunk)
                await send_delta(chunk)
            usage = result.usage()
        return "".join(response_chunks).strip(), UsagePayload(
            requests=usage.requests,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
        )

    async def build_chat_context(self, *, session_context: ChatSessionContext) -> ChatContextEnvelope:
        session_context.sync_prompt_context()
        history = await self.repository.build_history(session_context.conversation_id)
        history_payload = [
            ChatHistoryItem(user_text=user_payload.text, assistant_text=assistant_payload.text)
            for user_payload, assistant_payload in history
        ]
        return session_context.artifacts.build_context(history_payload)

    def can_access_collection(self, collection: Collection, *, account_id: int | None, is_staff: bool) -> bool:
        if is_staff:
            return True
        if account_id is None:
            return False
        return bool(collection.is_public or collection.user_id is None or collection.user_id == account_id)

    def build_document_list(self, document: Document | None) -> DocumentList | None:
        return DocumentList.model_validate(document) if document is not None else None

    def build_document_chunk_list(self, chunk: DocumentChunk | None) -> DocumentChunkList | None:
        return DocumentChunkList.model_validate(chunk) if chunk is not None else None
