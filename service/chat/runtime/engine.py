from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Any, Callable, Awaitable, cast
from datetime import UTC, datetime
from dataclasses import field, dataclass

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
    ChatTurnTriggerEnum,
)
from service.document.schema import DocumentList, DocumentChunkList
from service.llm_model.factory import LLMModelFactory
from service.chat.domain.schema import (
    ChatEvent,
    TextBlock,
    ChatRoleEnum,
    UsagePayload,
    EventNameEnum,
    ChatDataSchema,
    RetrievalBlock,
    ToolCallConfig,
    WarningPayload,
    ChatHistoryItem,
    DataEventPayload,
    StepEventPayload,
    TurnEventPayload,
    TurnStartRequest,
    LLMResponseConfig,
    ResourceSelection,
    StepMetricPayload,
    FunctionCallConfig,
    SystemPromptConfig,
    ChatContextEnvelope,
    ChatPayloadTypeEnum,
    ChatWarningCodeEnum,
    ConversationSummary,
    IntentResultPayload,
    MessageDeltaPayload,
    MessageBundlePayload,
    PromptContextPayload,
    RetrievalListPayload,
    FunctionResultPayload,
    IntentDetectionConfig,
    ChatCapabilityKindEnum,
    IntentRecognitionResult,
    ToolExecutionPolicyEnum,
    CapabilityTerminalOutput,
    FunctionExecutionSummary,
    KnowledgeRetrievalConfig,
    FunctionCallResultModeEnum,
)
from service.chat.runtime.session import ChatSessionContext
from service.chat.store.repository import ChatRepository
from service.chat.runtime.prompting import ChatPromptBuilder
from service.chat.capability.registry import (
    CapabilityRegistry,
    CapabilityDescriptor,
    create_default_capability_registry,
)
from service.chat.runtime.function_tools import (
    FunctionToolRegistry,
    create_default_function_tool_registry,
)
from ext.ext_tortoise.models.knowledge_base import Document, Collection, DocumentChunk


class TurnCancelledError(Exception):
    pass


EventSender = Callable[[ChatEvent[Any]], Awaitable[None]]


@dataclass
class RunningTurn:
    task: asyncio.Task[None]
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)


class ChatRuntime:
    def __init__(
        self,
        repository: ChatRepository,
        capability_registry: CapabilityRegistry | None = None,
        function_tool_registry: FunctionToolRegistry | None = None,
    ) -> None:
        self.repository = repository
        self.running_turns: dict[int, RunningTurn] = {}
        self.capability_registry = capability_registry or create_default_capability_registry()
        self.function_tool_registry = function_tool_registry or create_default_function_tool_registry()
        self.prompt_builder = ChatPromptBuilder()

    def is_running(self, turn_id: int) -> bool:
        return turn_id in self.running_turns

    async def cancel_turn(self, turn_id: int) -> bool:
        running = self.running_turns.get(turn_id)
        if not running:
            return False
        running.cancel_event.set()
        running.task.cancel()
        await self.repository.mark_turn_cancel_requested(turn_id)
        return True

    def start_turn_task(
        self,
        turn_id: int,
        coro: Callable[[asyncio.Event], Awaitable[None]],
    ) -> asyncio.Task[None]:
        cancel_event = asyncio.Event()

        async def runner() -> None:
            try:
                await coro(cancel_event)
            finally:
                self.running_turns.pop(turn_id, None)

        task = asyncio.create_task(runner())
        self.running_turns[turn_id] = RunningTurn(task=task, cancel_event=cancel_event)
        return task

    async def ensure_not_canceled(self, cancel_event: asyncio.Event) -> None:
        if cancel_event.is_set():
            raise TurnCancelledError

    def clear_current_task_cancellation(self) -> None:
        task = asyncio.current_task()
        if task is None:
            return
        while task.cancelling():
            task.uncancel()

    async def finalize_canceled_turn(
        self,
        *,
        turn,
        seq: int,
        emit: Callable[..., Awaitable[None]],
    ) -> None:
        seq = await self.next_seq(turn.id)
        await self.repository.finalize_turn(
            turn,
            status=ChatTurnStatusEnum.canceled,
            finished_at=datetime.now(UTC),
            set_head=False,
        )
        await emit(
            EventNameEnum.turn_canceled.value,
            TurnEventPayload(turn=await self.repository.summarize_turn(turn)),
            seq=seq,
        )

    def capability_step_metadata(self, descriptor: CapabilityDescriptor) -> dict[str, Any]:
        return {
            "capability_id": descriptor.capability_id,
            "capability_kind": descriptor.kind.value,
            "capability_name": descriptor.name,
            "capability_source": descriptor.source,
            "capability_profile_id": descriptor.profile_id,
            "capability_binding_id": descriptor.binding_id,
            **descriptor.metadata,
        }

    def normalize_payload_type(self, payload_type: ChatPayloadTypeEnum | str) -> ChatPayloadTypeEnum:
        return payload_type if isinstance(payload_type, ChatPayloadTypeEnum) else ChatPayloadTypeEnum(payload_type)

    def resolve_capability_pipeline(self, resource_selection: ResourceSelection) -> list[CapabilityDescriptor]:
        descriptors: list[CapabilityDescriptor] = []
        for capability in resource_selection.normalized_capabilities():
            descriptor = self.capability_registry.build(capability)
            if descriptor is None:
                logger.warning("Ignoring unknown chat capability: {}", capability.kind)
                continue
            descriptors.append(descriptor)
        return descriptors

    async def create_capability_step(
        self,
        descriptor: CapabilityDescriptor,
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
            name=descriptor.name,
            kind=descriptor.step_kind,
            sequence=step_sequence,
            metadata=self.capability_step_metadata(descriptor),
        )
        await emit(
            EventNameEnum.step_created.value,
            StepEventPayload(step=await self.repository.summarize_step(step)),
            seq=seq,
            step_id=step.id,
        )
        return step, seq + 1

    async def execute_capability(
        self,
        descriptor: CapabilityDescriptor,
        *,
        turn,
        root_step,
        session_context: ChatSessionContext,
        cancel_event: asyncio.Event,
        send_event: EventSender,
        emit: Callable[..., Awaitable[None]],
        seq: int,
        step_sequence: int,
    ) -> int:
        session_context.set_state("active_capability_id", descriptor.capability_id)
        if descriptor.kind == ChatCapabilityKindEnum.knowledge_retrieval:
            return await self.execute_retrieval_capability(
                descriptor,
                turn=turn,
                root_step=root_step,
                session_context=session_context,
                cancel_event=cancel_event,
                emit=emit,
                seq=seq,
                step_sequence=step_sequence,
            )
        if descriptor.kind == ChatCapabilityKindEnum.intent_detection:
            return await self.execute_intent_capability(
                descriptor,
                turn=turn,
                root_step=root_step,
                session_context=session_context,
                emit=emit,
                seq=seq,
                step_sequence=step_sequence,
            )
        if descriptor.kind == ChatCapabilityKindEnum.system_prompt:
            return await self.execute_system_prompt_capability(
                descriptor,
                turn=turn,
                root_step=root_step,
                session_context=session_context,
                emit=emit,
                seq=seq,
                step_sequence=step_sequence,
            )
        if descriptor.kind == ChatCapabilityKindEnum.function_call:
            return await self.execute_function_capability(
                descriptor,
                turn=turn,
                root_step=root_step,
                session_context=session_context,
                emit=emit,
                seq=seq,
                step_sequence=step_sequence,
            )
        if descriptor.kind == ChatCapabilityKindEnum.tool_call:
            return await self.execute_tool_capability(
                descriptor,
                turn=turn,
                root_step=root_step,
                emit=emit,
                seq=seq,
                step_sequence=step_sequence,
            )
        if descriptor.kind == ChatCapabilityKindEnum.mcp_call:
            return await self.execute_mcp_capability(
                descriptor,
                turn=turn,
                root_step=root_step,
                emit=emit,
                seq=seq,
                step_sequence=step_sequence,
            )
        if descriptor.kind == ChatCapabilityKindEnum.llm_response:
            return await self.execute_llm_capability(
                descriptor,
                turn=turn,
                root_step=root_step,
                session_context=session_context,
                cancel_event=cancel_event,
                send_event=send_event,
                emit=emit,
                seq=seq,
                step_sequence=step_sequence,
            )
        return seq

    async def execute_retrieval_capability(
        self,
        descriptor: CapabilityDescriptor,
        *,
        turn,
        root_step,
        session_context: ChatSessionContext,
        cancel_event: asyncio.Event,
        emit: Callable[..., Awaitable[None]],
        seq: int,
        step_sequence: int,
    ) -> int:
        assert isinstance(descriptor.config, KnowledgeRetrievalConfig)
        collection_ids = list(descriptor.config.collection_ids)
        top_k = descriptor.config.top_k
        if not collection_ids:
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

        retrieval_step, seq = await self.create_capability_step(
            descriptor,
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
        await emit(
            EventNameEnum.step_started.value,
            StepEventPayload(step=await self.repository.summarize_step(retrieval_step)),
            seq=seq,
            step_id=retrieval_step.id,
        )
        seq += 1

        retrieval_results = await self.retrieve_context(
            query=session_context.query,
            collection_ids=collection_ids,
            top_k=top_k,
            cancel_event=cancel_event,
            session_context=session_context,
        )
        session_context.artifacts.add_retrieval_context(
            descriptor,
            retrievals=retrieval_results,
            title=f"{descriptor.name} 检索结果",
        )
        if retrieval_results:
            retrieval_data = await self.repository.create_data(
                turn_id=turn.id,
                step_id=retrieval_step.id,
                kind=ChatDataKindEnum.reference,
                payload_type=ChatPayloadTypeEnum.retrieval_hit_list,
                payload={"items": [item.model_dump(mode="json") for item in retrieval_results]},
                is_final=True,
                metadata=self.capability_step_metadata(descriptor),
            )
            await self.repository.update_step(
                retrieval_step,
                output_data_ids=[retrieval_data.id],
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

    async def execute_intent_capability(
        self,
        descriptor: CapabilityDescriptor,
        *,
        turn,
        root_step,
        session_context: ChatSessionContext,
        emit: Callable[..., Awaitable[None]],
        seq: int,
        step_sequence: int,
    ) -> int:
        assert isinstance(descriptor.config, IntentDetectionConfig)
        intent_step, seq = await self.create_capability_step(
            descriptor,
            turn=turn,
            root_step=root_step,
            emit=emit,
            seq=seq,
            step_sequence=step_sequence,
        )
        started = perf_counter()
        await self.repository.update_step(
            intent_step,
            status=ChatStepStatusEnum.running,
            started_at=datetime.now(UTC),
        )
        await emit(
            EventNameEnum.step_started.value,
            StepEventPayload(step=await self.repository.summarize_step(intent_step)),
            seq=seq,
            step_id=intent_step.id,
        )
        seq += 1

        result, matched_rule = self.detect_intent(session_context.query, descriptor.config)
        session_context.artifacts.set_intent_result(result)
        if descriptor.config.add_to_context:
            session_context.artifacts.add_json_context(
                descriptor,
                data=result.model_dump(mode="json"),
                title=f"{descriptor.name} 意图识别结果",
            )
        if descriptor.config.add_instruction and matched_rule is not None:
            for instruction in matched_rule.instructions:
                session_context.artifacts.add_instruction(instruction)
        if descriptor.config.set_terminal_response and matched_rule is not None and matched_rule.terminal_response:
            session_context.artifacts.set_terminal_output(
                descriptor,
                payload=MessageBundlePayload(
                    role=ChatRoleEnum.assistant,
                    blocks=[TextBlock(text=matched_rule.terminal_response)],
                ),
                metadata={
                    **self.capability_step_metadata(descriptor),
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
                **self.capability_step_metadata(descriptor),
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

    async def execute_system_prompt_capability(
        self,
        descriptor: CapabilityDescriptor,
        *,
        turn,
        root_step,
        session_context: ChatSessionContext,
        emit: Callable[..., Awaitable[None]],
        seq: int,
        step_sequence: int,
    ) -> int:
        assert isinstance(descriptor.config, SystemPromptConfig)
        prompt_step, seq = await self.create_capability_step(
            descriptor,
            turn=turn,
            root_step=root_step,
            emit=emit,
            seq=seq,
            step_sequence=step_sequence,
        )
        started_at = datetime.now(UTC)
        await self.repository.update_step(
            prompt_step,
            status=ChatStepStatusEnum.running,
            started_at=started_at,
        )
        await emit(
            EventNameEnum.step_started.value,
            StepEventPayload(step=await self.repository.summarize_step(prompt_step)),
            seq=seq,
            step_id=prompt_step.id,
        )
        seq += 1

        session_context.apply_system_prompt_config(descriptor, descriptor.config)
        prompt_payload = session_context.artifacts.prompt_context or PromptContextPayload(
            template_key=descriptor.config.template_key,
        )
        prompt_data = await self.repository.create_data(
            turn_id=turn.id,
            step_id=prompt_step.id,
            kind=ChatDataKindEnum.control,
            payload_type=ChatPayloadTypeEnum.prompt_context,
            payload=prompt_payload.model_dump(mode="json"),
            is_final=True,
            metadata=self.capability_step_metadata(descriptor),
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

    async def execute_function_capability(
        self,
        descriptor: CapabilityDescriptor,
        *,
        turn,
        root_step,
        session_context: ChatSessionContext,
        emit: Callable[..., Awaitable[None]],
        seq: int,
        step_sequence: int,
    ) -> int:
        assert isinstance(descriptor.config, FunctionCallConfig)
        function_step, seq = await self.create_capability_step(
            descriptor,
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

        output_data_ids: list[int] = []
        executed = False
        for tool_spec in descriptor.config.tools:
            execution = await self.function_tool_registry.execute(tool_spec, session=session_context)
            if execution is None:
                continue
            executed = True
            definition, result = execution
            result_mode = self.function_tool_registry.resolve_result_mode(definition, tool_spec, result)
            execution_summary = FunctionExecutionSummary(
                tool_name=tool_spec.tool_name,
                title=tool_spec.title or definition.title,
                result_mode=result_mode,
                matched=True,
                summary=result.summary,
                metadata=result.metadata,
            )
            session_context.artifacts.add_function_execution(execution_summary)

            if result_mode == FunctionCallResultModeEnum.context:
                if result.data is not None:
                    session_context.artifacts.add_json_context(
                        descriptor,
                        data=result.data,
                        title=tool_spec.title or definition.title,
                        metadata={"tool_name": tool_spec.tool_name, **result.metadata},
                    )
                elif result.text:
                    session_context.artifacts.add_text_context(
                        descriptor,
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
                    descriptor,
                    payload=terminal_payload,
                    metadata={
                        **self.capability_step_metadata(descriptor),
                        "tool_name": tool_spec.tool_name,
                        **result.metadata,
                    },
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
                metadata={
                    **self.capability_step_metadata(descriptor),
                    "tool_name": tool_spec.tool_name,
                },
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

            if result_mode == FunctionCallResultModeEnum.terminal and descriptor.config.stop_after_terminal:
                break

        if not executed:
            message = f"function capability `{descriptor.name}` 未匹配到任何可执行函数"
            if descriptor.config.fail_on_no_match:
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
                metrics=StepMetricPayload(latency_ms=int((perf_counter() - started) * 1000)).model_dump(
                    mode="json",
                ),
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

    def detect_intent(
        self,
        query: str,
        config: IntentDetectionConfig,
    ) -> tuple[IntentRecognitionResult, Any | None]:
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

        confidence = min(1.0, 0.35 + 0.2 * len(best_matches))
        return (
            IntentRecognitionResult(
                intent=best_rule.intent,
                confidence=confidence,
                matched_keywords=best_matches,
                description=best_rule.description,
            ),
            best_rule,
        )

    async def execute_tool_capability(
        self,
        descriptor: CapabilityDescriptor,
        *,
        turn,
        root_step,
        emit: Callable[..., Awaitable[None]],
        seq: int,
        step_sequence: int,
    ) -> int:
        tool_step, seq = await self.create_capability_step(
            descriptor,
            turn=turn,
            root_step=root_step,
            emit=emit,
            seq=seq,
            step_sequence=step_sequence,
        )
        assert isinstance(descriptor.config, ToolCallConfig)
        started_at = datetime.now(UTC)
        if descriptor.config.policy in (
            ToolExecutionPolicyEnum.stub,
            ToolExecutionPolicyEnum.optional,
        ):
            await self.repository.update_step(
                tool_step,
                status=ChatStepStatusEnum.completed,
                started_at=started_at,
                finished_at=started_at,
                metrics=StepMetricPayload(latency_ms=0).model_dump(mode="json"),
            )
            message = (
                "tool capability 当前使用 stub 策略，未实际执行"
                if descriptor.config.policy == ToolExecutionPolicyEnum.stub
                else "tool capability 当前未接入执行器，已按 optional 策略跳过"
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

        error_message = f"tool capability `{descriptor.name}` 要求必须执行，但当前未注册执行器"
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

    async def execute_mcp_capability(
        self,
        descriptor: CapabilityDescriptor,
        *,
        turn,
        root_step,
        emit: Callable[..., Awaitable[None]],
        seq: int,
        step_sequence: int,
    ) -> int:
        mcp_step, seq = await self.create_capability_step(
            descriptor,
            turn=turn,
            root_step=root_step,
            emit=emit,
            seq=seq,
            step_sequence=step_sequence,
        )
        error_message = f"mcp capability `{descriptor.name}` 已启用，但当前未注册执行器"
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

    async def execute_llm_capability(
        self,
        descriptor: CapabilityDescriptor,
        *,
        turn,
        root_step,
        session_context: ChatSessionContext,
        cancel_event: asyncio.Event,
        send_event: EventSender,
        emit: Callable[..., Awaitable[None]],
        seq: int,
        step_sequence: int,
    ) -> int:
        llm_step, seq = await self.create_capability_step(
            descriptor,
            turn=turn,
            root_step=root_step,
            emit=emit,
            seq=seq,
            step_sequence=step_sequence,
        )
        await self.repository.update_turn(turn, status=ChatTurnStatusEnum.streaming)
        await self.repository.update_step(
            llm_step,
            status=ChatStepStatusEnum.streaming,
            started_at=datetime.now(UTC),
        )
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
                logger.warning(
                    "Failed to push message delta to websocket, event is already persisted: turn_id={}, seq={}",
                    turn.id,
                    seq,
                )
            seq += 1

        llm_started = perf_counter()
        if session_context.artifacts.terminal_output is not None:
            return await self.finalize_terminal_output(
                turn=turn,
                llm_step=llm_step,
                descriptor=descriptor,
                terminal_output=session_context.artifacts.terminal_output,
                usage=session_context.artifacts.usage or UsagePayload(),
                seq=seq,
                emit=emit,
            )
        llm_model_config_id = (
            descriptor.config.llm_model_config_id
            if isinstance(
                descriptor.config,
                LLMResponseConfig,
            )
            else None
        )
        context = await self.build_chat_context(session_context=session_context)
        response_text, usage = await self.generate_response(
            query=session_context.query,
            llm_model_config_id=llm_model_config_id,
            context=context,
            session_context=session_context,
            cancel_event=cancel_event,
            send_delta=emit_message_delta,
        )
        seq = await self.next_seq(turn.id)

        output_payload = MessageBundlePayload(
            role=ChatRoleEnum.assistant,
            blocks=[TextBlock(text=response_text)],
        )
        session_context.artifacts.output_payload = output_payload
        session_context.artifacts.set_usage(usage)
        terminal_output = CapabilityTerminalOutput(
            capability_id=descriptor.capability_id,
            capability_kind=descriptor.kind,
            capability_name=descriptor.name,
            source=descriptor.source,
            payload=output_payload,
            metadata=self.capability_step_metadata(descriptor),
        )
        return await self.finalize_terminal_output(
            turn=turn,
            llm_step=llm_step,
            descriptor=descriptor,
            terminal_output=terminal_output,
            usage=usage,
            seq=seq,
            emit=emit,
            latency_ms=int((perf_counter() - llm_started) * 1000),
        )

    async def finalize_terminal_output(
        self,
        *,
        turn,
        llm_step,
        descriptor: CapabilityDescriptor,
        terminal_output: CapabilityTerminalOutput,
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
                **self.capability_step_metadata(descriptor),
                "result_capability_id": terminal_output.capability_id,
                "result_capability_kind": terminal_output.capability_kind.value,
                "result_capability_name": terminal_output.capability_name,
                "result_capability_source": terminal_output.source,
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

    async def execute_turn(
        self,
        *,
        ws_session_id: int | None,
        ws_public_session_id: str | None,
        conversation: ConversationSummary,
        turn_request: TurnStartRequest,
        account_id: int | None,
        is_staff: bool,
        send_event: EventSender,
    ) -> int:
        conversation_id = turn_request.conversation_id
        if conversation_id is None:
            raise ValueError("conversation_id is required before starting a turn")

        resolved_capabilities = self.resolve_capability_pipeline(turn_request.resource_selection)
        session_context = ChatSessionContext(
            account_id=account_id,
            is_staff=is_staff,
            ws_session_id=ws_session_id,
            ws_public_session_id=ws_public_session_id,
            conversation=conversation,
            turn_request=turn_request,
            resolved_selection=turn_request.resource_selection,
            resolved_capabilities=resolved_capabilities,
        )

        turn = await self.repository.create_turn(
            conversation_id=conversation_id,
            request_id=turn_request.request_id,
            trigger=ChatTurnTriggerEnum.user,
            resource_selection=turn_request.resource_selection,
            metadata=turn_request.metadata,
        )

        async def emit(
            event_name: str,
            payload: Any,
            *,
            seq: int,
            step_id: int | None = None,
            data_id: int | None = None,
        ) -> None:
            event = ChatEvent[Any](
                id=f"evt_{turn.id}_{seq}",
                session_id=ws_public_session_id,
                conversation_id=conversation_id,
                turn_id=turn.id,
                seq=seq,
                event=event_name,
                ts=datetime.now(UTC),
                payload=payload,
            )
            await self.repository.create_event_log(
                conversation_id=conversation_id,
                turn_id=turn.id,
                ws_session_id=ws_session_id,
                seq=seq,
                event=event_name,
                payload=event.model_dump(mode="json"),
                step_id=step_id,
                data_id=data_id,
            )
            try:
                await send_event(event)
            except Exception:
                logger.warning(
                    "Failed to push event to websocket, event is already persisted: turn_id={}, event={}, seq={}",
                    turn.id,
                    event_name,
                    seq,
                )

        async def run(cancel_event: asyncio.Event) -> None:
            seq = 1
            root_step = await self.repository.create_step(
                turn_id=turn.id,
                name="turn_root",
                kind=ChatStepKindEnum.system,
                sequence=0,
            )
            await self.repository.update_step(root_step, root_step_id=root_step.id)
            await self.repository.update_turn(
                turn,
                status=ChatTurnStatusEnum.accepted,
                root_step_id=root_step.id,
                started_at=datetime.now(UTC),
            )

            input_data = await self.repository.create_data(
                turn_id=turn.id,
                step_id=root_step.id,
                kind=ChatDataKindEnum.input,
                payload_type=ChatPayloadTypeEnum.message_bundle,
                payload=turn_request.input.model_dump(mode="json"),
                role=turn_request.input.role,
                is_final=True,
            )
            await self.repository.update_turn(turn, input_root_data_id=input_data.id)
            await self.repository.update_step(root_step, input_data_ids=[input_data.id])

            await emit(
                EventNameEnum.turn_accepted.value,
                TurnEventPayload(turn=await self.repository.summarize_turn(turn)),
                seq=seq,
            )
            seq += 1
            await self.repository.update_turn(turn, status=ChatTurnStatusEnum.running)
            await emit(
                EventNameEnum.turn_started.value,
                TurnEventPayload(turn=await self.repository.summarize_turn(turn)),
                seq=seq,
            )
            seq += 1

            try:
                await self.ensure_not_canceled(cancel_event)
                for index, descriptor in enumerate(session_context.resolved_capabilities, start=1):
                    await self.ensure_not_canceled(cancel_event)
                    if (
                        session_context.artifacts.terminal_output is not None
                        and descriptor.kind != ChatCapabilityKindEnum.llm_response
                    ):
                        continue
                    seq = await self.execute_capability(
                        descriptor,
                        turn=turn,
                        root_step=root_step,
                        session_context=session_context,
                        cancel_event=cancel_event,
                        send_event=send_event,
                        emit=emit,
                        seq=seq,
                        step_sequence=index * 10,
                    )
                session_context.set_state("active_capability_id", None)

                if (
                    session_context.artifacts.output_payload is None
                    and session_context.artifacts.terminal_output is None
                ):
                    raise ValueError("No capability produced final chat output")

                await emit(
                    EventNameEnum.turn_completed.value,
                    TurnEventPayload(turn=await self.repository.summarize_turn(turn)),
                    seq=seq,
                )
            except TurnCancelledError:
                await self.finalize_canceled_turn(turn=turn, seq=seq, emit=emit)
            except asyncio.CancelledError:
                self.clear_current_task_cancellation()
                await self.finalize_canceled_turn(turn=turn, seq=seq, emit=emit)
            except Exception as exc:
                logger.exception("Chat turn failed: turn_id={}", turn.id)
                await self.repository.finalize_turn(
                    turn,
                    status=ChatTurnStatusEnum.failed,
                    error_message=str(exc),
                    finished_at=datetime.now(UTC),
                    set_head=False,
                )
                seq = await self.next_seq(turn.id)
                await emit(
                    EventNameEnum.turn_failed.value,
                    TurnEventPayload(turn=await self.repository.summarize_turn(turn)),
                    seq=seq,
                )

        self.start_turn_task(turn.id, run)
        return turn.id

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
    ) -> list[RetrievalBlock]:
        results: list[RetrievalBlock] = []
        for collection_id in collection_ids:
            await self.ensure_not_canceled(cancel_event)
            collection = await Collection.get_or_none(id=collection_id, deleted_at=0).prefetch_related(
                "embedding_model_config",
            )
            if not collection:
                continue
            if not self.can_access_collection(
                collection,
                account_id=session_context.account_id,
                is_staff=session_context.is_staff,
            ):
                continue
            helper = CollectionIndexModelHelper(collection)
            filter_clause = FilterClause(equals={"collection_id": collection.id})
            sparse = SparseSearchClause(query_text=query, top_k=top_k)
            try:
                if collection.embedding_model_config:
                    embedding_model = await EmbeddingModelFactory.create(collection.embedding_model_config)
                    vector = (await embedding_model.embed_batch([query]))[0]
                    dense_model = await helper.get_dense_model()
                    index_results = await dense_model.search(
                        query_clause=HybridSearchClause(
                            dense=DenseSearchClause(vector=vector, top_k=top_k),
                            sparse=sparse,
                        ),
                        filter_clause=filter_clause,
                        limit=top_k,
                    )
                else:
                    index_results = await helper.sparse_model.search(
                        query_clause=sparse,
                        filter_clause=filter_clause,
                        limit=top_k,
                    )
            except Exception:
                logger.exception("Retrieval failed for collection_id={}", collection.id)
                continue

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
                results.append(
                    RetrievalBlock(
                        source_id=(
                            f"collection:{collection.id}:chunk:{getattr(index_item, 'db_chunk_id', index_item.id)}"
                        ),
                        collection_id=collection.id,  # type: ignore[arg-type]
                        document_id=document.id if document else None,
                        score=float(score),
                        snippet=snippet[:1200],
                        document=self.build_document_list(document),
                        chunk=self.build_document_chunk_list(chunk),
                        metadata={"collection_name": collection.name},
                    ),
                )
        return results[:top_k]

    def can_access_collection(
        self,
        collection: Collection,
        *,
        account_id: int | None,
        is_staff: bool,
    ) -> bool:
        if is_staff:
            return True
        if account_id is None:
            return False
        return bool(collection.is_public or collection.user_id is None or collection.user_id == account_id)

    def build_document_list(self, document: Document | None) -> DocumentList | None:
        if document is None:
            return None
        return DocumentList.model_validate(document)

    def build_document_chunk_list(self, chunk: DocumentChunk | None) -> DocumentChunkList | None:
        if chunk is None:
            return None
        return DocumentChunkList.model_validate(chunk)

    async def generate_response(
        self,
        *,
        query: str,
        llm_model_config_id: int | None,
        context: ChatContextEnvelope,
        session_context: ChatSessionContext,
        cancel_event: asyncio.Event,
        send_delta: Callable[[str], Awaitable[None]],
    ) -> tuple[str, UsagePayload]:
        model = (
            await LLMModelFactory.create_by_id(llm_model_config_id)
            if llm_model_config_id
            else await LLMModelFactory.create_default()
        )
        prompt = self.prompt_builder.build(query=query, context=context, session=session_context)
        agent = Agent(model=model, system_prompt=prompt.system_prompt)
        response_chunks: list[str] = []
        async with agent.run_stream(
            prompt.user_prompt,
            model_settings={"temperature": 0.2, "max_tokens": 2048},
        ) as result:
            async for chunk in result.stream_text(delta=True, debounce_by=None):
                await self.ensure_not_canceled(cancel_event)
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

    async def build_chat_context(
        self,
        *,
        session_context: ChatSessionContext,
    ) -> ChatContextEnvelope:
        session_context.sync_prompt_context()
        history = await self.repository.build_history(session_context.conversation_id)
        history_payload = [
            ChatHistoryItem(user_text=user_payload.text, assistant_text=assistant_payload.text)
            for user_payload, assistant_payload in history
        ]
        return session_context.artifacts.build_context(history_payload)
