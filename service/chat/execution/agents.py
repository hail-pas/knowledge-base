from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from pydantic_ai import Agent

from service.chat.domain.schema import (
    ChatContextEnvelope,
    EventNameEnum,
    ChatPayloadTypeEnum,
    ChatRoleEnum,
    LLMResponseConfig,
    MessageBundlePayload,
    MessageDeltaPayload,
    ResourceSelection,
    SubAgentCallConfig,
    SubAgentResultPayload,
    TextBlock,
    UsagePayload,
)
from service.chat.execution.registry import ExecutionAction
from service.chat.execution.steps import ChatExecutionStepManager
from service.chat.runtime.prompting import ChatPromptBuilder
from service.chat.runtime.session import ChatSessionContext
from service.chat.store.repository import ChatRepository
from service.llm_model.factory import LLMModelFactory

LiveEmitter = Callable[[str, Any], Awaitable[None]]
ActionCancellationChecker = Callable[[asyncio.Event], Awaitable[None]]
NestedActionExecutor = Callable[..., Awaitable[None]]
ResourceSelectionBuilder = Callable[..., tuple[ResourceSelection, list[ExecutionAction]]]
ChatContextBuilder = Callable[..., Awaitable[ChatContextEnvelope]]
ResponseGenerator = Callable[..., Awaitable[tuple[str, UsagePayload]]]
TerminalPolicy = Callable[[ExecutionAction], bool]


class ChatAgentActionExecutor:
    def __init__(
        self,
        *,
        repository: ChatRepository,
        prompt_builder: ChatPromptBuilder,
        step_manager: ChatExecutionStepManager,
    ) -> None:
        self.repository = repository
        self.prompt_builder = prompt_builder
        self.step_manager = step_manager

    def should_emit_terminal_from_sub_agent(self, action: ExecutionAction) -> bool:
        output_contract = str(
            action.metadata.delegation.output_contract
            if action.metadata.delegation is not None
            else "",
        ).strip().casefold()
        if not output_contract:
            return True
        return output_contract.startswith("terminal")

    async def execute_sub_agent_action(
        self,
        action: ExecutionAction,
        *,
        turn,
        session_context: ChatSessionContext,
        cancel_event: asyncio.Event,
        emit: LiveEmitter,
        ensure_not_canceled: ActionCancellationChecker,
        parent_step_id: int | None,
        build_execution_actions_from_resource_selection: ResourceSelectionBuilder,
        execute_actions: NestedActionExecutor,
        should_emit_terminal_from_sub_agent: TerminalPolicy,
    ) -> None:
        assert isinstance(action.config, SubAgentCallConfig)
        delegation = action.metadata.delegation
        started_step = await self.step_manager.start_action_step(
            action,
            turn=turn,
            session_context=session_context,
            emit=emit,
            parent_step_id=parent_step_id,
            message="执行子代理委派",
            data={
                "nested_action_count": len(action.config.actions),
                "output_contract": delegation.output_contract if delegation is not None else None,
            },
        )
        nested_selection, nested_actions = build_execution_actions_from_resource_selection(
            actions=list(action.config.actions),
            source=f"sub_agent:{action.action_id}",
            prefix=f"sub_agent:{action.action_id}",
        )
        sub_session_context = session_context.derive(
            turn_request=session_context.turn_request.model_copy(
                update={
                    "agent_key": (
                        str(delegation.mounted_agent_key)
                        if delegation is not None and delegation.mounted_agent_key
                        else action.name
                    ),
                    "resource_selection": nested_selection,
                },
            ),
            resolved_selection=nested_selection,
            resolved_actions=nested_actions,
        )
        pass_deps_fields = list(delegation.pass_deps_fields) if delegation is not None else []
        if pass_deps_fields:
            sub_session_context.copy_state_fields_from(
                session_context,
                fields=pass_deps_fields,
            )
        sub_session_context.configure_llm_execution(
            system_prompt_prefix=action.config.system_prompt,
            extra_instructions=list(action.config.instructions),
            include_history=delegation.pass_message_history if delegation is not None else True,
        )
        try:
            await execute_actions(
                turn=turn,
                session_context=sub_session_context,
                cancel_event=cancel_event,
                emit=emit,
                ensure_not_canceled=ensure_not_canceled,
                parent_step_id=started_step.step.id,
            )
        except Exception as exc:
            await started_step.fail_for_exception(exc, fallback=f"sub-agent `{action.name}` 执行失败")

        delegated_output = sub_session_context.artifacts.terminal_output
        if delegated_output is None:
            await started_step.fail(
                message=f"sub-agent `{action.name}` 未产出任何结果",
            )
            raise ValueError(f"sub-agent `{action.name}` 未产出任何结果")

        usage = sub_session_context.artifacts.usage
        terminal = should_emit_terminal_from_sub_agent(action)
        agent_key = (
            str(delegation.mounted_agent_key)
            if delegation is not None and delegation.mounted_agent_key
            else None
        )
        agent_name = (
            str(delegation.mounted_agent_name)
            if delegation is not None and delegation.mounted_agent_name
            else None
        )
        output_contract = (
            str(delegation.output_contract)
            if delegation is not None and delegation.output_contract
            else None
        )
        output_payload = SubAgentResultPayload(
            agent_key=agent_key,
            agent_name=agent_name,
            output_contract=output_contract,
            terminal=terminal,
            summary=delegated_output.payload.text,
            content_text=delegated_output.payload.text,
            usage=usage,
        )
        output_row = await started_step.complete(
            output_payload=output_payload,
            output_payload_type=ChatPayloadTypeEnum.sub_agent_result,
        )
        if terminal:
            session_context.artifacts.set_terminal_output(
                action,
                payload=delegated_output.payload,
                output_data_id=output_row.id,
            )
            if usage is not None:
                session_context.artifacts.set_usage(usage)
            return

        session_context.artifacts.add_text_context(
            action,
            text=delegated_output.payload.text,
            title=f"{action.name} 子代理结论",
        )

    async def execute_llm_action(
        self,
        action: ExecutionAction,
        *,
        turn,
        session_context: ChatSessionContext,
        cancel_event: asyncio.Event,
        emit: LiveEmitter,
        ensure_not_canceled: ActionCancellationChecker,
        parent_step_id: int | None,
        build_chat_context: ChatContextBuilder,
        generate_response: ResponseGenerator,
    ) -> None:
        if session_context.artifacts.terminal_output is not None:
            return
        started_step = await self.step_manager.start_action_step(
            action,
            turn=turn,
            session_context=session_context,
            emit=emit,
            parent_step_id=parent_step_id,
            message="生成最终回答",
            data={
                "query": session_context.query,
                "context_item_count": len(session_context.artifacts.context_items),
                "history_enabled": session_context.include_history,
            },
        )
        llm_model_config_id = (
            action.config.llm_model_config_id if isinstance(action.config, LLMResponseConfig) else None
        )
        context = await build_chat_context(
            session_context=session_context,
            include_history=session_context.include_history,
        )

        async def send_delta(text: str) -> None:
            await emit(EventNameEnum.message_delta.value, MessageDeltaPayload(text=text))

        try:
            response_text, usage = await generate_response(
                query=session_context.query,
                llm_model_config_id=llm_model_config_id,
                context=context,
                session_context=session_context,
                cancel_event=cancel_event,
                send_delta=send_delta,
                ensure_not_canceled=ensure_not_canceled,
                system_prompt_prefix=session_context.llm_system_prompt_prefix,
                extra_instructions=session_context.llm_extra_instructions,
            )
        except Exception as exc:
            await started_step.fail_for_exception(exc, fallback="LLM 响应生成失败")
        output_payload = MessageBundlePayload(
            role=ChatRoleEnum.assistant,
            blocks=[TextBlock(text=response_text)],
        )
        output_row = await started_step.complete(
            output_payload=output_payload,
            output_payload_type=ChatPayloadTypeEnum.message_bundle,
        )
        session_context.artifacts.set_terminal_output(
            action,
            payload=output_payload,
            output_data_id=output_row.id,
        )
        session_context.artifacts.set_usage(usage)

    async def generate_response(
        self,
        *,
        query: str,
        llm_model_config_id: int | None,
        context: ChatContextEnvelope,
        session_context: ChatSessionContext,
        cancel_event: asyncio.Event,
        send_delta: Callable[[str], Awaitable[None]],
        ensure_not_canceled: ActionCancellationChecker,
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
        agent = Agent(
            model=model,
            system_prompt=system_prompt,
            deps_type=ChatSessionContext,
            name=session_context.turn_request.agent_key or session_context.conversation.agent_key,
        )
        run_metadata = {
            "conversation_id": session_context.conversation_id,
            "request_id": session_context.request_id,
            "agent_key": session_context.turn_request.agent_key or session_context.conversation.agent_key,
            "session_id": str(session_context.session_id),
            "resolved_action_ids": [item.action_id for item in session_context.resolved_actions],
            "context_item_count": len(context.context_items),
            "history_turns": len(context.history),
        }
        if not stream:
            result = await agent.run(
                prompt.user_prompt,
                deps=session_context,
                metadata=run_metadata,
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
            deps=session_context,
            metadata=run_metadata,
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

    async def build_chat_context(
        self,
        *,
        session_context: ChatSessionContext,
        include_history: bool = True,
    ) -> ChatContextEnvelope:
        session_context.sync_prompt_context()
        history = await self.repository.build_history(session_context.conversation_id) if include_history else []
        return session_context.artifacts.build_context(history)
