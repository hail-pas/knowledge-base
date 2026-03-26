from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from loguru import logger
from pydantic import JsonValue
from pydantic_ai import Agent

from service.chat.domain.schema import (
    ActionResultDispositionEnum,
    PersistedMCPResultPayload,
    PersistedToolResultPayload,
    EventNameEnum,
    ChatPayloadTypeEnum,
    ChatWarningCodeEnum,
    MCPCallConfig,
    ProgressLevelEnum,
    ProgressPayload,
    SelectionModeEnum,
    StrictModel,
    ToolCallConfig,
    ToolExecutionSummary,
    ToolResultModeEnum,
    ToolSpec,
)
from service.chat.execution.registry import ExecutionAction
from service.chat.execution.steps import ChatExecutionStepManager
from service.chat.runtime.mcp_executor import MCPServerRegistry
from service.chat.runtime.session import ChatSessionContext
from service.chat.runtime.tool_executor import (
    ToolRegistry,
    clear_tool_runtime_state,
    set_tool_runtime_state,
)
from service.llm_model.factory import LLMModelFactory

LiveEmitter = Callable[[str, Any], Awaitable[None]]
ToolPlanner = Callable[..., Awaitable[Any | None]]


class _LLMToolPlan(StrictModel):
    selected_tool_names: list[str]
    summary: str = ""


class ChatToolActionExecutor:
    def __init__(
        self,
        *,
        tool_registry: ToolRegistry,
        mcp_registry: MCPServerRegistry,
        step_manager: ChatExecutionStepManager,
        emit_warning: Callable[..., Awaitable[None]],
    ) -> None:
        self.tool_registry = tool_registry
        self.mcp_registry = mcp_registry
        self.step_manager = step_manager
        self.emit_warning = emit_warning

    async def plan_tools_with_llm(
        self,
        config: ToolCallConfig,
        *,
        session_context: ChatSessionContext,
        scored_specs: list[tuple[ToolSpec, float]],
    ) -> _LLMToolPlan | None:
        planner_model_config_id = config.planner_model_config_id
        if planner_model_config_id is None or not scored_specs:
            return None
        try:
            model = await LLMModelFactory.create_by_id(planner_model_config_id)
            agent = Agent(
                model=model,
                output_type=_LLMToolPlan,
                system_prompt=(
                    "你是 tool planner。"
                    "从候选本地工具中选择最适合当前用户问题的 0 到多个工具。"
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
                        if (definition := self.tool_registry.get(tool_spec.tool_name)) is not None
                    ],
                ],
            )
            result = await agent.run(prompt, model_settings={"temperature": 0.0, "max_tokens": 400})
            return result.output
        except Exception:
            logger.exception("Tool planner LLM fallback to configured selection mode")
            return None

    async def select_tools(
        self,
        action: ExecutionAction,
        *,
        session_context: ChatSessionContext,
        plan_tools_with_llm: ToolPlanner,
    ) -> list[ToolSpec]:
        assert isinstance(action.config, ToolCallConfig)
        config = action.config
        scored_specs = [
            (spec, self.tool_registry.match_score(spec.tool_name, session_context.query, session_context))
            for spec in config.tools
        ]
        if config.selection_mode == SelectionModeEnum.explicit:
            return list(config.tools)

        heuristic = [
            spec
            for spec, score in sorted(scored_specs, key=lambda item: item[1], reverse=True)
            if score > 0
        ][: config.max_selected_tools]

        if config.selection_mode == SelectionModeEnum.heuristic:
            return heuristic

        plan = await plan_tools_with_llm(
            config,
            session_context=session_context,
            scored_specs=scored_specs,
        )
        selected_names = list(plan.selected_tool_names) if plan is not None else []
        if config.selection_mode == SelectionModeEnum.llm:
            return [spec for spec in config.tools if spec.tool_name in set(selected_names)]
        if selected_names:
            return [spec for spec in config.tools if spec.tool_name in set(selected_names)]
        return heuristic

    async def execute_tool_action(
        self,
        action: ExecutionAction,
        *,
        turn,
        session_context: ChatSessionContext,
        cancel_event: asyncio.Event,
        emit: LiveEmitter,
        ensure_not_canceled: Callable[[asyncio.Event], Awaitable[None]],
        parent_step_id: int | None,
        selected_tools: list[ToolSpec],
        is_required_action: bool,
    ) -> None:
        assert isinstance(action.config, ToolCallConfig)
        if not selected_tools:
            message = f"action `{action.name}` 未匹配到可执行工具"
            if action.config.fail_on_no_match or is_required_action:
                started_step = await self.step_manager.start_action_step(
                    action,
                    turn=turn,
                    session_context=session_context,
                    emit=emit,
                    parent_step_id=parent_step_id,
                    message="工具选择为空",
                    data={"query": session_context.query},
                )
                await started_step.fail(message=message)
                raise ValueError(message)
            await self.emit_warning(emit, message=message, code=ChatWarningCodeEnum.tool_call_skipped)
            return

        for spec in selected_tools:
            definition = self.tool_registry.get(spec.tool_name)
            if definition is None:
                started_step = await self.step_manager.start_action_step(
                    action,
                    turn=turn,
                    session_context=session_context,
                    emit=emit,
                    parent_step_id=parent_step_id,
                    name=spec.tool_name,
                    message="准备执行本地工具",
                    data={"tool_name": spec.tool_name, "tool_args": spec.args or {}, "query": session_context.query},
                )
                await started_step.fail(message=f"工具 `{spec.tool_name}` 未注册")
                raise ValueError(f"工具 `{spec.tool_name}` 未注册")

            started_step = await self.step_manager.start_action_step(
                action,
                turn=turn,
                session_context=session_context,
                emit=emit,
                parent_step_id=parent_step_id,
                name=spec.tool_name,
                message="执行本地工具",
                data={
                    "tool_name": spec.tool_name,
                    "tool_args": spec.args or {},
                    "selection_mode": action.config.selection_mode,
                    "query": session_context.query,
                },
            )
            try:
                set_tool_runtime_state(
                    session_context,
                    cancel_event=cancel_event,
                    ensure_not_canceled=ensure_not_canceled,
                    progress_callback=(
                        self._build_progress_callback(emit=emit)
                        if action.metadata.runtime.emit_intermediate_events
                        else None
                    ),
                )
                execution = await self.tool_registry.execute(
                    spec.tool_name,
                    session=session_context,
                    args=spec.args or {},
                    force=True,
                )
            except Exception as exc:
                await started_step.fail_for_exception(exc, fallback=f"工具 `{spec.tool_name}` 执行失败")
            finally:
                clear_tool_runtime_state(session_context)
            if execution is None:
                await started_step.fail(message=f"工具 `{spec.tool_name}` 执行失败")
                raise ValueError(f"工具 `{spec.tool_name}` 执行失败")

            _definition, result = execution
            if is_required_action and not result.required_ok:
                await started_step.fail(
                    message=result.required_message or f"工具 `{spec.tool_name}` 未满足必需能力",
                )
                raise ValueError(result.required_message or f"工具 `{spec.tool_name}` 未满足必需能力")

            for warning in result.warnings:
                await self.emit_warning(emit, message=warning.message, code=warning.code)

            result_mode = self.tool_registry.resolve_result_mode(spec, result)
            disposition = (
                ActionResultDispositionEnum.terminal
                if result_mode == ToolResultModeEnum.terminal
                else ActionResultDispositionEnum.context
            )
            output_payload = PersistedToolResultPayload(
                tool_name=result.tool_name,
                title=result.title or definition.title,
                disposition=disposition,
                summary=result.summary,
                content_text=result.text,
                content=result.data,
                terminal=disposition == ActionResultDispositionEnum.terminal,
            )
            output_row = await started_step.complete(
                output_payload=output_payload,
                output_payload_type=ChatPayloadTypeEnum.tool_result,
            )
            session_context.artifacts.add_tool_execution(
                ToolExecutionSummary(
                    tool_name=result.tool_name,
                    title=result.title or definition.title,
                    disposition=disposition,
                    summary=result.summary,
                ),
            )
            if disposition == ActionResultDispositionEnum.context and result.retrievals:
                session_context.artifacts.add_retrieval_context(
                    action,
                    retrievals=result.retrievals,
                    title=result.title or definition.title,
                )
            else:
                started_step.apply_result(
                    title=result.title or definition.title,
                    disposition=disposition,
                    text=result.text,
                    summary=result.summary,
                    data=result.data,
                    terminal_payload=result.terminal_payload,
                    fallback=result.tool_name,
                    output_data_id=output_row.id if disposition == ActionResultDispositionEnum.terminal else None,
                )
            if disposition == ActionResultDispositionEnum.terminal and action.config.stop_after_terminal:
                break

    def _build_progress_callback(
        self,
        *,
        emit: LiveEmitter,
    ) -> Callable[..., Awaitable[None]]:
        async def callback(
            stage: str,
            message: str,
            *,
            level: ProgressLevelEnum = ProgressLevelEnum.info,
            data: JsonValue | None = None,
        ) -> None:
            await emit(
                EventNameEnum.progress.value,
                ProgressPayload[JsonValue](stage=stage, message=message, level=level, data=data),
            )

        return callback

    async def execute_mcp_action(
        self,
        action: ExecutionAction,
        *,
        turn,
        session_context: ChatSessionContext,
        emit: LiveEmitter,
        parent_step_id: int | None,
    ) -> None:
        assert isinstance(action.config, MCPCallConfig)
        if not action.config.tool_names:
            started_step = await self.step_manager.start_action_step(
                action,
                turn=turn,
                session_context=session_context,
                emit=emit,
                parent_step_id=parent_step_id,
                message="MCP 工具列表为空",
                data={"server_name": action.config.server_name},
            )
            await started_step.fail(message=f"MCP server `{action.config.server_name}` 未配置任何工具")
            raise ValueError(f"MCP server `{action.config.server_name}` 未配置任何工具")

        try:
            executions = await asyncio.wait_for(
                self.mcp_registry.execute_many(
                    server_name=action.config.server_name,
                    tool_names=list(action.config.tool_names),
                    session=session_context,
                ),
                timeout=action.config.timeout_ms / 1000,
            )
        except Exception as exc:
            started_step = await self.step_manager.start_action_step(
                action,
                turn=turn,
                session_context=session_context,
                emit=emit,
                parent_step_id=parent_step_id,
                message="执行 MCP 工具集",
                data={
                    "server_name": action.config.server_name,
                    "tool_names": list(action.config.tool_names),
                    "query": session_context.query,
                },
            )
            await started_step.fail_for_exception(
                exc,
                fallback=f"MCP server `{action.config.server_name}` 执行失败",
            )

        for definition, result in executions:
            started_step = await self.step_manager.start_action_step(
                action,
                turn=turn,
                session_context=session_context,
                emit=emit,
                parent_step_id=parent_step_id,
                name=f"{action.config.server_name}.{definition.tool_name}",
                message="执行 MCP 工具",
                data={
                    "server_name": action.config.server_name,
                    "tool_name": definition.tool_name,
                    "query": session_context.query,
                },
            )
            output_payload = PersistedMCPResultPayload(
                server_name=definition.server_name,
                tool_name=definition.tool_name,
                title=result.title or definition.title,
                disposition=result.disposition,
                summary=result.summary,
                content_text=result.text,
                content=result.data,
                terminal=result.disposition == ActionResultDispositionEnum.terminal,
            )
            output_row = await started_step.complete(
                output_payload=output_payload,
                output_payload_type=ChatPayloadTypeEnum.mcp_result,
            )
            started_step.apply_result(
                title=result.title or definition.title,
                disposition=result.disposition,
                text=result.text,
                summary=result.summary,
                data=result.data,
                terminal_payload=result.terminal_payload,
                fallback=f"{definition.server_name}.{definition.tool_name}",
                output_data_id=output_row.id if result.disposition == ActionResultDispositionEnum.terminal else None,
            )
            if result.disposition == ActionResultDispositionEnum.terminal:
                break
