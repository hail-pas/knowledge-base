from __future__ import annotations

import ast
import asyncio
import re
from datetime import datetime
from typing import Any, Callable, Awaitable
from dataclasses import field, dataclass

from pydantic import JsonValue, TypeAdapter, ValidationError
from pydantic_ai.models import Model
from pydantic_ai.tools import RunContext, Tool
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import RunUsage

from ext.ext_tortoise.models.knowledge_base import Collection
from service.chat.domain.schema import (
    ActionResultDispositionEnum,
    ChatRoleEnum,
    ChatWarningCodeEnum,
    MessageBundlePayload,
    RetrievalBlock,
    TextBlock,
    ToolResultModeEnum,
    ToolSpec,
)
from service.chat.runtime.retrieval import KnowledgeRetrievalService
from service.chat.runtime.session import ChatSessionContext

_MATH_PATTERN = re.compile(r"(?P<expr>[0-9\.\(\)\+\-\*/%\s]{3,})")
_TIME_KEYWORDS = ("时间", "几点", "日期", "今天几号", "当前时间", "what time", "date", "time")
_SESSION_KEYWORDS = ("会话上下文", "session context", "当前会话", "当前请求", "本轮上下文")
_EXECUTION_KEYWORDS = (
    "capability",
    "capabilities",
    "执行计划",
    "执行步骤",
    "执行动作",
    "能力",
    "已启用功能",
    "启用的能力",
    "能力列表",
)
_WORK_ORDER_KEYWORDS = ("创建工单", "提单", "创建任务", "报修单", "create ticket", "work order")
_RETRIEVAL_KEYWORDS = ("知识库", "文档", "资料", "检索", "查询文档", "rag", "kb")
_JSON_VALUE_ADAPTER = TypeAdapter(JsonValue)


class _DirectToolExecutionModel(Model):
    async def request(
        self,
        messages: list[Any],
        model_settings: Any,
        model_request_parameters: Any,
    ) -> Any:
        raise RuntimeError("Direct tool execution model does not support LLM requests")

    @property
    def model_name(self) -> str:
        return "direct-tool-execution"

    @property
    def system(self) -> str:
        return "internal"


_DIRECT_TOOL_EXECUTION_MODEL = _DirectToolExecutionModel()


@dataclass(slots=True)
class ToolExecutionResult:
    tool_name: str
    title: str | None
    disposition: ActionResultDispositionEnum = ActionResultDispositionEnum.context
    summary: str | None = None
    text: str | None = None
    data: JsonValue | None = None
    terminal_payload: MessageBundlePayload | None = None
    retrievals: list[RetrievalBlock] = field(default_factory=list)
    warnings: list[ToolExecutionWarning] = field(default_factory=list)
    required_ok: bool = True
    required_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolExecutionWarning:
    message: str
    code: ChatWarningCodeEnum = ChatWarningCodeEnum.warning


@dataclass(slots=True)
class ToolRuntimeState:
    cancel_event: asyncio.Event | None = None
    ensure_not_canceled: Callable[[asyncio.Event], Awaitable[None]] | None = None
    progress_callback: Callable[..., Awaitable[None]] | None = None


ToolExecutor = Callable[..., Awaitable[ToolExecutionResult] | ToolExecutionResult]
Matcher = Callable[[str, ChatSessionContext], float]
_TOOL_RUNTIME_STATE_KEY = "_tool_runtime_state"


@dataclass(slots=True)
class ToolDefinition:
    name: str
    title: str
    tool: Tool[ChatSessionContext]
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    matcher: Matcher | None = None


class ToolRegistry:
    def __init__(self, *, toolset_id: str = "chat.platform.toolset") -> None:
        self._toolset = FunctionToolset[ChatSessionContext](id=toolset_id)
        self._definitions: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        *,
        title: str,
        executor: ToolExecutor,
        description: str = "",
        metadata: dict[str, Any] | None = None,
        matcher: Matcher | None = None,
        timeout: float | None = None,
    ) -> None:
        tool_metadata = {"title": title, **(metadata or {})}
        tool = Tool[ChatSessionContext](
            executor,
            takes_ctx=True,
            name=name,
            description=description or title,
            metadata=tool_metadata,
            timeout=timeout,
        )
        self._toolset.add_tool(tool)
        self._definitions[name] = ToolDefinition(
            name=name,
            title=title,
            tool=tool,
            description=description,
            metadata=tool_metadata,
            matcher=matcher,
        )

    def get(self, name: str) -> ToolDefinition | None:
        return self._definitions.get(name)

    def list_definitions(self, names: list[str] | None = None) -> list[ToolDefinition]:
        if names is None:
            return list(self._definitions.values())
        return [definition for name in names if (definition := self.get(name)) is not None]

    def get_toolset(self) -> FunctionToolset[ChatSessionContext]:
        return self._toolset

    async def execute(
        self,
        tool_name: str,
        *,
        session: ChatSessionContext,
        args: dict[str, Any] | None = None,
        force: bool = True,
    ) -> tuple[ToolDefinition, ToolExecutionResult] | None:
        definition = self.get(tool_name)
        if definition is None:
            return None
        if not force and definition.matcher is not None and definition.matcher(session.query, session) <= 0:
            return None
        ctx = self.build_run_context(session=session, tool_name=tool_name, args=args)
        tools = await self._toolset.get_tools(ctx)
        tool = tools.get(tool_name)
        if tool is None:
            return None
        raw_result = await self._toolset.call_tool(tool_name, dict(args or {}), ctx, tool)
        return definition, self.normalize_result(
            tool_name=tool_name,
            title=definition.title,
            raw_result=raw_result,
        )

    def match_score(self, name: str, query: str, session: ChatSessionContext) -> float:
        definition = self.get(name)
        if definition is None or definition.matcher is None:
            return 0.0
        return max(0.0, float(definition.matcher(query, session)))

    def resolve_result_mode(
        self,
        spec: ToolSpec,
        result: ToolExecutionResult,
    ) -> ToolResultModeEnum:
        if spec.result_mode != ToolResultModeEnum.auto:
            return spec.result_mode
        if result.disposition == ActionResultDispositionEnum.terminal:
            return ToolResultModeEnum.terminal
        return ToolResultModeEnum.context

    def build_run_context(
        self,
        *,
        session: ChatSessionContext,
        tool_name: str,
        args: dict[str, Any] | None = None,
    ) -> RunContext[ChatSessionContext]:
        return RunContext(
            deps=session,
            model=_DIRECT_TOOL_EXECUTION_MODEL,
            usage=RunUsage(),
            prompt=session.query,
            tool_name=tool_name,
            metadata={
                "conversation_id": session.conversation_id,
                "request_id": session.request_id,
                "session_id": str(session.session_id),
                "tool_args": dict(args or {}),
            },
        )

    def normalize_result(
        self,
        *,
        tool_name: str,
        title: str,
        raw_result: Any,
    ) -> ToolExecutionResult:
        if isinstance(raw_result, ToolExecutionResult):
            return raw_result
        if isinstance(raw_result, MessageBundlePayload):
            return ToolExecutionResult(
                tool_name=tool_name,
                title=title,
                disposition=ActionResultDispositionEnum.terminal,
                summary=raw_result.text,
                text=raw_result.text,
                terminal_payload=raw_result,
            )
        if isinstance(raw_result, str):
            return ToolExecutionResult(
                tool_name=tool_name,
                title=title,
                summary=raw_result,
                text=raw_result,
            )
        if isinstance(raw_result, dict):
            disposition = raw_result.get("disposition", ActionResultDispositionEnum.context)
            terminal_payload = raw_result.get("terminal_payload")
            warnings: list[ToolExecutionWarning] = []
            for item in raw_result.get("warnings") or []:
                if not isinstance(item, dict):
                    continue
                message = str(item.get("message") or "").strip()
                if not message:
                    continue
                code = item.get("code", ChatWarningCodeEnum.warning)
                warnings.append(
                    ToolExecutionWarning(
                        message=message,
                        code=code if isinstance(code, ChatWarningCodeEnum) else ChatWarningCodeEnum(str(code)),
                    ),
                )
            return ToolExecutionResult(
                tool_name=tool_name,
                title=str(raw_result.get("title") or title),
                disposition=(
                    disposition
                    if isinstance(disposition, ActionResultDispositionEnum)
                    else ActionResultDispositionEnum(str(disposition))
                ),
                summary=raw_result.get("summary"),
                text=raw_result.get("text"),
                data=self._normalize_structured_data(tool_name=tool_name, raw_data=raw_result.get("data")),
                terminal_payload=(
                    terminal_payload
                    if isinstance(terminal_payload, MessageBundlePayload)
                    else (
                        MessageBundlePayload.model_validate(terminal_payload)
                        if isinstance(terminal_payload, dict)
                        else None
                    )
                ),
                retrievals=[
                    RetrievalBlock.model_validate(item)
                    for item in (raw_result.get("retrievals") or [])
                    if isinstance(item, dict)
                ],
                warnings=warnings,
                required_ok=bool(raw_result.get("required_ok", True)),
                required_message=(
                    str(raw_result.get("required_message")).strip()
                    if raw_result.get("required_message") is not None
                    else None
                ),
                metadata=dict(raw_result.get("metadata") or {}),
            )
        raise TypeError(f"Unsupported PydanticAI tool result for `{tool_name}`: {type(raw_result)!r}")

    def _normalize_structured_data(
        self,
        *,
        tool_name: str,
        raw_data: Any,
    ) -> JsonValue | None:
        if raw_data is None:
            return None
        try:
            return _JSON_VALUE_ADAPTER.validate_python(raw_data)
        except ValidationError as exc:
            raise TypeError(f"Tool `{tool_name}` returned non-JSON-serializable structured data") from exc


def create_default_tool_registry(
    *,
    retrieval_service: KnowledgeRetrievalService | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    kb_retrieval_service = retrieval_service or KnowledgeRetrievalService()
    registry.register(
        "current_datetime",
        title="当前时间",
        description="回答当前时间、日期、时区相关问题。",
        executor=_execute_current_datetime,
        matcher=_match_current_datetime,
    )
    registry.register(
        "calculate_expression",
        title="表达式计算",
        description="解析并计算基础数学表达式。",
        executor=_execute_calculate_expression,
        matcher=_match_calculate_expression,
    )
    registry.register(
        "session_context",
        title="会话上下文",
        description="返回当前会话、请求和解析动作摘要。",
        executor=_execute_session_context,
        matcher=_match_session_context,
    )
    registry.register(
        "execution_overview",
        title="执行概览",
        description="列出当前 turn 中已启用的 capability 和执行动作。",
        executor=_execute_execution_overview,
        matcher=_match_execution_overview,
    )
    registry.register(
        "create_work_order",
        title="模拟创建工单",
        description="模拟创建工单，用于 guarded capability 审批链路演示。",
        executor=_execute_create_work_order,
        matcher=_match_create_work_order,
    )
    registry.register(
        "lookup",
        title="平台查询",
        description="返回当前会话、Agent 和已解析动作的摘要。",
        executor=_execute_lookup,
    )
    registry.register(
        "collection_catalog",
        title="集合目录",
        description="列出当前用户可访问的知识库集合。",
        executor=_execute_collection_catalog,
    )
    registry.register(
        "knowledge_base_search",
        title="知识库检索",
        description="按指定 collection 执行知识库检索，并把命中内容补充到上下文。",
        executor=_build_knowledge_base_search_executor(kb_retrieval_service),
        matcher=_match_knowledge_base_search,
    )
    return registry


def _normalized_query(query: str) -> str:
    return query.strip().lower()


def _match_current_datetime(query: str, _: ChatSessionContext) -> float:
    normalized = _normalized_query(query)
    return 1.0 if any(keyword in normalized for keyword in _TIME_KEYWORDS) else 0.0


def _match_calculate_expression(query: str, _: ChatSessionContext) -> float:
    normalized = _normalized_query(query)
    if "计算" in normalized or "算一下" in normalized or "evaluate" in normalized:
        return 1.0 if _extract_math_expression(query) else 0.0
    return 0.85 if _extract_math_expression(query) else 0.0


def _match_session_context(query: str, _: ChatSessionContext) -> float:
    normalized = _normalized_query(query)
    return 1.0 if any(keyword in normalized for keyword in _SESSION_KEYWORDS) else 0.0


def _match_execution_overview(query: str, _: ChatSessionContext) -> float:
    normalized = _normalized_query(query)
    return 1.0 if any(keyword in normalized for keyword in _EXECUTION_KEYWORDS) else 0.0


def _match_create_work_order(query: str, _: ChatSessionContext) -> float:
    normalized = _normalized_query(query)
    return 1.0 if any(keyword in normalized for keyword in _WORK_ORDER_KEYWORDS) else 0.0


def _match_knowledge_base_search(query: str, _: ChatSessionContext) -> float:
    normalized = _normalized_query(query)
    return 1.0 if any(keyword in normalized for keyword in _RETRIEVAL_KEYWORDS) else 0.0


async def _execute_current_datetime(ctx: RunContext[ChatSessionContext]) -> ToolExecutionResult:
    now = datetime.now().astimezone()
    text = f"当前时间：{now.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    return ToolExecutionResult(
        tool_name=ctx.tool_name or "current_datetime",
        title="当前时间",
        disposition=ActionResultDispositionEnum.terminal,
        summary=text,
        text=text,
        data={
            "iso": now.isoformat(),
            "timezone": str(now.tzinfo),
            "conversation_id": ctx.deps.conversation_id,
        },
        terminal_payload=_text_payload(text),
    )


async def _execute_calculate_expression(ctx: RunContext[ChatSessionContext]) -> ToolExecutionResult:
    expression = _extract_math_expression(ctx.deps.query)
    if not expression:
        raise ValueError("未识别到可计算的表达式")
    result = _evaluate_math_expression(expression)
    text = f"计算结果：{expression} = {result}"
    return ToolExecutionResult(
        tool_name=ctx.tool_name or "calculate_expression",
        title="表达式计算",
        disposition=ActionResultDispositionEnum.terminal,
        summary=text,
        text=text,
        data={"expression": expression, "result": result},
        terminal_payload=_text_payload(text),
    )


async def _execute_session_context(ctx: RunContext[ChatSessionContext]) -> ToolExecutionResult:
    session = ctx.deps
    payload = {
        "conversation_id": session.conversation.id,
        "agent_key": session.turn_request.agent_key or session.conversation.agent_key,
        "conversation_title": session.conversation.title,
        "request_id": str(session.turn_request.request_id) if session.turn_request.request_id is not None else None,
        "resolved_action_ids": [item.action_id for item in session.resolved_actions],
        "selected_capability_keys": list(session.selected_capability_keys),
        "planner_summary": session.planner_summary,
        "prompt_placeholders": [item.value for item in session.prompt_state.selected_placeholders()],
    }
    text = (
        f"当前会话：#{session.conversation.id} {session.conversation.title}\n"
        f"本轮请求：{payload['request_id'] or '未指定'}\n"
        f"已解析执行动作数量：{len(session.resolved_actions)}\n"
        f"已选能力数量：{len(session.selected_capability_keys)}"
    )
    return ToolExecutionResult(
        tool_name=ctx.tool_name or "session_context",
        title="会话上下文",
        disposition=ActionResultDispositionEnum.terminal,
        summary="输出当前会话上下文摘要",
        text=text,
        data=payload,
        terminal_payload=_text_payload(text),
    )


async def _execute_execution_overview(ctx: RunContext[ChatSessionContext]) -> ToolExecutionResult:
    session = ctx.deps
    items = [
        {
            "action_id": item.action_id,
            "name": item.name,
            "kind": item.kind.value,
            "source": item.source,
            "priority": item.priority,
            "capability_key": item.metadata.get("capability_key"),
            "capability_category": item.metadata.get("capability_category"),
        }
        for item in session.resolved_actions
    ]
    selected_capability_lines = [
        f"- {item.name} ({item.capability_key} / {item.runtime_kind.value})"
        for item in session.selected_capabilities
    ]
    text_sections = []
    if selected_capability_lines:
        text_sections.append("当前能力计划：\n" + "\n".join(selected_capability_lines))
    text_sections.append(
        "当前执行动作：\n"
        + "\n".join(f"- {item['name']} ({item['kind']} / {item['source']})" for item in items),
    )
    text = "\n\n".join(text_sections)
    return ToolExecutionResult(
        tool_name=ctx.tool_name or "execution_overview",
        title="执行概览",
        disposition=ActionResultDispositionEnum.terminal,
        summary="输出当前 turn 的执行概览",
        text=text,
        data={
            "items": items,
            "selected_capability_keys": list(session.selected_capability_keys),
            "planner_summary": session.planner_summary,
        },
        terminal_payload=_text_payload(text),
    )


async def _execute_create_work_order(ctx: RunContext[ChatSessionContext]) -> ToolExecutionResult:
    session = ctx.deps
    title = session.query.strip()[:80] or "未命名工单"
    work_order_id = f"WO-{session.conversation_id}-{len(title)}"
    text = f"已模拟创建工单：{work_order_id}，标题：{title}"
    return ToolExecutionResult(
        tool_name=ctx.tool_name or "create_work_order",
        title="模拟创建工单",
        disposition=ActionResultDispositionEnum.terminal,
        summary="模拟完成工单创建",
        text=text,
        data={
            "work_order_id": work_order_id,
            "title": title,
            "conversation_id": session.conversation_id,
            "request_id": session.request_id,
        },
        terminal_payload=_text_payload(text),
    )


async def _execute_lookup(ctx: RunContext[ChatSessionContext]) -> ToolExecutionResult:
    session = ctx.deps
    resolved_capability_keys = list(session.selected_capability_keys)
    text = (
        f"查询结果：会话#{session.conversation_id}，"
        f"agent={session.turn_request.agent_key or session.conversation.agent_key}，"
        f"已解析动作={len(session.resolved_actions)}"
    )
    return ToolExecutionResult(
        tool_name=ctx.tool_name or "lookup",
        title="平台查询",
        disposition=ActionResultDispositionEnum.terminal,
        summary="返回当前会话和执行计划摘要",
        text=text,
        data={
            "conversation_id": session.conversation_id,
            "conversation_title": session.conversation_title,
            "agent_key": session.turn_request.agent_key or session.conversation.agent_key,
            "resolved_action_ids": [item.action_id for item in session.resolved_actions],
            "resolved_capability_keys": resolved_capability_keys,
        },
        terminal_payload=_text_payload(text),
    )


async def _execute_collection_catalog(ctx: RunContext[ChatSessionContext]) -> ToolExecutionResult:
    session = ctx.deps
    collections = await Collection.filter(deleted_at=0).order_by("id").limit(20)
    visible_items = [
        {
            "id": item.id,
            "name": item.name,
            "description": item.description,
            "is_public": item.is_public,
        }
        for item in collections
        if session.is_staff or item.is_public or item.user_id is None or str(item.user_id) == str(session.account_id)
    ]
    text = "可访问集合：" + (", ".join(item["name"] for item in visible_items) if visible_items else "无")
    return ToolExecutionResult(
        tool_name=ctx.tool_name or "collection_catalog",
        title="集合目录",
        disposition=ActionResultDispositionEnum.context,
        summary="返回当前用户可访问的集合列表",
        text=text,
        data={"items": visible_items},
    )


def _build_knowledge_base_search_executor(
    retrieval_service: KnowledgeRetrievalService,
) -> Callable[..., Awaitable[ToolExecutionResult]]:
    async def execute(
        ctx: RunContext[ChatSessionContext],
        collection_ids: list[int],
        top_k: int = 5,
    ) -> ToolExecutionResult:
        session = ctx.deps
        runtime_state = _get_tool_runtime_state(session)
        result = await retrieval_service.retrieve(
            query=session.query,
            collection_ids=list(collection_ids),
            top_k=top_k,
            session_context=session,
            cancel_event=runtime_state.cancel_event if runtime_state is not None else None,
            ensure_not_canceled=runtime_state.ensure_not_canceled if runtime_state is not None else None,
            progress_callback=runtime_state.progress_callback if runtime_state is not None else None,
        )
        warnings: list[ToolExecutionWarning] = []
        required_ok = True
        required_message: str | None = None
        summary = f"知识库检索命中 {len(result.retrievals)} 条结果"
        if not result.searched_collection_ids:
            warnings.append(
                ToolExecutionWarning(
                    message="知识库检索未实际执行",
                    code=ChatWarningCodeEnum.knowledge_retrieval_unavailable,
                ),
            )
            summary = "知识库检索未实际执行"
            required_ok = False
            required_message = "知识库检索未实际执行，无法满足必需能力"
        elif not result.retrievals:
            warnings.append(
                ToolExecutionWarning(
                    message="知识库检索无命中",
                    code=ChatWarningCodeEnum.knowledge_retrieval_no_hit,
                ),
            )
            summary = "知识库检索无命中"
        return ToolExecutionResult(
            tool_name=ctx.tool_name or "knowledge_base_search",
            title="知识库检索",
            disposition=ActionResultDispositionEnum.context,
            summary=summary,
            data={
                "requested_collection_ids": result.requested_collection_ids,
                "searched_collection_ids": result.searched_collection_ids,
                "missing_collection_ids": result.missing_collection_ids,
                "inaccessible_collection_ids": result.inaccessible_collection_ids,
                "failed_collection_ids": result.failed_collection_ids,
            },
            retrievals=list(result.retrievals),
            warnings=warnings,
            required_ok=required_ok,
            required_message=required_message,
        )

    return execute


def set_tool_runtime_state(
    session: ChatSessionContext,
    *,
    cancel_event: asyncio.Event | None = None,
    ensure_not_canceled: Callable[[asyncio.Event], Awaitable[None]] | None = None,
    progress_callback: Callable[..., Awaitable[None]] | None = None,
) -> None:
    session.set_state(
        _TOOL_RUNTIME_STATE_KEY,
        ToolRuntimeState(
            cancel_event=cancel_event,
            ensure_not_canceled=ensure_not_canceled,
            progress_callback=progress_callback,
        ),
    )


def clear_tool_runtime_state(session: ChatSessionContext) -> None:
    session.state.pop(_TOOL_RUNTIME_STATE_KEY, None)


def _get_tool_runtime_state(session: ChatSessionContext) -> ToolRuntimeState | None:
    value = session.get_state(_TOOL_RUNTIME_STATE_KEY)
    return value if isinstance(value, ToolRuntimeState) else None


def _text_payload(text: str) -> MessageBundlePayload:
    return MessageBundlePayload(
        role=ChatRoleEnum.assistant,
        blocks=[TextBlock(text=text)],
    )


def _extract_math_expression(query: str) -> str | None:
    matches = [match.group("expr").strip() for match in _MATH_PATTERN.finditer(query)]
    if not matches:
        return None
    return max(matches, key=len)


def _evaluate_math_expression(expression: str) -> float | int:
    node = ast.parse(expression, mode="eval")
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Constant,
        ast.Load,
    )
    for item in ast.walk(node):
        if not isinstance(item, allowed_nodes):
            raise ValueError("表达式包含不支持的语法")
        if isinstance(item, ast.Constant) and not isinstance(item.value, int | float):
            raise ValueError("表达式仅支持数字常量")
    result = eval(compile(node, "<math-expression>", "eval"), {"__builtins__": {}}, {})
    if isinstance(result, float) and result.is_integer():
        return int(result)
    if not isinstance(result, int | float):
        raise ValueError("表达式结果类型非法")
    return result
