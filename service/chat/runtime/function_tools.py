from __future__ import annotations

import re
import ast
from typing import Any, Callable, Awaitable
from datetime import datetime
from dataclasses import field, dataclass

from service.chat.domain.schema import (
    TextBlock,
    ChatRoleEnum,
    FunctionToolSpec,
    MessageBundlePayload,
    FunctionCallResultModeEnum,
)
from service.chat.runtime.session import ChatSessionContext

_MATH_PATTERN = re.compile(r"(?P<expr>[0-9\.\(\)\+\-\*/%\s]{3,})")
_TIME_KEYWORDS = ("时间", "几点", "日期", "今天几号", "当前时间", "what time", "date", "time")
_SESSION_KEYWORDS = ("会话上下文", "session context", "当前会话", "当前请求", "本轮上下文")
_CAPABILITY_KEYWORDS = ("capability", "capabilities", "能力", "已启用功能", "启用的能力", "能力列表")


@dataclass(slots=True)
class FunctionToolExecutionResult:
    tool_name: str
    title: str | None
    summary: str | None = None
    text: str | None = None
    data: dict[str, Any] | None = None
    prefer_terminal: bool = False
    terminal_payload: MessageBundlePayload | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


Matcher = Callable[[str, ChatSessionContext], float]
Executor = Callable[[ChatSessionContext, FunctionToolSpec], Awaitable[FunctionToolExecutionResult]]


@dataclass(slots=True)
class FunctionToolDefinition:
    name: str
    title: str
    matcher: Matcher
    executor: Executor
    default_result_mode: FunctionCallResultModeEnum = FunctionCallResultModeEnum.context


class FunctionToolRegistry:
    def __init__(self) -> None:
        self._definitions: dict[str, FunctionToolDefinition] = {}

    def register(
        self,
        name: str,
        *,
        title: str,
        matcher: Matcher,
        executor: Executor,
        default_result_mode: FunctionCallResultModeEnum = FunctionCallResultModeEnum.context,
    ) -> None:
        self._definitions[name] = FunctionToolDefinition(
            name=name,
            title=title,
            matcher=matcher,
            executor=executor,
            default_result_mode=default_result_mode,
        )

    def get(self, name: str) -> FunctionToolDefinition | None:
        return self._definitions.get(name)

    def match_score(self, name: str, query: str, session: ChatSessionContext) -> float:
        definition = self.get(name)
        if definition is None:
            return 0.0
        return max(0.0, float(definition.matcher(query, session)))

    async def execute(
        self,
        spec: FunctionToolSpec,
        *,
        session: ChatSessionContext,
    ) -> tuple[FunctionToolDefinition, FunctionToolExecutionResult] | None:
        definition = self.get(spec.tool_name)
        if definition is None:
            return None
        if definition.matcher(session.query, session) <= 0:
            return None
        return definition, await definition.executor(session, spec)

    def resolve_result_mode(
        self,
        definition: FunctionToolDefinition,
        spec: FunctionToolSpec,
        result: FunctionToolExecutionResult,
    ) -> FunctionCallResultModeEnum:
        if spec.result_mode != FunctionCallResultModeEnum.auto:
            return spec.result_mode
        if result.prefer_terminal:
            return FunctionCallResultModeEnum.terminal
        return definition.default_result_mode


def create_default_function_tool_registry() -> FunctionToolRegistry:
    registry = FunctionToolRegistry()
    registry.register(
        "current_datetime",
        title="当前时间",
        matcher=_match_current_datetime,
        executor=_execute_current_datetime,
        default_result_mode=FunctionCallResultModeEnum.terminal,
    )
    registry.register(
        "calculate_expression",
        title="表达式计算",
        matcher=_match_calculate_expression,
        executor=_execute_calculate_expression,
        default_result_mode=FunctionCallResultModeEnum.terminal,
    )
    registry.register(
        "session_context",
        title="会话上下文",
        matcher=_match_session_context,
        executor=_execute_session_context,
        default_result_mode=FunctionCallResultModeEnum.terminal,
    )
    registry.register(
        "capability_overview",
        title="能力概览",
        matcher=_match_capability_overview,
        executor=_execute_capability_overview,
        default_result_mode=FunctionCallResultModeEnum.terminal,
    )
    return registry


def _normalized_query(query: str) -> str:
    return query.strip().lower()


def _match_current_datetime(query: str, _: ChatSessionContext) -> float:
    normalized = _normalized_query(query)
    return 1.0 if any(keyword in normalized for keyword in _TIME_KEYWORDS) else 0.0


async def _execute_current_datetime(
    session: ChatSessionContext,
    spec: FunctionToolSpec,
) -> FunctionToolExecutionResult:
    now = datetime.now().astimezone()
    text = f"当前时间：{now.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    return FunctionToolExecutionResult(
        tool_name=spec.tool_name,
        title=spec.title,
        summary=text,
        text=text,
        data={
            "iso": now.isoformat(),
            "timezone": str(now.tzinfo),
            "conversation_id": session.conversation_id,
        },
        prefer_terminal=True,
        terminal_payload=_text_payload(text),
    )


def _match_calculate_expression(query: str, _: ChatSessionContext) -> float:
    normalized = _normalized_query(query)
    if "计算" in normalized or "算一下" in normalized or "evaluate" in normalized:
        return 1.0 if _extract_math_expression(query) else 0.0
    return 0.85 if _extract_math_expression(query) else 0.0


async def _execute_calculate_expression(
    session: ChatSessionContext,
    spec: FunctionToolSpec,
) -> FunctionToolExecutionResult:
    expression = _extract_math_expression(session.query)
    if not expression:
        raise ValueError("未识别到可计算的表达式")
    result = _evaluate_math_expression(expression)
    text = f"计算结果：{expression} = {result}"
    return FunctionToolExecutionResult(
        tool_name=spec.tool_name,
        title=spec.title,
        summary=text,
        text=text,
        data={"expression": expression, "result": result},
        prefer_terminal=True,
        terminal_payload=_text_payload(text),
    )


def _match_session_context(query: str, _: ChatSessionContext) -> float:
    normalized = _normalized_query(query)
    return 1.0 if any(keyword in normalized for keyword in _SESSION_KEYWORDS) else 0.0


async def _execute_session_context(
    session: ChatSessionContext,
    spec: FunctionToolSpec,
) -> FunctionToolExecutionResult:
    payload = {
        "conversation_id": session.conversation.id,
        "conversation_title": session.conversation.title,
        "conversation_status": session.conversation.status,
        "request_id": session.turn_request.request_id,
        "resolved_capability_ids": [item.capability_id for item in session.resolved_capabilities],
        "prompt_placeholders": [item.value for item in session.prompt_state.selected_placeholders()],
        "intent": session.artifacts.intent_result and session.artifacts.intent_result.intent,
    }
    text = (
        f"当前会话：#{session.conversation.id} {session.conversation.title}\n"
        f"本轮请求：{session.turn_request.request_id or '未指定'}\n"
        f"已解析 capability 数量：{len(session.resolved_capabilities)}"
    )
    return FunctionToolExecutionResult(
        tool_name=spec.tool_name,
        title=spec.title,
        summary="输出当前会话上下文摘要",
        text=text,
        data=payload,
        prefer_terminal=True,
        terminal_payload=_text_payload(text),
    )


def _match_capability_overview(query: str, _: ChatSessionContext) -> float:
    normalized = _normalized_query(query)
    return 1.0 if any(keyword in normalized for keyword in _CAPABILITY_KEYWORDS) else 0.0


async def _execute_capability_overview(
    session: ChatSessionContext,
    spec: FunctionToolSpec,
) -> FunctionToolExecutionResult:
    items = [
        {
            "capability_id": item.capability_id,
            "name": item.name,
            "kind": item.kind.value,
            "source": item.source,
            "priority": item.priority,
        }
        for item in session.resolved_capabilities
    ]
    text = "当前启用能力：\n" + "\n".join(f"- {item['name']} ({item['kind']} / {item['source']})" for item in items)
    return FunctionToolExecutionResult(
        tool_name=spec.tool_name,
        title=spec.title,
        summary="输出当前 turn 的 capability 概览",
        text=text,
        data={"items": items},
        prefer_terminal=True,
        terminal_payload=_text_payload(text),
    )


def _text_payload(text: str) -> MessageBundlePayload:
    return MessageBundlePayload(
        role=ChatRoleEnum.assistant,
        blocks=[TextBlock(text=text)],
    )


def _extract_math_expression(query: str) -> str | None:
    matches = [match.group("expr").strip() for match in _MATH_PATTERN.finditer(query)]
    for expression in sorted(matches, key=len, reverse=True):
        compact = expression.replace(" ", "")
        if any(operator in compact for operator in "+-*/%") and sum(char.isdigit() for char in compact) >= 2:
            return expression
    return None


def _evaluate_math_expression(expression: str) -> float | int:
    node = ast.parse(expression, mode="eval")
    return _evaluate_ast(node.body)


def _evaluate_ast(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp):
        left = _evaluate_ast(node.left)
        right = _evaluate_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Mod):
            return left % right
        raise ValueError("表达式包含不支持的运算符")
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_ast(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError("表达式包含不支持的单目运算")
    raise ValueError("表达式包含不支持的语法")
