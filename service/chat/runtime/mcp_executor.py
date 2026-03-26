from __future__ import annotations

import json
import sys
import inspect
from pathlib import Path
from typing import Any, Callable, Awaitable
from dataclasses import field, dataclass

from pydantic import JsonValue, TypeAdapter, ValidationError
from pydantic_ai.mcp import MCPServer, MCPServerStdio

from ext.ext_tortoise.models.knowledge_base import Collection
from service.chat.domain.schema import (
    MessageBundlePayload,
    ActionResultDispositionEnum,
)
from service.chat.runtime.session import ChatSessionContext

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_STDIO_SERVER_MODULE = "service.chat.runtime.mcp_stdio_server"
_JSON_VALUE_ADAPTER = TypeAdapter(JsonValue)


@dataclass(slots=True)
class MCPExecutionResult:
    server_name: str
    tool_name: str
    title: str | None
    disposition: ActionResultDispositionEnum = ActionResultDispositionEnum.context
    summary: str | None = None
    text: str | None = None
    data: JsonValue | None = None
    terminal_payload: MessageBundlePayload | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


MCPToolArgBuilder = Callable[[ChatSessionContext], dict[str, JsonValue] | Awaitable[dict[str, JsonValue]]]


@dataclass(slots=True)
class MCPToolDefinition:
    server_name: str
    tool_name: str
    title: str
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    arg_builder: MCPToolArgBuilder | None = None


class MCPServerRegistry:
    def __init__(self) -> None:
        self._servers: dict[str, MCPServer] = {}
        self._tool_definitions: dict[tuple[str, str], MCPToolDefinition] = {}

    def register_server(self, server_name: str, server: MCPServer) -> None:
        self._servers[server_name] = server

    def register_stdio_tool(
        self,
        server_name: str,
        tool_name: str,
        *,
        title: str,
        description: str = "",
        metadata: dict[str, Any] | None = None,
        arg_builder: MCPToolArgBuilder | None = None,
    ) -> None:
        if server_name not in self._servers:
            self.register_server(server_name, create_local_stdio_mcp_server(server_name))
        self._tool_definitions[(server_name, tool_name)] = MCPToolDefinition(
            server_name=server_name,
            tool_name=tool_name,
            title=title,
            description=description,
            metadata=metadata or {},
            arg_builder=arg_builder,
        )

    def get_server(self, server_name: str) -> MCPServer | None:
        return self._servers.get(server_name)

    def get_tool(self, server_name: str, tool_name: str) -> MCPToolDefinition | None:
        return self._tool_definitions.get((server_name, tool_name))

    def list_tool_names(self, server_name: str) -> list[str]:
        return sorted(tool_name for name, tool_name in self._tool_definitions if name == server_name)

    async def execute_many(
        self,
        *,
        server_name: str,
        tool_names: list[str],
        session: ChatSessionContext,
    ) -> list[tuple[MCPToolDefinition, MCPExecutionResult]]:
        server = self.get_server(server_name)
        if server is None:
            raise ValueError(f"MCP server `{server_name}` 未注册")
        definitions: list[MCPToolDefinition] = []
        for tool_name in tool_names:
            definition = self.get_tool(server_name, tool_name)
            if definition is None:
                raise ValueError(f"MCP tool `{server_name}.{tool_name}` 未注册")
            definitions.append(definition)

        executions: list[tuple[MCPToolDefinition, MCPExecutionResult]] = []
        async with server:
            for definition in definitions:
                raw_result = await server.direct_call_tool(
                    definition.tool_name,
                    await self.build_tool_args(definition, session=session),
                    metadata={
                        "conversation_id": session.conversation_id,
                        "request_id": session.request_id,
                        "session_id": str(session.session_id),
                    },
                )
                executions.append(
                    (
                        definition,
                        self.normalize_result(
                            server_name=server_name,
                            tool_name=definition.tool_name,
                            title=definition.title,
                            raw_result=raw_result,
                        ),
                    ),
                )
        return executions

    async def build_tool_args(
        self,
        definition: MCPToolDefinition,
        *,
        session: ChatSessionContext,
    ) -> dict[str, JsonValue]:
        if definition.arg_builder is None:
            return {}
        args = definition.arg_builder(session)
        if inspect.isawaitable(args):
            args = await args
        return args

    def normalize_result(
        self,
        *,
        server_name: str,
        tool_name: str,
        title: str,
        raw_result: Any,
    ) -> MCPExecutionResult:
        if isinstance(raw_result, MCPExecutionResult):
            return raw_result
        if isinstance(raw_result, MessageBundlePayload):
            return MCPExecutionResult(
                server_name=server_name,
                tool_name=tool_name,
                title=title,
                disposition=ActionResultDispositionEnum.terminal,
                summary=raw_result.text,
                text=raw_result.text,
                terminal_payload=raw_result,
            )
        if isinstance(raw_result, str):
            return MCPExecutionResult(
                server_name=server_name,
                tool_name=tool_name,
                title=title,
                summary=raw_result,
                text=raw_result,
            )
        if isinstance(raw_result, dict):
            disposition = raw_result.get("disposition", ActionResultDispositionEnum.context)
            terminal_payload = raw_result.get("terminal_payload")
            return MCPExecutionResult(
                server_name=server_name,
                tool_name=tool_name,
                title=str(raw_result.get("title") or title),
                disposition=(
                    disposition
                    if isinstance(disposition, ActionResultDispositionEnum)
                    else ActionResultDispositionEnum(str(disposition))
                ),
                summary=raw_result.get("summary"),
                text=raw_result.get("text"),
                data=self._normalize_structured_data(
                    tool_name=f"{server_name}.{tool_name}",
                    raw_data=raw_result.get("data"),
                ),
                terminal_payload=(
                    terminal_payload
                    if isinstance(terminal_payload, MessageBundlePayload)
                    else (
                        MessageBundlePayload.model_validate(terminal_payload)
                        if isinstance(terminal_payload, dict)
                        else None
                    )
                ),
                metadata=dict(raw_result.get("metadata") or {}),
            )
        if isinstance(raw_result, list):
            return MCPExecutionResult(
                server_name=server_name,
                tool_name=tool_name,
                title=title,
                summary=f"返回 {len(raw_result)} 条 MCP 结果",
                data=self._normalize_structured_data(
                    tool_name=f"{server_name}.{tool_name}",
                    raw_data=raw_result,
                ),
            )
        raise TypeError(f"Unsupported MCP result for `{server_name}.{tool_name}`: {type(raw_result)!r}")

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
            raise TypeError(f"MCP tool `{tool_name}` returned non-JSON-serializable structured data") from exc


def create_local_stdio_mcp_server(server_name: str) -> MCPServerStdio:
    return MCPServerStdio(
        command=sys.executable,
        args=["-m", _STDIO_SERVER_MODULE, server_name],
        cwd=_PROJECT_ROOT,
        timeout=15,
        read_timeout=300,
        id=f"chat.stdio.mcp.{server_name}",
        cache_tools=True,
        cache_resources=False,
    )


def create_default_mcp_registry() -> MCPServerRegistry:
    registry = MCPServerRegistry()
    registry.register_stdio_tool(
        "platform",
        "session_state",
        title="会话状态",
        description="返回当前 chat 会话与 agent 状态。",
        arg_builder=_build_platform_session_state_args,
    )
    registry.register_stdio_tool(
        "platform",
        "capability_overview",
        title="能力总览",
        description="返回当前 turn 的 capability 与 action 总览。",
        arg_builder=_build_platform_capability_overview_args,
    )
    registry.register_stdio_tool(
        "knowledge_base",
        "collection_catalog",
        title="知识库集合目录",
        description="列出可访问集合。",
        arg_builder=_build_knowledge_base_collection_catalog_args,
    )
    return registry


def _build_platform_session_state_args(session: ChatSessionContext) -> dict[str, Any]:
    return {
        "conversation_id": session.conversation_id,
        "conversation_title": session.conversation_title,
        "agent_key": session.turn_request.agent_key or session.conversation.agent_key,
        "request_id": session.request_id,
    }


def _build_platform_capability_overview_args(session: ChatSessionContext) -> dict[str, Any]:
    return {
        "items_json": json.dumps(
            [
                {
                    "action_id": item.action_id,
                    "action_name": item.name,
                    "action_kind": item.kind.value,
                    "capability_key": item.metadata.get("capability_key"),
                    "capability_category": item.metadata.get("capability_category"),
                }
                for item in session.resolved_actions
            ],
            ensure_ascii=False,
        ),
        "selected_capabilities_json": json.dumps(
            [
                {
                    "capability_key": item.capability_key,
                    "name": item.name,
                    "runtime_kind": item.runtime_kind.value,
                }
                for item in session.selected_capabilities
            ],
            ensure_ascii=False,
        ),
        "planner_summary": session.planner_summary or "",
    }


async def _build_visible_collection_items(session: ChatSessionContext, *, limit: int = 20) -> list[dict[str, Any]]:
    collections = await Collection.filter(deleted_at=0).order_by("id").limit(limit)
    return [
        {
            "id": item.id,
            "name": item.name,
            "description": item.description,
            "is_public": item.is_public,
        }
        for item in collections
        if session.is_staff or item.is_public or item.user_id is None or str(item.user_id) == str(session.account_id)
    ]


async def _build_knowledge_base_collection_catalog_args(session: ChatSessionContext) -> dict[str, Any]:
    visible_items = await _build_visible_collection_items(session)
    return {"items_json": json.dumps(visible_items, ensure_ascii=False)}
