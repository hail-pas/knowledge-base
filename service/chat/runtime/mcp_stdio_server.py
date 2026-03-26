from __future__ import annotations

import json
import argparse
from typing import Any

from mcp.server.fastmcp import FastMCP


def build_platform_server() -> FastMCP:
    server = FastMCP("chat-platform")

    @server.tool(
        name="session_state",
        title="会话状态",
        description="返回当前 chat 会话与 agent 状态。",
        structured_output=True,
    )
    async def session_state(
        conversation_id: int,
        conversation_title: str,
        agent_key: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        text = (
            f"MCP会话状态：conversation={conversation_id}，"
            f"agent={agent_key or 'n/a'}，"
            f"request={request_id or 'n/a'}"
        )
        return {
            "title": "会话状态",
            "disposition": "terminal",
            "summary": "返回当前会话状态",
            "text": text,
            "data": {
                "conversation_id": conversation_id,
                "conversation_title": conversation_title,
                "agent_key": agent_key,
                "request_id": request_id,
            },
        }

    @server.tool(
        name="capability_overview",
        title="能力总览",
        description="返回当前 turn 的 capability 与 action 总览。",
        structured_output=True,
    )
    async def capability_overview(items_json: str) -> dict[str, Any]:
        items = json.loads(items_json)
        return {
            "title": "能力总览",
            "disposition": "context",
            "summary": "返回当前 turn 的能力与动作总览",
            "text": f"当前能力动作数量：{len(items)}",
            "data": {"items": items},
        }

    return server


def build_knowledge_base_server() -> FastMCP:
    server = FastMCP("chat-knowledge-base")

    @server.tool(
        name="collection_catalog",
        title="知识库集合目录",
        description="列出可访问集合。",
        structured_output=True,
    )
    async def collection_catalog(items_json: str) -> dict[str, Any]:
        items = json.loads(items_json)
        return {
            "title": "知识库集合目录",
            "disposition": "context",
            "summary": "返回知识库集合目录",
            "text": "已返回集合目录",
            "data": {"items": items},
        }

    return server


def build_server(server_name: str) -> FastMCP:
    if server_name == "platform":
        return build_platform_server()
    if server_name == "knowledge_base":
        return build_knowledge_base_server()
    raise ValueError(f"Unsupported MCP stdio server: {server_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local stdio MCP server for chat runtime")
    parser.add_argument("server_name")
    args = parser.parse_args()
    build_server(args.server_name).run(transport="stdio")


if __name__ == "__main__":
    main()
