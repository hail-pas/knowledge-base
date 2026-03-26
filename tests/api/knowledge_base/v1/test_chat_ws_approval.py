from fastapi.testclient import TestClient

from api.knowledge_base.factory import knowledge_api


def test_chat_guarded_capability_executes_without_approval_flow():
    with TestClient(knowledge_api).websocket_connect("/v1/chat/ws?token=token") as websocket:
        ready_payload = websocket.receive_json()
        assert ready_payload["event"] == "session.ready"

        websocket.send_json(
            {
                "command": "turn.start",
                "payload": {
                    "conversation_id": None,
                    "input": {
                        "role": "user",
                        "blocks": [{"type": "text", "text": "请创建工单处理 VPN 权限问题"}],
                    },
                    "resource_selection": {
                        "capabilities": [
                            {
                                "capability_key": "guarded_work_order_create",
                                "kind": "skill",
                            },
                        ],
                    },
                },
            },
        )

        events = []
        while True:
            item = websocket.receive_json()
            events.append(item)
            if item["event"] in {"turn.completed", "turn.failed"}:
                break

    assert not any(item["event"].startswith("approval.") for item in events)
    assert any(
        item["event"] == "step.completed"
        and item["payload"]["data"]["payload_type"] == "tool_result"
        and "已模拟创建工单" in (item["payload"]["data"]["payload"]["content_text"] or "")
        for item in events
    )
