from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from api.knowledge_base.factory import knowledge_api
from ext.ext_tortoise.models.user_center import Account, Role
from service.chat import chat_app_service
from service.chat.domain.schema import ChatErrorCodeEnum


def _receive_next_non_ready(websocket):
    while True:
        payload = websocket.receive_json()
        if payload["event"] != "session.ready":
            return payload


def _read_until_terminal(websocket):
    events = []
    while True:
        item = websocket.receive_json()
        events.append(item)
        if item["event"] in {"turn.completed", "turn.failed", "turn.canceled"}:
            return events


def _send_turn(websocket, payload: dict) -> list[dict]:
    websocket.send_json({"command": "turn.start", "payload": payload})
    return _read_until_terminal(websocket)


def _create_conversation(*, token: str, title: str) -> int:
    events = []
    with TestClient(knowledge_api).websocket_connect(f"/v1/chat/ws?token={token}") as websocket:
        ready_payload = websocket.receive_json()
        assert ready_payload["event"] == "session.ready"
        websocket.send_json(
            {
                "command": "turn.start",
                "payload": {
                    "conversation_id": None,
                    "conversation_title": title,
                    "request_id": str(uuid4()),
                    "input": {
                        "role": "user",
                        "blocks": [{"type": "text", "text": "请计算 1 + 1"}],
                    },
                    "resource_selection": {
                        "actions": [
                            {
                                "action_id": "inline:tool",
                                "kind": "tool_call",
                                "priority": 10,
                                "config": {
                                    "tools": [{"tool_name": "calculate_expression"}],
                                    "stop_after_terminal": True,
                                },
                            },
                        ],
                    },
                },
            },
        )
        events.extend(_read_until_terminal(websocket))

    conversation_ids = {item["conversation_id"] for item in events if item.get("conversation_id")}
    assert len(conversation_ids) == 1
    return conversation_ids.pop()


def _client(token: str) -> TestClient:
    return TestClient(knowledge_api, headers={"Authorization": f"Bearer {token}"})


def _install_token_auth(
    monkeypatch: pytest.MonkeyPatch,
    owner: Account,
    viewer: Account,
    staff: Account,
) -> None:
    token_map = {"owner": owner, "viewer": viewer, "staff": staff}

    async def mock_validate(request, token):
        account = token_map[token.credentials]
        request.scope["user"] = account
        request.scope["scene"] = "test"
        request.scope["is_staff"] = account.is_staff
        request.scope["is_super_admin"] = account.is_super_admin
        return account

    async def allow_permissions(self, apis, conn_name=None):
        return True

    monkeypatch.setattr("service.depend._validate_jwt_token", mock_validate)
    monkeypatch.setattr(Account, "has_permission", allow_permissions)


def _assert_api_error(response, message: str) -> None:
    assert response.status_code == 200
    payload = response.json()
    assert payload["code"] != 0
    assert payload["message"] == message


@pytest.fixture
async def chat_access_accounts():
    suffix = uuid4().hex[:8]
    digits = str(int(uuid4().hex, 16))[-7:]
    role = await Role.create(label=f"chat-access-role-{uuid4().hex[:8]}", remark="chat access role")
    owner = await Account.create(
        username=f"chatowner{suffix}",
        phone=f"1777{digits}",
        email=f"chat_owner_{suffix}@example.com",
        password="hashed",
        is_active=True,
        is_staff=False,
        is_super_admin=False,
        role=role,
    )
    viewer = await Account.create(
        username=f"chatviewer{suffix}",
        phone=f"1666{digits}",
        email=f"chat_viewer_{suffix}@example.com",
        password="hashed",
        is_active=True,
        is_staff=False,
        is_super_admin=False,
        role=role,
    )
    staff = await Account.create(
        username=f"chatstaff{suffix}",
        phone=f"1555{digits}",
        email=f"chat_staff_{suffix}@example.com",
        password="hashed",
        is_active=True,
        is_staff=True,
        is_super_admin=False,
        role=role,
    )
    return owner, viewer, staff


def test_chat_create_conversation_endpoint_removed(client):
    response = client.post(
        "/v1/chat/conversation",
        json={"title": "legacy", "resource_selection": {}},
    )

    assert response.status_code == 405


def test_chat_turn_event_replay_endpoint_removed(client):
    response = client.get("/v1/chat/turn/1/events")

    assert response.status_code == 404


def test_update_conversation_default_resource_selection(client):
    conversation_id = _create_conversation(token="token", title="conversation-defaults")

    update_response = client.put(
        f"/v1/chat/conversation/{conversation_id}/resource-selection",
        json={
            "actions": [
                {
                    "action_id": "inline:tool",
                    "kind": "tool_call",
                    "priority": 10,
                    "config": {"tools": [{"tool_name": "calculate_expression"}]},
                },
                {
                    "action_id": "inline:lookup",
                    "kind": "tool_call",
                    "priority": 30,
                    "config": {"tools": [{"tool_name": "lookup"}]},
                },
            ],
        },
    )
    assert update_response.status_code == 200
    update_payload = update_response.json()
    assert update_payload["code"] == 0
    assert [item["action_id"] for item in update_payload["data"]["default_resource_selection"]["actions"]] == [
        "inline:tool",
        "inline:lookup",
        "builtin:llm_response",
    ]


def test_first_turn_resource_selection_is_not_persisted_as_conversation_default(client):
    conversation_id = _create_conversation(token="token", title="turn-only-selection")

    response = client.get(f"/v1/chat/conversation/{conversation_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["code"] == 0
    assert payload["data"]["default_resource_selection"]["actions"] == []
    assert payload["data"]["default_resource_selection"]["capabilities"] == []


def test_chat_conversation_scope_respects_owner(monkeypatch: pytest.MonkeyPatch, chat_access_accounts):
    owner, viewer, staff = chat_access_accounts
    _install_token_auth(monkeypatch, owner, viewer, staff)

    viewer_client = _client("viewer")
    conversation_id = _create_conversation(token="owner", title="owner-conversation")

    _assert_api_error(viewer_client.get(f"/v1/chat/conversation/{conversation_id}"), "会话不存在")
    _assert_api_error(
        viewer_client.put(
            f"/v1/chat/conversation/{conversation_id}/resource-selection",
            json={"actions": []},
        ),
        "会话不存在",
    )
    viewer_conversations = viewer_client.get("/v1/chat/conversation").json()["data"]
    assert all(item["conversation"]["id"] != conversation_id for item in viewer_conversations)
    _assert_api_error(
        viewer_client.get(f"/v1/chat/conversation/{conversation_id}/timeline"),
        "会话不存在",
    )


def test_chat_websocket_reports_auth_error_when_token_missing():
    client = TestClient(knowledge_api)

    with client.websocket_connect("/v1/chat/ws") as websocket:
        payload = websocket.receive_json()

    assert payload["event"] == "error"
    assert payload["payload"]["code"] == ChatErrorCodeEnum.auth_failed.value
    assert payload["payload"]["message"] == "授权头部缺失"


def test_chat_websocket_unknown_command_returns_error():
    with TestClient(knowledge_api).websocket_connect("/v1/chat/ws?token=token") as websocket:
        ready_payload = websocket.receive_json()
        assert ready_payload["event"] == "session.ready"
        websocket.send_json({"command": "session.bind", "payload": {}})
        payload = _receive_next_non_ready(websocket)

    assert payload["event"] == "error"
    assert payload["payload"]["code"] == ChatErrorCodeEnum.invalid_command_payload.value


def test_chat_websocket_runtime_command_error_is_not_reported_as_invalid_payload(
    monkeypatch: pytest.MonkeyPatch,
):
    async def explode_prepare_turn(*args, **kwargs):
        raise RuntimeError("prepare turn exploded")

    monkeypatch.setattr(chat_app_service, "prepare_turn", explode_prepare_turn)

    with TestClient(knowledge_api).websocket_connect("/v1/chat/ws?token=token") as websocket:
        ready_payload = websocket.receive_json()
        assert ready_payload["event"] == "session.ready"
        websocket.send_json(
            {
                "command": "turn.start",
                "payload": {
                    "conversation_id": None,
                    "request_id": str(uuid4()),
                    "input": {
                        "role": "user",
                        "blocks": [{"type": "text", "text": "hello"}],
                    },
                },
            },
        )
        payload = _receive_next_non_ready(websocket)

    assert payload["event"] == "error"
    assert payload["payload"]["code"] == ChatErrorCodeEnum.command_error.value
    assert payload["payload"]["message"] == "prepare turn exploded"


def test_chat_websocket_ack_precedes_turn_execution_events():
    with TestClient(knowledge_api).websocket_connect("/v1/chat/ws?token=token") as websocket:
        ready_payload = websocket.receive_json()
        assert ready_payload["event"] == "session.ready"
        websocket.send_json(
            {
                "command": "turn.start",
                "payload": {
                    "conversation_id": None,
                    "conversation_title": "ack-first",
                    "request_id": str(uuid4()),
                    "input": {
                        "role": "user",
                        "blocks": [{"type": "text", "text": "请计算 2 + 3"}],
                    },
                    "resource_selection": {
                        "actions": [
                            {
                                "kind": "tool_call",
                                "priority": 10,
                                "config": {
                                    "tools": [{"tool_name": "calculate_expression"}],
                                    "stop_after_terminal": True,
                                },
                            },
                        ],
                    },
                },
            },
        )
        first_payload = _receive_next_non_ready(websocket)
        remaining_events = _read_until_terminal(websocket)

    assert first_payload["event"] == "ack"
    assert any(item["event"] == "turn.started" for item in remaining_events)


def test_chat_websocket_reuses_current_conversation_and_allows_explicit_switch(client):
    with TestClient(knowledge_api).websocket_connect("/v1/chat/ws?token=token") as websocket:
        ready_payload = websocket.receive_json()
        assert ready_payload["event"] == "session.ready"

        first_events = _send_turn(
            websocket,
            {
                "conversation_id": None,
                "conversation_title": "switch-a",
                "request_id": str(uuid4()),
                "input": {
                    "role": "user",
                    "blocks": [{"type": "text", "text": "请计算 1 + 2"}],
                },
                "resource_selection": {
                    "actions": [
                        {
                            "kind": "tool_call",
                            "priority": 10,
                            "config": {"tools": [{"tool_name": "calculate_expression"}]},
                        },
                    ],
                },
            },
        )
        conversation_a = next(item["conversation_id"] for item in first_events if item.get("conversation_id"))

        second_events = _send_turn(
            websocket,
            {
                "request_id": str(uuid4()),
                "input": {
                    "role": "user",
                    "blocks": [{"type": "text", "text": "请继续算 3 + 4"}],
                },
                "resource_selection": {
                    "actions": [
                        {
                            "kind": "tool_call",
                            "priority": 10,
                            "config": {"tools": [{"tool_name": "calculate_expression"}]},
                        },
                    ],
                },
            },
        )
        assert {item["conversation_id"] for item in second_events if item.get("conversation_id")} == {conversation_a}

        third_events = _send_turn(
            websocket,
            {
                "conversation_id": None,
                "conversation_title": "switch-b",
                "request_id": str(uuid4()),
                "input": {
                    "role": "user",
                    "blocks": [{"type": "text", "text": "请计算 5 + 6"}],
                },
                "resource_selection": {
                    "actions": [
                        {
                            "kind": "tool_call",
                            "priority": 10,
                            "config": {"tools": [{"tool_name": "calculate_expression"}]},
                        },
                    ],
                },
            },
        )
        conversation_b = next(item["conversation_id"] for item in third_events if item.get("conversation_id"))

        switched_events = _send_turn(
            websocket,
            {
                "conversation_id": conversation_a,
                "request_id": str(uuid4()),
                "input": {
                    "role": "user",
                    "blocks": [{"type": "text", "text": "切回原会话，计算 7 + 8"}],
                },
                "resource_selection": {
                    "actions": [
                        {
                            "kind": "tool_call",
                            "priority": 10,
                            "config": {"tools": [{"tool_name": "calculate_expression"}]},
                        },
                    ],
                },
            },
        )

    assert conversation_a != conversation_b
    assert {item["conversation_id"] for item in switched_events if item.get("conversation_id")} == {conversation_a}

    timeline_a = client.get(f"/v1/chat/conversation/{conversation_a}/timeline").json()["data"]
    timeline_b = client.get(f"/v1/chat/conversation/{conversation_b}/timeline").json()["data"]
    assert len(timeline_a["turns"]) == 3
    assert len(timeline_b["turns"]) == 1


def test_chat_debug_page_uses_relative_chat_urls(client):
    response = client.get("/v1/chat/debug")

    assert response.status_code == 200
    assert "/knowledge/v1/chat/ws" in response.text
    assert "localStorage.getItem('chat_debug_token')" in response.text


def test_chat_websocket_rejects_foreign_conversation(monkeypatch: pytest.MonkeyPatch, chat_access_accounts):
    owner, viewer, staff = chat_access_accounts
    _install_token_auth(monkeypatch, owner, viewer, staff)

    conversation_id = _create_conversation(token="owner", title="ws-owner-conversation")

    with TestClient(knowledge_api).websocket_connect("/v1/chat/ws?token=viewer") as websocket:
        ready_payload = websocket.receive_json()
        assert ready_payload["event"] == "session.ready"
        websocket.send_json(
            {
                "command": "turn.start",
                "payload": {
                    "conversation_id": conversation_id,
                    "request_id": str(uuid4()),
                    "input": {
                        "role": "user",
                        "blocks": [{"type": "text", "text": "不应该成功"}],
                    },
                },
            },
        )
        payload = _receive_next_non_ready(websocket)

    assert payload["event"] == "error"
    assert payload["payload"]["code"] == ChatErrorCodeEnum.command_error.value
    assert payload["payload"]["message"] == "会话不存在"


def test_chat_timeline_returns_steps_not_events(client):
    conversation_id = _create_conversation(token="token", title="timeline-conversation")

    response = client.get(f"/v1/chat/conversation/{conversation_id}/timeline")

    assert response.status_code == 200
    payload = response.json()
    assert payload["code"] == 0
    turns = payload["data"]["turns"]
    assert len(turns) == 1
    turn = turns[0]
    assert "events" not in turn
    assert [step["step"]["name"] for step in turn["steps"]] == ["user_message", "calculate_expression"]
