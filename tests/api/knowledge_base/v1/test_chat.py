import pytest

from fastapi.testclient import TestClient
from uuid import uuid4

from api.knowledge_base.factory import knowledge_api
from ext.ext_tortoise.models.user_center import Account, Role
from service.chat.domain.schema import ChatErrorCodeEnum


def _create_conversation(*, token: str, title: str) -> int:
    events = []
    with TestClient(knowledge_api).websocket_connect(f"/v1/chat/ws?token={token}") as websocket:
        websocket.send_json(
            {
                "command": "turn.start",
                "payload": {
                    "conversation_id": None,
                    "conversation_title": title,
                    "request_id": f"seed-{uuid4().hex[:8]}",
                    "input": {
                        "role": "user",
                        "blocks": [{"type": "text", "text": "请计算 1 + 1"}],
                    },
                    "resource_selection": {
                        "actions": [
                            {
                                "action_id": "inline:function",
                                "kind": "function_call",
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
        while True:
            item = websocket.receive_json()
            events.append(item)
            if item["event"] in {"turn.completed", "turn.failed", "turn.canceled"}:
                break

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


def test_update_conversation_default_resource_selection(client):
    conversation_id = _create_conversation(token="token", title="conversation-defaults")

    update_response = client.put(
        f"/v1/chat/conversation/{conversation_id}/resource-selection",
        json={
            "actions": [
                {
                    "action_id": "inline:function",
                    "kind": "function_call",
                    "priority": 10,
                    "config": {"tools": [{"tool_name": "calculate_expression"}]},
                },
                {
                    "action_id": "inline:tool",
                    "kind": "tool_call",
                    "priority": 30,
                    "config": {"policy": "optional", "tool_names": ["lookup"]},
                },
            ],
        },
    )
    assert update_response.status_code == 200
    update_payload = update_response.json()
    assert update_payload["code"] == 0
    assert "capability_profile_ids" not in update_payload["data"]["default_resource_selection"]
    assert "capability_binding_ids" not in update_payload["data"]["default_resource_selection"]

    detail_response = client.get(f"/v1/chat/conversation/{conversation_id}")
    assert detail_response.status_code == 200
    detail_payload = detail_response.json()
    assert detail_payload["code"] == 0
    selection = detail_payload["data"]["default_resource_selection"]
    assert "capability_profile_ids" not in selection
    assert "capability_binding_ids" not in selection
    assert [item["action_id"] for item in selection["actions"]] == [
        "inline:function",
        "inline:tool",
        "builtin:llm_response",
    ]


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


def test_chat_websocket_session_bind_removed():
    with TestClient(knowledge_api).websocket_connect("/v1/chat/ws?token=token") as websocket:
        websocket.send_json({"command": "session.bind", "payload": {}})
        payload = websocket.receive_json()

    assert payload["event"] == "error"
    assert payload["payload"]["code"] == ChatErrorCodeEnum.unknown_command.value


def test_chat_demo_page_uses_relative_chat_urls(client):
    response = client.get("/v1/chat/demo")

    assert response.status_code == 200
    assert 'const chatRoot = new URL(".", window.location.href);' in response.text
    assert 'const chatBase = "/v1/chat";' not in response.text


def test_chat_websocket_rejects_foreign_conversation(monkeypatch: pytest.MonkeyPatch, chat_access_accounts):
    owner, viewer, staff = chat_access_accounts
    _install_token_auth(monkeypatch, owner, viewer, staff)

    conversation_id = _create_conversation(token="owner", title="ws-owner-conversation")

    with TestClient(knowledge_api).websocket_connect("/v1/chat/ws?token=viewer") as websocket:
        websocket.send_json(
            {
                "command": "turn.start",
                "payload": {
                    "conversation_id": conversation_id,
                    "request_id": "viewer-forbidden",
                    "input": {
                        "role": "user",
                        "blocks": [{"type": "text", "text": "不应该成功"}],
                    },
                },
            },
        )
        payload = websocket.receive_json()

    assert payload["event"] == "error"
    assert payload["payload"]["message"] == "会话不存在"


def test_chat_first_turn_creates_conversation_and_timeline(
    monkeypatch: pytest.MonkeyPatch,
    chat_access_accounts,
):
    owner, viewer, staff = chat_access_accounts
    _install_token_auth(monkeypatch, owner, viewer, staff)

    events = []
    with TestClient(knowledge_api).websocket_connect("/v1/chat/ws?token=owner") as websocket:
        websocket.send_json(
            {
                "command": "turn.start",
                "payload": {
                    "conversation_id": None,
                    "request_id": "req-auto-create",
                    "input": {
                        "role": "user",
                        "blocks": [{"type": "text", "text": "请计算 12 + 7 * 2"}],
                    },
                    "resource_selection": {
                        "actions": [
                            {
                                "action_id": "inline:function",
                                "kind": "function_call",
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

        while True:
            item = websocket.receive_json()
            events.append(item)
            if item["event"] in {"turn.completed", "turn.failed", "turn.canceled"}:
                break

    assert any(item["event"] == "ack" for item in events)
    conversation_ids = {item["conversation_id"] for item in events if item.get("conversation_id")}
    assert len(conversation_ids) == 1
    conversation_id = conversation_ids.pop()

    owner_client = _client("owner")
    conversation_list = owner_client.get("/v1/chat/conversation").json()["data"]
    matched = next(item for item in conversation_list if item["conversation"]["id"] == conversation_id)
    assert matched["latest_user_text"] == "请计算 12 + 7 * 2"
    assert "26" in (matched["latest_assistant_text"] or "")

    timeline = owner_client.get(f"/v1/chat/conversation/{conversation_id}/timeline").json()["data"]
    assert timeline["conversation"]["id"] == conversation_id
    assert timeline["conversation"]["title"] == "请计算 12 + 7 * 2"
    assert len(timeline["turns"]) == 1
    turn = timeline["turns"][0]
    assert turn["input"]["blocks"][0]["text"] == "请计算 12 + 7 * 2"
    assert "26" in turn["output"]["blocks"][0]["text"]
    assert any(item["event"] == "step.created" for item in turn["events"])
    assert any(item["event"] == "message.completed" for item in turn["events"])
