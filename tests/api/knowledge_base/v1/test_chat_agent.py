def test_chat_agent_and_mount_crud(client):
    list_response = client.get("/v1/chat/agent")
    assert list_response.status_code == 200
    builtin_agents = list_response.json()["data"]
    assert any(item["agent_key"] == "orchestrator.default" for item in builtin_agents)

    orchestrator_response = client.post(
        "/v1/chat/agent",
        json={
            "manifest": {
                "agent_key": "custom.orchestrator",
                "name": "自定义编排代理",
                "role": "orchestrator",
                "description": "测试用编排代理",
                "system_prompt": "你是测试编排代理。",
            },
        },
    )
    assert orchestrator_response.status_code == 200
    orchestrator = orchestrator_response.json()["data"]

    specialist_response = client.post(
        "/v1/chat/agent",
        json={
            "manifest": {
                "agent_key": "custom.specialist",
                "name": "自定义专家代理",
                "role": "specialist",
                "description": "测试用专家代理",
                "system_prompt": "你是测试专家代理。",
                "default_resource_selection": {
                    "actions": [
                        {
                            "kind": "tool_call",
                            "priority": 10,
                            "config": {"tools": [{"tool_name": "session_context"}]},
                        },
                    ],
                },
            },
        },
    )
    assert specialist_response.status_code == 200
    specialist = specialist_response.json()["data"]

    mount_response = client.post(
        "/v1/chat/agent/mount",
        json={
            "source_agent_id": orchestrator["id"],
            "mounted_agent_id": specialist["id"],
            "mode": "delegate",
            "purpose": "以 capability 暴露专家能力",
            "trigger_tags": ["上下文"],
            "mounted_as_capability": "custom.specialist.delegate",
            "output_contract": "terminal_text",
        },
    )
    assert mount_response.status_code == 200
    mount = mount_response.json()["data"]
    assert mount["mounted_as_capability"] == "custom.specialist.delegate"

    mounts_response = client.get("/v1/chat/agent/mount/list", params={"source_agent_id": orchestrator["id"]})
    assert mounts_response.status_code == 200
    mounts = mounts_response.json()["data"]
    assert any(item["id"] == mount["id"] for item in mounts)
