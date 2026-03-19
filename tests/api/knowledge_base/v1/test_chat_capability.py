def test_chat_capability_crud(client):
    create_response = client.post(
        "/v1/chat/capability",
        json={
            "manifest": {
                "kind": "extension",
                "capability_key": "policy_lookup",
                "name": "政策检索",
                "description": "检索政策知识库",
                "routing": {"keywords": ["政策", "报销"]},
                "actions": [
                    {
                        "kind": "knowledge_retrieval",
                        "config": {"collection_ids": [1], "top_k": 3},
                    },
                ],
            },
        },
    )

    assert create_response.status_code == 200
    created = create_response.json()["data"]
    assert created["kind"] == "extension"
    assert created["capability_key"] == "policy_lookup"

    list_response = client.get("/v1/chat/capability", params={"kind": "extension"})
    assert list_response.status_code == 200
    items = list_response.json()["data"]
    assert any(item["id"] == created["id"] for item in items)

    detail_response = client.get(f"/v1/chat/capability/{created['id']}")
    assert detail_response.status_code == 200
    assert detail_response.json()["data"]["manifest"]["actions"][0]["kind"] == "knowledge_retrieval"
