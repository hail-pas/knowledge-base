from fastapi.testclient import TestClient


CREATED_ID = None


def test_create_role(client: TestClient):
    """测试创建角色接口"""
    response = client.post(
        "/v1/role",
        json={
            "label": "测试角色",
            "remark": "这是一个测试角色",
            "resources": [1,],
        },
    )
    assert response.status_code == 200


def test_get_role_list(client: TestClient):
    """测试获取角色列表接口"""
    response = client.get("/v1/role")
    created_id = response.json()["data"]["items"][0]["id"]
    global CREATED_ID
    CREATED_ID = created_id
    assert response.status_code == 200


def test_get_role_detail(client: TestClient):
    """测试获取角色详情接口"""
    global CREATED_ID
    if not CREATED_ID:
        test_create_role(client)
    response = client.get(f"/v1/role/{CREATED_ID}")
    assert response.status_code == 200


def test_update_role(client: TestClient):
    """测试更新角色接口"""
    global CREATED_ID
    if not CREATED_ID:
        test_create_role(client)
    response = client.put(
        f"/v1/role/{CREATED_ID}",
        json={
            "label": "更新后的角色",
            "remark": "更新后的描述",
            "resources": [1],
        },
    )
    assert response.status_code == 200


def test_delete_role(client: TestClient):
    """测试删除角色接口"""
    global CREATED_ID
    if not CREATED_ID:
        test_create_role(client)
    response = client.delete(f"/v1/role/{CREATED_ID}")
    assert response.status_code == 200
