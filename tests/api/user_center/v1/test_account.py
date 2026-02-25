import pytest
from fastapi.testclient import TestClient

CREATED_ID = None

def test_create_account(client: TestClient):
    """测试创建账号接口"""
    response = client.post(
        "/v1/account",
        json={
            "username": "testuser",
            "phone": "13800138000",
            "email": "test@example.com",
            "password": "test123456",
            "role_id": 1,
        },
    )
    assert response.status_code == 200
    assert response.json()["data"]["id"] is not None
    global CREATED_ID
    CREATED_ID = response.json()["data"]["id"]


def test_get_account_list(client: TestClient):
    """测试获取账号列表接口"""
    response = client.get("/v1/account")
    global CREATED_ID

    assert response.status_code == 200
    assert response.json()["data"]["items"][0]["id"] == CREATED_ID


def test_get_account_detail(client: TestClient):
    """测试获取账号详情接口"""
    global CREATED_ID
    if not CREATED_ID:
        pytest.skip("未创建账号")
    response = client.get(f"/v1/account/{CREATED_ID}")
    assert response.status_code == 200


def test_update_account(client: TestClient):
    """测试更新账号接口"""
    global CREATED_ID
    if not CREATED_ID:
        pytest.skip("未创建账号")
    response = client.put(
        f"/v1/account/{CREATED_ID}",
        json={
            "username": "updateduser",
            "email": "updated@example.com",
        },
    )
    assert response.status_code == 200


def test_delete_account(client: TestClient):
    """测试删除账号接口"""
    global CREATED_ID
    if not CREATED_ID:
        pytest.skip("未创建账号")
    response = client.delete(f"/v1/account/{CREATED_ID}")
    assert response.status_code == 200
