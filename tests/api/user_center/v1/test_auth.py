import pytest
from fastapi.testclient import TestClient


def test_login_with_password(client: TestClient):
    """测试密码登录接口"""
    response = client.post(
        "/v1/auth/login/pwd",
        json={
            "identifier": "test-admin",
            "password": "test-password",
            "scene": "General",
        },
    )
    assert response.status_code == 200


def test_login_with_code(client: TestClient):
    """测试验证码登录接口"""
    response = client.post(
        "/v1/auth/login/code",
        json={
            "identifier": "13800138000",
            "code": "123456",
            "scene": "General",
        },
    )
    assert response.status_code == 200


def test_logout(client: TestClient):
    """测试登出接口"""
    response = client.post(
        "/v1/auth/logout",
        params={"scene": "General"},
    )
    assert response.status_code == 200


def test_get_myself_info(client: TestClient):
    """测试获取个人信息接口"""
    response = client.get("/v1/auth/myself")
    assert response.status_code == 200
