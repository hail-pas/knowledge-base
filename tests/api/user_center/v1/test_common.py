import pytest
from fastapi.testclient import TestClient


def test_captcha_image(client: TestClient):
    """测试图片验证码接口"""
    response = client.get(
        "/v1/common/captcha/image",
        params={"scene": "login"},
    )
    assert response.status_code == 200
    assert "image" in response.headers.get("content-type", "")
    assert "x-unique-key" in response.headers


def test_captcha_code(client: TestClient):
    """测试发送验证码接口"""
    response = client.post(
        "/v1/common/captcha/code",
        json={
            "identifier": "13800138000",
            "scene": "login",
        },
    )
    assert response.status_code == 200
