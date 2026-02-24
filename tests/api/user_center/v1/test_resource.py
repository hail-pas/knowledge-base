import pytest
from fastapi.testclient import TestClient


def test_create_resource(client: TestClient):
    """测试创建系统资源接口"""
    response = client.post(
        "/v1/resource",
        json={
            "code": "test_resource",
            "label": "测试资源",
            "front_route": "/test",
            "resource_type": "menu",
            "sub_resource_type": "add_tab",
            "scene": "General",
        },
    )
    print(response.json())
    assert response.status_code == 200


def test_get_resource_trees(client: TestClient):
    """测试获取系统的全层级菜单接口"""
    response = client.get("/v1/resource/trees")
    assert response.status_code == 200
