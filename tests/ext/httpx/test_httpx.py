"""
httpx 扩展测试

测试全局 httpx 客户端的基本功能。
"""

import pytest
import httpx
from config.main import local_configs


@pytest.mark.asyncio
async def test_httpx_get():
    """测试 httpx_get 方法"""
    response = await local_configs.extensions.httpx.instance.get("https://httpbin.org/get")

    assert response.status_code == 200
    data = response.json()
    assert "url" in data


@pytest.mark.asyncio
async def test_httpx_post():
    """测试 httpx_post 方法"""
    payload = {"key": "value"}
    response = await local_configs.extensions.httpx.instance.post("https://httpbin.org/post", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["json"] == payload


@pytest.mark.asyncio
async def test_httpx_put():
    """测试 httpx_put 方法"""
    payload = {"key": "updated_value"}
    response = await local_configs.extensions.httpx.instance.put("https://httpbin.org/put", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["json"] == payload


@pytest.mark.asyncio
async def test_httpx_patch():
    """测试 httpx_patch 方法"""
    payload = {"key": "patched_value"}
    response = await local_configs.extensions.httpx.instance.patch("https://httpbin.org/patch", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["json"] == payload


@pytest.mark.asyncio
async def test_httpx_delete():
    """测试 httpx_delete 方法"""
    response = await local_configs.extensions.httpx.instance.delete("https://httpbin.org/delete")

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_httpx_request():
    """测试 httpx_request 方法"""
    response = await local_configs.extensions.httpx.instance.request("GET", "https://httpbin.org/get")

    assert response.status_code == 200
    data = response.json()
    assert "url" in data


@pytest.mark.asyncio
async def test_httpx_custom_timeout():
    """测试自定义超时参数"""
    response = await local_configs.extensions.httpx.instance.get("https://httpbin.org/delay/1", timeout=5.0)

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_httpx_custom_headers():
    """测试自定义 headers"""
    custom_header = {"X-Custom-Header": "test-value"}
    response = await local_configs.extensions.httpx.instance.get("https://httpbin.org/headers", headers=custom_header)

    assert response.status_code == 200
    data = response.json()
    assert data["headers"]["X-Custom-Header"] == "test-value"


@pytest.mark.asyncio
async def test_httpx_with_params():
    """测试 URL 参数"""
    params = {"param1": "value1", "param2": "value2"}
    response = await local_configs.extensions.httpx.instance.get("https://httpbin.org/get", params=params)

    assert response.status_code == 200
    data = response.json()
    assert data["args"] == params


@pytest.mark.asyncio
async def test_httpx_error_handling():
    """测试错误处理"""
    with pytest.raises(httpx.HTTPStatusError):
        response = await local_configs.extensions.httpx.instance.get("https://httpbin.org/status/404")
        response.raise_for_status()


@pytest.mark.asyncio
async def test_httpx_multiple_requests():
    """测试多个请求共享同一个客户端"""
    response1 = await local_configs.extensions.httpx.instance.get("https://httpbin.org/get")
    response2 = await local_configs.extensions.httpx.instance.get("https://httpbin.org/uuid")

    assert response1.status_code == 200
    assert response2.status_code == 200
    assert response2.json()["uuid"] is not None
