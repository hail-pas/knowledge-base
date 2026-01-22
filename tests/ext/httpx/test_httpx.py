"""
httpx 扩展测试

测试全局 httpx 客户端的基本功能。
"""

import pytest
import httpx

from ext.ext_httpx.main import HttpxConfig
from core.context import ctx


@pytest.fixture
async def httpx_config():
    """初始化 httpx 配置"""
    config = HttpxConfig()
    async with ctx():
        await config.register()
        yield config
        await config.unregister()


@pytest.mark.asyncio
async def test_httpx_get(httpx_config):
    """测试 httpx_get 方法"""
    response = await httpx_config.instance.get("https://httpbin.org/get")

    assert response.status_code == 200
    data = response.json()
    assert "url" in data


@pytest.mark.asyncio
async def test_httpx_post(httpx_config):
    """测试 httpx_post 方法"""
    payload = {"key": "value"}
    response = await httpx_config.instance.post("https://httpbin.org/post", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["json"] == payload


@pytest.mark.asyncio
async def test_httpx_put(httpx_config):
    """测试 httpx_put 方法"""
    payload = {"key": "updated_value"}
    response = await httpx_config.instance.put("https://httpbin.org/put", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["json"] == payload


@pytest.mark.asyncio
async def test_httpx_patch(httpx_config):
    """测试 httpx_patch 方法"""
    payload = {"key": "patched_value"}
    response = await httpx_config.instance.patch("https://httpbin.org/patch", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["json"] == payload


@pytest.mark.asyncio
async def test_httpx_delete(httpx_config):
    """测试 httpx_delete 方法"""
    response = await httpx_config.instance.delete("https://httpbin.org/delete")

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_httpx_request(httpx_config):
    """测试 httpx_request 方法"""
    response = await httpx_config.instance.request("GET", "https://httpbin.org/get")

    assert response.status_code == 200
    data = response.json()
    assert "url" in data


@pytest.mark.asyncio
async def test_httpx_custom_timeout(httpx_config):
    """测试自定义超时参数"""
    response = await httpx_config.instance.get("https://httpbin.org/delay/1", timeout=5.0)

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_httpx_custom_headers(httpx_config):
    """测试自定义 headers"""
    custom_header = {"X-Custom-Header": "test-value"}
    response = await httpx_config.instance.get("https://httpbin.org/headers", headers=custom_header)

    assert response.status_code == 200
    data = response.json()
    assert data["headers"]["X-Custom-Header"] == "test-value"


@pytest.mark.asyncio
async def test_httpx_with_params(httpx_config):
    """测试 URL 参数"""
    params = {"param1": "value1", "param2": "value2"}
    response = await httpx_config.instance.get("https://httpbin.org/get", params=params)

    assert response.status_code == 200
    data = response.json()
    assert data["args"] == params


@pytest.mark.asyncio
async def test_httpx_error_handling(httpx_config):
    """测试错误处理"""
    with pytest.raises(httpx.HTTPStatusError):
        response = await httpx_config.instance.get("https://httpbin.org/status/404")
        response.raise_for_status()


@pytest.mark.asyncio
async def test_httpx_multiple_requests(httpx_config):
    """测试多个请求共享同一个客户端"""
    response1 = await httpx_config.instance.get("https://httpbin.org/get")
    response2 = await httpx_config.instance.get("https://httpbin.org/uuid")

    assert response1.status_code == 200
    assert response2.status_code == 200
    assert response2.json()["uuid"] is not None
