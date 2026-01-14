"""EmbeddingModelConfig CRUD API 测试"""

import pytest


@pytest.mark.asyncio
class TestEmbeddingModelConfigCRUD:
    """EmbeddingModelConfig CRUD 测试"""

    async def test_create_embedding_config(self, client, embedding_config_data):
        """测试创建 EmbeddingModelConfig"""
        test_data = {**embedding_config_data, "name": "test_create_embedding"}
        response = await client.post("/v1/config/embedding", json=test_data)
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0

    async def test_list_embedding_configs(self, client, embedding_config_data):
        """测试获取 EmbeddingModelConfig 列表"""
        # 先创建一个配置（使用唯一名称）
        test_data = {**embedding_config_data, "name": "test_list_embeddings"}
        await client.post("/v1/config/embedding", json=test_data)

        # 获取列表
        response = await client.get("/v1/config/embedding")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert "data" in data
        assert "records" in data["data"]
        assert len(data["data"]["records"]) >= 1

        # 验证创建的记录在列表中
        items = data["data"]["records"]
        created_item = next((item for item in items if item["name"] == "test_list_embeddings"), None)
        assert created_item is not None
        assert created_item["type"] == "openai"

    async def test_get_embedding_config_detail(self, client, embedding_config_data):
        """测试获取 EmbeddingModelConfig 详情"""
        # 创建配置（使用唯一名称）
        test_data = {**embedding_config_data, "name": "test_get_detail"}
        await client.post("/v1/config/embedding", json=test_data)

        # 从列表中获取刚创建的记录
        response = await client.get("/v1/config/embedding")
        data = response.json()
        items = data["data"]["records"]
        created_item = next((item for item in items if item["name"] == "test_get_detail"), None)
        assert created_item is not None
        pk = created_item["id"]

        # 获取详情
        response = await client.get(f"/v1/config/embedding/{pk}")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert data["data"]["name"] == test_data["name"]
        assert data["data"]["type"] == test_data["type"]

    async def test_update_embedding_config(self, client, embedding_config_data):
        """测试更新 EmbeddingModelConfig"""
        # 创建配置（使用唯一名称）
        test_data = {**embedding_config_data, "name": "test_update"}
        await client.post("/v1/config/embedding", json=test_data)

        # 从列表中获取刚创建的记录
        response = await client.get("/v1/config/embedding")
        data = response.json()
        items = data["data"]["records"]
        created_item = next((item for item in items if item["name"] == "test_update"), None)
        assert created_item is not None
        pk = created_item["id"]

        # 更新配置
        update_data = {
            "description": "Updated description",
            "is_enabled": False,
        }
        response = await client.put(f"/v1/config/embedding/{pk}", json=update_data)
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0

        # 验证更新
        detail_response = await client.get(f"/v1/config/embedding/{pk}")
        detail_data = detail_response.json()
        assert detail_data["data"]["description"] == "Updated description"
        assert detail_data["data"]["is_enabled"] is False

    async def test_delete_embedding_config(self, client, embedding_config_data):
        """测试删除 EmbeddingModelConfig"""
        # 创建配置（使用唯一名称）
        test_data = {**embedding_config_data, "name": "test_delete"}
        await client.post("/v1/config/embedding", json=test_data)

        # 从列表中获取刚创建的记录
        response = await client.get("/v1/config/embedding")
        data = response.json()
        items = data["data"]["records"]
        created_item = next((item for item in items if item["name"] == "test_delete"), None)
        assert created_item is not None
        pk = created_item["id"]

        # 删除配置
        response = await client.delete(f"/v1/config/embedding/{pk}")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0

        # 验证删除（应该返回错误）
        detail_response = await client.get(f"/v1/config/embedding/{pk}")
        assert detail_response.status_code == 200
        assert detail_response.json()["code"] != 0

    async def test_filter_embedding_configs(self, client, embedding_config_data):
        """测试过滤 EmbeddingModelConfig 列表"""
        # 创建多个配置（使用唯一名称）
        await client.post("/v1/config/embedding", json={
            **embedding_config_data,
            "name": "test_filter_openai",
            "type": "openai",
        })
        await client.post("/v1/config/embedding", json={
            **embedding_config_data,
            "name": "test_filter_st",
            "type": "sentence_transformers",
        })

        # 过滤 type
        response = await client.get("/v1/config/embedding?type=openai")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        items = data["data"]["records"]
        assert all(item["type"] == "openai" for item in items)

        # 过滤 name
        response = await client.get("/v1/config/embedding?name__icontains=openai")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        items = data["data"]["records"]
        assert any("openai" in item["name"] for item in items)

    async def test_full_crud_flow(self, client, embedding_config_data):
        """测试完整的 CRUD 流程：创建 -> 列表 -> 详情 -> 更新 -> 删除"""
        # 1. 创建
        test_data = {**embedding_config_data, "name": "test_full_flow"}
        create_response = await client.post("/v1/config/embedding", json=test_data)
        assert create_response.status_code == 200
        assert create_response.json()["code"] == 0

        # 2. 列表并获取 id
        list_response = await client.get("/v1/config/embedding")
        assert list_response.status_code == 200
        list_data = list_response.json()
        assert list_data["code"] == 0
        items = list_data["data"]["records"]
        created_item = next((item for item in items if item["name"] == "test_full_flow"), None)
        assert created_item is not None
        pk = created_item["id"]

        # 3. 详情
        detail_response = await client.get(f"/v1/config/embedding/{pk}")
        assert detail_response.status_code == 200
        detail_data = detail_response.json()
        assert detail_data["code"] == 0
        assert detail_data["data"]["name"] == test_data["name"]

        # 4. 更新
        update_response = await client.put(f"/v1/config/embedding/{pk}", json={"description": "Full flow update"})
        assert update_response.status_code == 200
        assert update_response.json()["code"] == 0

        # 验证更新
        updated_detail = await client.get(f"/v1/config/embedding/{pk}")
        assert updated_detail.json()["data"]["description"] == "Full flow update"

        # 5. 删除
        delete_response = await client.delete(f"/v1/config/embedding/{pk}")
        assert delete_response.status_code == 200
        assert delete_response.json()["code"] == 0

        # 验证删除
        final_detail = await client.get(f"/v1/config/embedding/{pk}")
        assert final_detail.json()["code"] != 0
