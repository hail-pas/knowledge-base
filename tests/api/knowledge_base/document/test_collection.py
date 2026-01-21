"""Collection CRUD API 测试"""
import uuid
import pytest
from tests.api.knowledge_base.document.conftest import _generate_unique_id

@pytest.mark.asyncio
class TestCollectionCRUD:
    """Collection CRUD 测试"""

    async def test_create_collection(self, client):
        """测试创建 Collection"""
        unique_id = _generate_unique_id()
        collection_data = {
            "name": f"test_collection_{unique_id}",
            "description": f"Test collection description {unique_id}",
            "is_public": False,
            "workflow_template": {}, "extra_config": {}
        }
        response = await client.post("/v1/collection", json=collection_data)
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0

    async def test_list_collections(self, client):
        """测试获取 Collection 列表"""

        unique_id_1 = _generate_unique_id()
        unique_id_2 = _generate_unique_id()

        await client.post("/v1/collection", json={
            "name": f"test_list_collection_{unique_id_1}",
            "description": "Test 1",
            "is_public": False,
            "workflow_template": {}, "extra_config": {}
        })
        await client.post("/v1/collection", json={
            "name": f"test_list_collection_{unique_id_2}",
            "description": "Test 2",
            "is_public": False,
            "workflow_template": {}, "extra_config": {}
        })

        # 获取列表
        response = await client.get("/v1/collection")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert "data" in data
        assert "items" in data["data"]
        assert len(data["data"]["items"]) >= 2

    async def test_get_collection_detail(self, client):
        """测试获取 Collection 详情"""

        unique_id = _generate_unique_id()
        collection_data = {
            "name": f"test_detail_collection_{unique_id}",
            "description": f"Test description {unique_id}",
            "is_public": False,
            "workflow_template": {}, "extra_config": {}
        }

        # 创建集合
        response = await client.post("/v1/collection", json=collection_data)
        assert response.status_code == 200

        # 从列表中获取刚创建的记录
        list_response = await client.get("/v1/collection")
        list_data = list_response.json()
        items = list_data["data"]["items"]
        created_item = next((item for item in items if item["name"] == collection_data["name"]), None)
        assert created_item is not None
        pk = created_item["id"]

        # 获取详情
        response = await client.get(f"/v1/collection/{pk}")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert data["data"]["name"] == collection_data["name"]
        assert data["data"]["description"] == collection_data["description"]

    async def test_update_collection(self, client):
        """测试更新 Collection"""

        unique_id = _generate_unique_id()
        collection_data = {
            "name": f"test_update_collection_{unique_id}",
            "description": f"Original description {unique_id}",
            "is_public": False,
            "workflow_template": {}, "extra_config": {}
        }

        # 创建集合
        await client.post("/v1/collection", json=collection_data)

        # 从列表中获取刚创建的记录
        list_response = await client.get("/v1/collection")
        list_data = list_response.json()
        items = list_data["data"]["items"]
        created_item = next((item for item in items if item["name"] == collection_data["name"]), None)
        assert created_item is not None
        pk = created_item["id"]

        # 更新集合
        update_data = {
            "description": "Updated collection description",
            "is_public": True,
        }
        response = await client.put(f"/v1/collection/{pk}", json=update_data)
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0

        # 验证更新
        detail_response = await client.get(f"/v1/collection/{pk}")
        detail_data = detail_response.json()
        assert detail_data["data"]["description"] == "Updated collection description"
        assert detail_data["data"]["is_public"] is True

    async def test_delete_collection(self, client):
        """测试删除 Collection"""

        unique_id = _generate_unique_id()
        collection_data = {
            "name": f"test_delete_collection_{unique_id}",
            "description": f"Test delete {unique_id}",
            "is_public": False,
            "workflow_template": {}, "extra_config": {}
        }

        # 创建集合
        await client.post("/v1/collection", json=collection_data)

        # 从列表中获取刚创建的记录
        list_response = await client.get("/v1/collection")
        list_data = list_response.json()
        items = list_data["data"]["items"]
        created_item = next((item for item in items if item["name"] == collection_data["name"]), None)
        assert created_item is not None
        pk = created_item["id"]

        # 删除集合
        response = await client.delete(f"/v1/collection/{pk}")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0

        # 验证删除（应该返回错误）
        detail_response = await client.get(f"/v1/collection/{pk}")
        assert detail_response.status_code == 200
        assert detail_response.json()["code"] != 0

    async def test_filter_collections(self, client):
        """测试过滤 Collection 列表"""

        unique_id_1 = _generate_unique_id()
        unique_id_2 = _generate_unique_id()

        # 创建多个集合
        await client.post("/v1/collection", json={
            "name": f"test_filter_public_{unique_id_1}",
            "is_public": True,
            "workflow_template": {}, "extra_config": {}
        })
        await client.post("/v1/collection", json={
            "name": f"test_filter_private_{unique_id_2}",
            "is_public": False,
            "workflow_template": {}, "extra_config": {}
        })

        # 过滤 is_public
        response = await client.get("/v1/collection?is_public=true")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        items = data["data"]["items"]
        public_items = [item for item in items if "test_filter" in item["name"]]
        assert all(item["is_public"] is True for item in public_items)

        # 过滤 name
        response = await client.get("/v1/collection?name__icontains=public")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        items = data["data"]["items"]
        assert any("public" in item["name"] for item in items)

    async def test_full_crud_flow(self, client):
        """测试完整的 CRUD 流程：创建 -> 列表 -> 详情 -> 更新 -> 删除"""

        unique_id = _generate_unique_id()
        collection_data = {
            "name": f"test_full_flow_{unique_id}",
            "description": f"Full flow test {unique_id}",
            "is_public": False,
            "workflow_template": {}, "extra_config": {}
        }

        # 1. 创建
        create_response = await client.post("/v1/collection", json=collection_data)
        assert create_response.status_code == 200
        assert create_response.json()["code"] == 0

        # 2. 列表并获取 id
        list_response = await client.get("/v1/collection")
        assert list_response.status_code == 200
        list_data = list_response.json()
        assert list_data["code"] == 0
        items = list_data["data"]["items"]
        created_item = next((item for item in items if item["name"] == collection_data["name"]), None)
        assert created_item is not None
        pk = created_item["id"]

        # 3. 详情
        detail_response = await client.get(f"/v1/collection/{pk}")
        assert detail_response.status_code == 200
        detail_data = detail_response.json()
        assert detail_data["code"] == 0
        assert detail_data["data"]["name"] == collection_data["name"]

        # 4. 更新
        update_response = await client.put(
            f"/v1/collection/{pk}",
            json={"description": "Full flow collection update"}
        )
        assert update_response.status_code == 200
        assert update_response.json()["code"] == 0

        # 验证更新
        updated_detail = await client.get(f"/v1/collection/{pk}")
        assert updated_detail.json()["data"]["description"] == "Full flow collection update"

        # 5. 删除
        delete_response = await client.delete(f"/v1/collection/{pk}")
        assert delete_response.status_code == 200
        assert delete_response.json()["code"] == 0

        # 验证删除
        final_detail = await client.get(f"/v1/collection/{pk}")
        assert final_detail.json()["code"] != 0

    async def test_collection_name_uniqueness(self, client):
        """测试 Collection 名称唯一性约束"""

        unique_id = _generate_unique_id()
        collection_data = {
            "name": f"test_unique_{unique_id}",
            "description": "Uniqueness test",
            "is_public": False,
            "workflow_template": {}, "extra_config": {}, "user_id": str(uuid.uuid4())
        }

        # 创建第一个集合
        await client.post("/v1/collection", json=collection_data)

        # 尝试创建同名集合（应该失败）
        response = await client.post("/v1/collection", json=collection_data)
        assert response.status_code == 200
        data = response.json()
        # 应该返回错误，因为名称重复
        assert data["code"] != 0

    async def test_search_collections(self, client):
        """测试搜索 Collection"""

        unique_id_1 = _generate_unique_id()
        unique_id_2 = _generate_unique_id()

        # 创建多个集合
        await client.post("/v1/collection", json={
            "name": f"search_test_document_collection_{unique_id_1}",
            "description": "This is a collection for documents",
            "is_public": False,
            "workflow_template": {}, "extra_config": {}
        })
        await client.post("/v1/collection", json={
            "name": f"search_test_image_collection_{unique_id_2}",
            "description": "This is a collection for images",
            "is_public": False,
            "workflow_template": {}, "extra_config": {}
        })

        # 搜索 "document"
        response = await client.get("/v1/collection?search=document")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        items = data["data"]["items"]
        # 应该至少包含一个结果
        assert len(items) > 0
        # 所有结果都应该包含 "document"（在 name 或 description 中）
        test_items = [item for item in items if "search_test" in item["name"]]
        assert any("document" in item["name"].lower() or "document" in (item.get("description") or "").lower()
                   for item in test_items)
