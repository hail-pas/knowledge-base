"""FileSource CRUD API 测试"""

import pytest


@pytest.mark.asyncio
class TestFileSourceCRUD:
    """FileSource CRUD 测试"""

    async def test_create_file_source(self, client, file_source_data):
        """测试创建 FileSource"""
        response = await client.post("/v1/service/document/file-source", json=file_source_data)
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0

    async def test_list_file_sources(self, client, file_source_data):
        """测试获取 FileSource 列表"""
        # 创建多个文件源（使用唯一名称）
        unique_id_1 = f"{int(__import__('time').time() * 1000)}{''.join(__import__('random').choices(__import__('string').ascii_lowercase, k=4))}"
        unique_id_2 = f"{int(__import__('time').time() * 1000)}{''.join(__import__('random').choices(__import__('string').ascii_lowercase, k=4))}"

        await client.post("/v1/service/document/file-source", json={
            **file_source_data,
            "name": f"test_list_source_{unique_id_1}",
        })
        await client.post("/v1/service/document/file-source", json={
            **file_source_data,
            "name": f"test_list_source_{unique_id_2}",
        })

        # 获取列表
        response = await client.get("/v1/service/document/file-source")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert "data" in data
        assert "items" in data["data"]
        assert len(data["data"]["items"]) >= 2

    async def test_get_file_source_detail(self, client, file_source_data):
        """测试获取 FileSource 详情"""
        # 创建文件源
        response = await client.post("/v1/service/document/file-source", json=file_source_data)
        assert response.status_code == 200

        # 从列表中获取刚创建的记录
        list_response = await client.get("/v1/service/document/file-source")
        list_data = list_response.json()
        items = list_data["data"]["items"]
        created_item = items[-1]  # 获取最后一个创建的
        pk = created_item["id"]

        # 获取详情
        response = await client.get(f"/v1/service/document/file-source/{pk}")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert data["data"]["name"] == file_source_data["name"]
        assert data["data"]["type"] == file_source_data["type"]

    async def test_update_file_source(self, client, file_source_data):
        """测试更新 FileSource"""
        # 创建文件源
        await client.post("/v1/service/document/file-source", json=file_source_data)

        # 从列表中获取刚创建的记录
        list_response = await client.get("/v1/service/document/file-source")
        list_data = list_response.json()
        items = list_data["data"]["items"]
        created_item = items[-1]
        pk = created_item["id"]

        # 更新文件源
        update_data = {
            "description": "Updated file source description",
            "is_enabled": False,
        }
        response = await client.put(f"/v1/service/document/file-source/{pk}", json=update_data)
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0

        # 验证更新
        detail_response = await client.get(f"/v1/service/document/file-source/{pk}")
        detail_data = detail_response.json()
        assert detail_data["data"]["description"] == "Updated file source description"
        assert detail_data["data"]["is_enabled"] is False

    async def test_delete_file_source(self, client, file_source_data):
        """测试删除 FileSource"""
        # 创建文件源
        await client.post("/v1/service/document/file-source", json=file_source_data)

        # 从列表中获取刚创建的记录
        list_response = await client.get("/v1/service/document/file-source")
        list_data = list_response.json()
        items = list_data["data"]["items"]
        created_item = items[-1]
        pk = created_item["id"]

        # 删除文件源
        response = await client.delete(f"/v1/service/document/file-source/{pk}")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0

        # 验证删除（应该返回错误）
        detail_response = await client.get(f"/v1/service/document/file-source/{pk}")
        assert detail_response.status_code == 200
        assert detail_response.json()["code"] != 0

    async def test_filter_file_sources(self, client, file_source_data):
        """测试过滤 FileSource 列表"""
        # 创建多个文件源
        await client.post("/v1/service/document/file-source", json={
            **file_source_data,
            "name": "test_filter_enabled",
            "is_enabled": True,
        })
        await client.post("/v1/service/document/file-source", json={
            **file_source_data,
            "name": "test_filter_disabled",
            "is_enabled": False,
        })

        # 过滤 is_enabled
        response = await client.get("/v1/service/document/file-source?is_enabled=true")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        items = data["data"]["items"]
        assert all(item["is_enabled"] is True for item in items)

        # 过滤 type
        response = await client.get("/v1/service/document/file-source?type=local_file")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        items = data["data"]["items"]
        assert all(item["type"] == "local_file" for item in items)

    async def test_filter_by_name(self, client, file_source_data):
        """测试按名称过滤 FileSource"""
        # 创建多个文件源
        await client.post("/v1/service/document/file-source", json={
            **file_source_data,
            "name": "test_name_filter_oss",
            "type": "aliyun_oss",
        })
        await client.post("/v1/service/document/file-source", json={
            **file_source_data,
            "name": "test_name_filter_s3",
            "type": "s3",
        })

        # 过滤 name__icontains
        response = await client.get("/v1/service/document/file-source?name__icontains=oss")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        items = data["data"]["items"]
        assert any("oss" in item["name"].lower() for item in items)

    async def test_full_crud_flow(self, client, file_source_data):
        """测试完整的 CRUD 流程：创建 -> 列表 -> 详情 -> 更新 -> 删除"""
        # 1. 创建
        create_response = await client.post("/v1/service/document/file-source", json=file_source_data)
        assert create_response.status_code == 200
        assert create_response.json()["code"] == 0

        # 2. 列表并获取 id
        list_response = await client.get("/v1/service/document/file-source")
        assert list_response.status_code == 200
        list_data = list_response.json()
        assert list_data["code"] == 0
        items = list_data["data"]["items"]
        created_item = items[-1]
        pk = created_item["id"]

        # 3. 详情
        detail_response = await client.get(f"/v1/service/document/file-source/{pk}")
        assert detail_response.status_code == 200
        detail_data = detail_response.json()
        assert detail_data["code"] == 0
        assert detail_data["data"]["name"] == file_source_data["name"]

        # 4. 更新
        update_response = await client.put(
            f"/v1/service/document/file-source/{pk}",
            json={"description": "Full flow update", "is_default": True}
        )
        assert update_response.status_code == 200
        assert update_response.json()["code"] == 0

        # 验证更新
        updated_detail = await client.get(f"/v1/service/document/file-source/{pk}")
        assert updated_detail.json()["data"]["description"] == "Full flow update"
        assert updated_detail.json()["data"]["is_default"] is True

        # 5. 删除
        delete_response = await client.delete(f"/v1/service/document/file-source/{pk}")
        assert delete_response.status_code == 200
        assert delete_response.json()["code"] == 0

        # 验证删除
        final_detail = await client.get(f"/v1/service/document/file-source/{pk}")
        assert final_detail.json()["code"] != 0

    async def test_file_source_config_validation(self, client, file_source_data):
        """测试 FileSource 配置验证"""
        # 创建一个有效的文件源
        response = await client.post("/v1/service/document/file-source", json=file_source_data)
        assert response.status_code == 200
        assert response.json()["code"] == 0

        # 尝试创建不同类型的文件源
        oss_data = {
            **file_source_data,
            "name": "test_oss_source",
            "type": "aliyun_oss",
            "config": {
                "endpoint": "oss-cn-hangzhou.aliyuncs.com",
                "access_key_id": "test_key",
                "access_key_secret": "test_secret",
                "bucket_name": "test-bucket",
            },
        }
        response = await client.post("/v1/service/document/file-source", json=oss_data)
        assert response.status_code == 200
        assert response.json()["code"] == 0

    async def test_default_file_source(self, client, file_source_data):
        """测试默认文件源"""
        # 创建多个文件源
        await client.post("/v1/service/document/file-source", json={
            **file_source_data,
            "name": "test_default_source_1",
            "is_default": False,
        })
        await client.post("/v1/service/document/file-source", json={
            **file_source_data,
            "name": "test_default_source_2",
            "is_default": True,
        })

        # 过滤默认文件源
        response = await client.get("/v1/service/document/file-source?is_default=true")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        items = data["data"]["items"]
        assert len(items) >= 1
        assert all(item["is_default"] is True for item in items)

    async def test_search_file_sources(self, client, file_source_data):
        """测试搜索 FileSource"""
        # 创建多个文件源
        await client.post("/v1/service/document/file-source", json={
            **file_source_data,
            "name": "search_test_oss_source",
            "description": "This is an OSS source for testing",
        })
        await client.post("/v1/service/document/file-source", json={
            **file_source_data,
            "name": "search_test_s3_source",
            "description": "This is an S3 source for testing",
        })

        # 搜索 "OSS"
        response = await client.get("/v1/service/document/file-source?search=OSS")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        items = data["data"]["items"]
        # 应该至少包含一个结果
        assert len(items) > 0
        # 所有结果都应该包含 "oss"（在 name 或 description 中）
        assert any("oss" in item["name"].lower() or "oss" in (item.get("description") or "").lower()
                   for item in items)

    async def test_multiple_file_sources_with_same_name(self, client, file_source_data):
        """测试同名但不同 deleted_at 的 FileSource 可以创建"""
        # 创建第一个文件源
        response = await client.post("/v1/service/document/file-source", json=file_source_data)
        assert response.status_code == 200
        assert response.json()["code"] == 0

        # 从列表中获取刚创建的记录
        list_response = await client.get("/v1/service/document/file-source")
        list_data = list_response.json()
        items = list_data["data"]["items"]
        created_item = items[-1]
        pk = created_item["id"]

        # 删除第一个文件源
        delete_response = await client.delete(f"/v1/service/document/file-source/{pk}")
        assert delete_response.status_code == 200
        assert delete_response.json()["code"] == 0

        # 使用相同名称创建新文件源（应该成功，因为旧的已删除）
        response = await client.post("/v1/service/document/file-source", json=file_source_data)
        assert response.status_code == 200
        data = response.json()
        # 应该成功
        assert data["code"] == 0
