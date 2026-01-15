"""Document CRUD API 测试"""

import pytest
from pathlib import Path


@pytest.mark.asyncio
class TestDocumentCRUD:
    """Document CRUD 测试"""

    async def test_create_document_by_upload(self, client, test_collection, default_file_source):
        """测试通过上传文件创建 Document"""
        # 准备上传数据
        from io import BytesIO
        file_content = b"This is a test document content for uploading."
        file = BytesIO(file_content)
        file.name = "test_upload.txt"

        # 准备表单数据
        data = {
            "collection_id": test_collection.id,
            "display_name": "Uploaded Test Document",
            "short_summary": "Test upload summary",
            "status": "pending",
        }

        # 上传文件
        response = await client.post(
            "/v1/service/document/document/upload",
            data={"file": ("test_upload.txt", file, "text/plain")},
            params=data
        )
        assert response.status_code == 200
        result = response.json()
        assert result["code"] == 0

    async def test_create_document_by_upload_with_default_source(self, client, test_collection, default_file_source):
        """测试使用默认文件源上传文件创建 Document"""
        # 确保有默认文件源
        assert default_file_source.is_default is True

        # 准备上传数据
        from io import BytesIO
        file_content = b"This is a test document for default source."
        file = BytesIO(file_content)
        file.name = "test_default_source.txt"

        # 准备表单数据（不指定 file_source_id）
        data = {
            "collection_id": test_collection.id,
            "display_name": "Default Source Document",
            "status": "pending",
        }

        # 上传文件
        response = await client.post(
            "/v1/service/document/document/upload",
            data={"file": ("test_default_source.txt", file, "text/plain")},
            params=data
        )
        assert response.status_code == 200
        result = response.json()
        assert result["code"] == 0

    async def test_create_document_by_upload_with_specific_source(self, client, test_collection, file_source_data):
        """测试使用指定文件源上传文件创建 Document"""
        # 先创建一个非默认的文件源
        from ext.ext_tortoise.models.knowledge_base import FileSource
        file_source = await FileSource.create(
            name="test_upload_source",
            type=file_source_data["type"],
            config=file_source_data["config"],
            is_enabled=True,
            is_default=False,
        )

        try:
            # 准备上传数据
            from io import BytesIO
            file_content = b"This is a test document for specific source."
            file = BytesIO(file_content)
            file.name = "test_specific_source.txt"

            # 准备表单数据（指定 file_source_id）
            data = {
                "collection_id": test_collection.id,
                "display_name": "Specific Source Document",
                "file_source_id": file_source.id,
                "status": "pending",
            }

            # 上传文件
            response = await client.post(
                "/v1/service/document/document/upload",
                data={"file": ("test_specific_source.txt", file, "text/plain")},
                params=data
            )
            assert response.status_code == 200
            result = response.json()
            assert result["code"] == 0
        finally:
            await file_source.delete()

    async def test_create_document_by_uri(self, client, test_collection, default_file_source):
        """测试通过 URI 创建 Document（文件源中已存在文件）"""
        # 先在文件源中创建一个测试文件
        import tempfile
        import os

        # 获取文件源的 base_path
        base_path = default_file_source.config.get("base_path", "/tmp")
        os.makedirs(base_path, exist_ok=True)

        # 创建测试文件
        test_file_path = os.path.join(base_path, "test_uri_file.txt")
        test_content = b"This is a test file for URI creation."
        with open(test_file_path, "wb") as f:
            f.write(test_content)

        try:
            # 准备创建数据
            document_data = {
                "collection_id": test_collection.id,
                "file_source_id": default_file_source.id,
                "uri": test_file_path,
                "file_name": "test_uri_file.txt",
                "display_name": "URI Test Document",
                "extension": "txt",
                "status": "pending",
            }

            # 创建文档
            response = await client.post("/v1/service/document/document", json=document_data)
            assert response.status_code == 200
            result = response.json()
            assert result["code"] == 0
        finally:
            # 清理测试文件
            if os.path.exists(test_file_path):
                os.remove(test_file_path)

    async def test_create_document_by_uri_file_not_found(self, client, test_collection, default_file_source):
        """测试通过 URI 创建 Document 时文件不存在的情况"""
        # 准备创建数据（使用不存在的 URI）
        document_data = {
            "collection_id": test_collection.id,
            "file_source_id": default_file_source.id,
            "uri": "/tmp/nonexistent_file.txt",
            "file_name": "nonexistent.txt",
            "display_name": "Nonexistent Document",
            "extension": "txt",
            "status": "pending",
        }

        # 创建文档（应该失败）
        response = await client.post("/v1/service/document/document", json=document_data)
        assert response.status_code == 200
        result = response.json()
        assert result["code"] != 0
        assert "不存在" in result["message"] or "not found" in result["message"].lower()

    async def test_create_document_by_uri_invalid_source(self, client, test_collection):
        """测试通过 URI 创建 Document 时文件源无效的情况"""
        # 准备创建数据（使用不存在的 file_source_id）
        document_data = {
            "collection_id": test_collection.id,
            "file_source_id": 99999,  # 不存在的 ID
            "uri": "/tmp/test.txt",
            "file_name": "test.txt",
            "display_name": "Test Document",
            "extension": "txt",
            "status": "pending",
        }

        # 创建文档（应该失败）
        response = await client.post("/v1/service/document/document", json=document_data)
        assert response.status_code == 200
        result = response.json()
        assert result["code"] != 0

    async def test_list_documents(self, client, test_collection, default_file_source):
        """测试获取 Document 列表"""
        # 先创建几个文档
        import tempfile
        import os

        base_path = default_file_source.config.get("base_path", "/tmp")
        os.makedirs(base_path, exist_ok=True)

        # 创建测试文件
        test_file_1 = os.path.join(base_path, "test_list_1.txt")
        test_file_2 = os.path.join(base_path, "test_list_2.txt")

        with open(test_file_1, "wb") as f:
            f.write(b"Test content 1")
        with open(test_file_2, "wb") as f:
            f.write(b"Test content 2")

        try:
            # 创建文档
            await client.post("/v1/service/document/document", json={
                "collection_id": test_collection.id,
                "file_source_id": default_file_source.id,
                "uri": test_file_1,
                "file_name": "test_list_1.txt",
                "display_name": "List Test 1",
                "extension": "txt",
                "status": "pending",
            })
            await client.post("/v1/service/document/document", json={
                "collection_id": test_collection.id,
                "file_source_id": default_file_source.id,
                "uri": test_file_2,
                "file_name": "test_list_2.txt",
                "display_name": "List Test 2",
                "extension": "txt",
                "status": "pending",
            })

            # 获取列表
            response = await client.get("/v1/service/document/document")
            assert response.status_code == 200
            data = response.json()
            assert data["code"] == 0
            assert "data" in data
            assert "items" in data["data"]
            assert len(data["data"]["items"]) >= 2
        finally:
            # 清理测试文件
            for file_path in [test_file_1, test_file_2]:
                if os.path.exists(file_path):
                    os.remove(file_path)

    async def test_get_document_detail(self, client, test_collection, default_file_source):
        """测试获取 Document 详情"""
        import os

        base_path = default_file_source.config.get("base_path", "/tmp")
        os.makedirs(base_path, exist_ok=True)

        # 创建测试文件
        test_file = os.path.join(base_path, "test_detail.txt")
        with open(test_file, "wb") as f:
            f.write(b"Test content for detail")

        try:
            # 创建文档
            await client.post("/v1/service/document/document", json={
                "collection_id": test_collection.id,
                "file_source_id": default_file_source.id,
                "uri": test_file,
                "file_name": "test_detail.txt",
                "display_name": "Detail Test Document",
                "extension": "txt",
                "status": "pending",
            })

            # 从列表中获取刚创建的记录
            list_response = await client.get("/v1/service/document/document")
            list_data = list_response.json()
            items = list_data["data"]["items"]
            created_item = items[-1]  # 获取最后一个创建的
            pk = created_item["id"]

            # 获取详情
            response = await client.get(f"/v1/service/document/document/{pk}")
            assert response.status_code == 200
            data = response.json()
            assert data["code"] == 0
            assert data["data"]["file_name"] == "test_detail.txt"
            assert data["data"]["display_name"] == "Detail Test Document"
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    async def test_update_document(self, client, test_collection, default_file_source):
        """测试更新 Document"""
        import os

        base_path = default_file_source.config.get("base_path", "/tmp")
        os.makedirs(base_path, exist_ok=True)

        # 创建测试文件
        test_file = os.path.join(base_path, "test_update.txt")
        with open(test_file, "wb") as f:
            f.write(b"Test content for update")

        try:
            # 创建文档
            await client.post("/v1/service/document/document", json={
                "collection_id": test_collection.id,
                "file_source_id": default_file_source.id,
                "uri": test_file,
                "file_name": "test_update.txt",
                "display_name": "Original Name",
                "extension": "txt",
                "status": "pending",
            })

            # 从列表中获取刚创建的记录
            list_response = await client.get("/v1/service/document/document")
            list_data = list_response.json()
            items = list_data["data"]["items"]
            created_item = items[-1]
            pk = created_item["id"]

            # 更新文档
            update_data = {
                "display_name": "Updated Document Name",
                "short_summary": "Updated summary",
                "status": "indexed",
            }
            response = await client.put(f"/v1/service/document/document/{pk}", json=update_data)
            assert response.status_code == 200
            data = response.json()
            assert data["code"] == 0

            # 验证更新
            detail_response = await client.get(f"/v1/service/document/document/{pk}")
            detail_data = detail_response.json()
            assert detail_data["data"]["display_name"] == "Updated Document Name"
            assert detail_data["data"]["short_summary"] == "Updated summary"
            assert detail_data["data"]["status"] == "indexed"
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    async def test_delete_document(self, client, test_collection, default_file_source):
        """测试删除 Document"""
        import os

        base_path = default_file_source.config.get("base_path", "/tmp")
        os.makedirs(base_path, exist_ok=True)

        # 创建测试文件
        test_file = os.path.join(base_path, "test_delete.txt")
        with open(test_file, "wb") as f:
            f.write(b"Test content for delete")

        try:
            # 创建文档
            await client.post("/v1/service/document/document", json={
                "collection_id": test_collection.id,
                "file_source_id": default_file_source.id,
                "uri": test_file,
                "file_name": "test_delete.txt",
                "display_name": "Delete Test Document",
                "extension": "txt",
                "status": "pending",
            })

            # 从列表中获取刚创建的记录
            list_response = await client.get("/v1/service/document/document")
            list_data = list_response.json()
            items = list_data["data"]["items"]
            created_item = items[-1]
            pk = created_item["id"]

            # 删除文档
            response = await client.delete(f"/v1/service/document/document/{pk}")
            assert response.status_code == 200
            data = response.json()
            assert data["code"] == 0

            # 验证删除（应该返回错误）
            detail_response = await client.get(f"/v1/service/document/document/{pk}")
            assert detail_response.status_code == 200
            assert detail_response.json()["code"] != 0
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    async def test_filter_documents(self, client, test_collection, default_file_source):
        """测试过滤 Document 列表"""
        import os

        base_path = default_file_source.config.get("base_path", "/tmp")
        os.makedirs(base_path, exist_ok=True)

        # 创建测试文件
        test_file_1 = os.path.join(base_path, "test_filter_pending.txt")
        test_file_2 = os.path.join(base_path, "test_filter_indexed.txt")

        with open(test_file_1, "wb") as f:
            f.write(b"Test content 1")
        with open(test_file_2, "wb") as f:
            f.write(b"Test content 2")

        try:
            # 创建不同状态的文档
            await client.post("/v1/service/document/document", json={
                "collection_id": test_collection.id,
                "file_source_id": default_file_source.id,
                "uri": test_file_1,
                "file_name": "test_filter_pending.txt",
                "display_name": "Pending Document",
                "extension": "txt",
                "status": "pending",
            })
            await client.post("/v1/service/document/document", json={
                "collection_id": test_collection.id,
                "file_source_id": default_file_source.id,
                "uri": test_file_2,
                "file_name": "test_filter_indexed.txt",
                "display_name": "Indexed Document",
                "extension": "txt",
                "status": "indexed",
            })

            # 过滤 status
            response = await client.get("/v1/service/document/document?status=pending")
            assert response.status_code == 200
            data = response.json()
            assert data["code"] == 0
            items = data["data"]["items"]
            assert all(item["status"] == "pending" for item in items)

            # 过滤 collection_id
            response = await client.get(f"/v1/service/document/document?collection_id={test_collection.id}")
            assert response.status_code == 200
            data = response.json()
            assert data["code"] == 0
            items = data["data"]["items"]
            assert all(item["collection_id"] == test_collection.id for item in items)
        finally:
            for file_path in [test_file_1, test_file_2]:
                if os.path.exists(file_path):
                    os.remove(file_path)

    async def test_full_crud_flow(self, client, test_collection, default_file_source):
        """测试完整的 CRUD 流程：创建 -> 列表 -> 详情 -> 更新 -> 删除"""
        import os

        base_path = default_file_source.config.get("base_path", "/tmp")
        os.makedirs(base_path, exist_ok=True)

        # 创建测试文件
        test_file = os.path.join(base_path, "test_full_flow.txt")
        with open(test_file, "wb") as f:
            f.write(b"Full flow test content")

        try:
            # 1. 创建
            create_response = await client.post("/v1/service/document/document", json={
                "collection_id": test_collection.id,
                "file_source_id": default_file_source.id,
                "uri": test_file,
                "file_name": "test_full_flow.txt",
                "display_name": "Full Flow Document",
                "extension": "txt",
                "status": "pending",
            })
            assert create_response.status_code == 200
            assert create_response.json()["code"] == 0

            # 2. 列表并获取 id
            list_response = await client.get("/v1/service/document/document")
            assert list_response.status_code == 200
            list_data = list_response.json()
            assert list_data["code"] == 0
            items = list_data["data"]["items"]
            created_item = items[-1]
            pk = created_item["id"]

            # 3. 详情
            detail_response = await client.get(f"/v1/service/document/document/{pk}")
            assert detail_response.status_code == 200
            detail_data = detail_response.json()
            assert detail_data["code"] == 0
            assert detail_data["data"]["file_name"] == "test_full_flow.txt"

            # 4. 更新
            update_response = await client.put(
                f"/v1/service/document/document/{pk}",
                json={
                    "display_name": "Updated Full Flow Document",
                    "short_summary": "Full flow summary",
                    "status": "indexed"
                }
            )
            assert update_response.status_code == 200
            assert update_response.json()["code"] == 0

            # 验证更新
            updated_detail = await client.get(f"/v1/service/document/document/{pk}")
            assert updated_detail.json()["data"]["display_name"] == "Updated Full Flow Document"
            assert updated_detail.json()["data"]["status"] == "indexed"

            # 5. 删除
            delete_response = await client.delete(f"/v1/service/document/document/{pk}")
            assert delete_response.status_code == 200
            assert delete_response.json()["code"] == 0

            # 验证删除
            final_detail = await client.get(f"/v1/service/document/document/{pk}")
            assert final_detail.json()["code"] != 0
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    async def test_search_documents(self, client, test_collection, default_file_source):
        """测试搜索 Document"""
        import os

        base_path = default_file_source.config.get("base_path", "/tmp")
        os.makedirs(base_path, exist_ok=True)

        # 创建测试文件
        test_file_1 = os.path.join(base_path, "search_test_document.txt")
        test_file_2 = os.path.join(base_path, "search_test_report.txt")

        with open(test_file_1, "wb") as f:
            f.write(b"Search test document content")
        with open(test_file_2, "wb") as f:
            f.write(b"Search test report content")

        try:
            # 创建文档
            await client.post("/v1/service/document/document", json={
                "collection_id": test_collection.id,
                "file_source_id": default_file_source.id,
                "uri": test_file_1,
                "file_name": "search_test_document.txt",
                "display_name": "Search Test Document",
                "extension": "txt",
                "status": "pending",
            })
            await client.post("/v1/service/document/document", json={
                "collection_id": test_collection.id,
                "file_source_id": default_file_source.id,
                "uri": test_file_2,
                "file_name": "search_test_report.txt",
                "display_name": "Search Test Report",
                "extension": "txt",
                "status": "pending",
            })

            # 搜索 "document"
            response = await client.get("/v1/service/document/document?search=document")
            assert response.status_code == 200
            data = response.json()
            assert data["code"] == 0
            items = data["data"]["items"]
            # 应该至少包含一个结果
            assert len(items) > 0
            # 所有结果都应该包含 "document"（在 file_name 或 display_name 中）
            assert any("document" in item["file_name"].lower() or "document" in item["display_name"].lower()
                       for item in items)
        finally:
            for file_path in [test_file_1, test_file_2]:
                if os.path.exists(file_path):
                    os.remove(file_path)

    async def test_filter_by_file_name(self, client, test_collection, default_file_source):
        """测试按文件名过滤 Document"""
        import os

        base_path = default_file_source.config.get("base_path", "/tmp")
        os.makedirs(base_path, exist_ok=True)

        # 创建测试文件
        test_file_1 = os.path.join(base_path, "name_test_1.pdf")
        test_file_2 = os.path.join(base_path, "name_test_2.txt")

        with open(test_file_1, "wb") as f:
            f.write(b"PDF content")
        with open(test_file_2, "wb") as f:
            f.write(b"TXT content")

        try:
            # 创建文档
            await client.post("/v1/service/document/document", json={
                "collection_id": test_collection.id,
                "file_source_id": default_file_source.id,
                "uri": test_file_1,
                "file_name": "name_test_1.pdf",
                "display_name": "PDF Document",
                "extension": "pdf",
                "status": "pending",
            })
            await client.post("/v1/service/document/document", json={
                "collection_id": test_collection.id,
                "file_source_id": default_file_source.id,
                "uri": test_file_2,
                "file_name": "name_test_2.txt",
                "display_name": "TXT Document",
                "extension": "txt",
                "status": "pending",
            })

            # 过滤 extension
            response = await client.get("/v1/service/document/document?extension=txt")
            assert response.status_code == 200
            data = response.json()
            assert data["code"] == 0
            items = data["data"]["items"]
            assert all(item["extension"] == "txt" for item in items)

            # 过滤 file_name__icontains
            response = await client.get("/v1/service/document/document?file_name__icontains=pdf")
            assert response.status_code == 200
            data = response.json()
            assert data["code"] == 0
            items = data["data"]["items"]
            assert any("pdf" in item["file_name"].lower() for item in items)
        finally:
            for file_path in [test_file_1, test_file_2]:
                if os.path.exists(file_path):
                    os.remove(file_path)

    async def test_document_status_transitions(self, client, test_collection, default_file_source):
        """测试文档状态转换"""
        import os
        from ext.ext_tortoise.enums import DocumentStatusEnum

        base_path = default_file_source.config.get("base_path", "/tmp")
        os.makedirs(base_path, exist_ok=True)

        # 创建测试文件
        test_file = os.path.join(base_path, "status_test.txt")
        with open(test_file, "wb") as f:
            f.write(b"Status test content")

        try:
            # 创建文档（初始状态为 pending）
            await client.post("/v1/service/document/document", json={
                "collection_id": test_collection.id,
                "file_source_id": default_file_source.id,
                "uri": test_file,
                "file_name": "status_test.txt",
                "display_name": "Status Test Document",
                "extension": "txt",
                "status": DocumentStatusEnum.pending.value,
            })

            # 获取文档 ID
            list_response = await client.get("/v1/service/document/document")
            items = list_response.json()["data"]["items"]
            created_item = items[-1]
            pk = created_item["id"]

            # 验证初始状态
            detail = await client.get(f"/v1/service/document/document/{pk}").json()
            assert detail["data"]["status"] == DocumentStatusEnum.pending.value

            # 更新为 fetching
            await client.put(f"/v1/service/document/document/{pk}", json={
                "status": DocumentStatusEnum.fetching.value
            })
            detail = await client.get(f"/v1/service/document/document/{pk}").json()
            assert detail["data"]["status"] == DocumentStatusEnum.fetching.value

            # 更新为 loaded
            await client.put(f"/v1/service/document/document/{pk}", json={
                "status": DocumentStatusEnum.loaded.value
            })
            detail = await client.get(f"/v1/service/document/document/{pk}").json()
            assert detail["data"]["status"] == DocumentStatusEnum.loaded.value

            # 更新为 indexed
            await client.put(f"/v1/service/document/document/{pk}", json={
                "status": DocumentStatusEnum.indexed.value
            })
            detail = await client.get(f"/v1/service/document/document/{pk}").json()
            assert detail["data"]["status"] == DocumentStatusEnum.indexed.value
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)
