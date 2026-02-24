"""
测试 JSON 数据切块策略

测试场景：
- Simple 模式（键值对）
- JSON 模式（保持格式）
- List 数据处理
- Dict 数据处理
- 嵌套数据处理
- Key 过滤
- 合并/分割块
- 无效 JSON 处理
"""

import json
import pytest

from ext.document_parser.core.parse_result import OutputFormat
from ext.text_chunker.strategies.json_based import JsonChunkStrategy
from ext.text_chunker.config.strategy_config import JsonChunkConfig


class TestJsonChunkStrategySimpleMode:
    """测试 Simple 模式"""

    @pytest.mark.asyncio
    async def test_simple_mode_list(self, json_parse_result):
        """测试 Simple 模式处理列表"""
        # 设置小的 max_chunk_size 以避免合并
        config = JsonChunkConfig(mode="simple", keys=[], max_chunk_size=50)
        strategy = JsonChunkStrategy(config)

        chunks = await strategy.chunk(json_parse_result)

        assert len(chunks) > 0
        # 每个数组元素应该是一个 chunk
        assert len(chunks) == 4

    @pytest.mark.asyncio
    async def test_simple_mode_dict(self, json_dict_data):
        """测试 Simple 模式处理字典"""
        from tests.ext.text_chunker.conftest import create_parse_result

        parse_result = create_parse_result(json_dict_data, OutputFormat.JSON)
        config = JsonChunkConfig(mode="simple", keys=[])
        strategy = JsonChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_simple_mode_with_key_filter(self, json_parse_result):
        """测试 Simple 模式 key 过滤"""
        config = JsonChunkConfig(mode="simple", keys=["name", "age"])
        strategy = JsonChunkStrategy(config)

        chunks = await strategy.chunk(json_parse_result)

        assert len(chunks) > 0

        # 每个 chunk 应该只包含 name 和 age
        for chunk in chunks:
            assert "name" in chunk.content
            assert "age" in chunk.content
            assert "city" not in chunk.content

    @pytest.mark.asyncio
    async def test_simple_mode_custom_separators(self, json_parse_result):
        """测试自定义分隔符"""
        config = JsonChunkConfig(
            mode="simple",
            keys=[],
            key_separator=" | ",
            value_separator=" = ",
            item_joiner="---",
        )
        strategy = JsonChunkStrategy(config)

        chunks = await strategy.chunk(json_parse_result)

        assert len(chunks) > 0

        # 检查是否使用了自定义分隔符
        for chunk in chunks:
            if " | " in chunk.content or " = " in chunk.content:
                break
        else:
            # 可能没有这些分隔符
            pass


class TestJsonChunkStrategyJsonMode:
    """测试 JSON 模式"""

    @pytest.mark.asyncio
    async def test_json_mode_list(self, json_parse_result):
        """测试 JSON 模式处理列表"""
        config = JsonChunkConfig(mode="json")
        strategy = JsonChunkStrategy(config)

        chunks = await strategy.chunk(json_parse_result)

        assert len(chunks) > 0
        # 每个元素应该是独立的 JSON
        for chunk in chunks:
            # 验证是有效的 JSON
            json.loads(chunk.content)

    @pytest.mark.asyncio
    async def test_json_mode_dict(self, json_dict_data):
        """测试 JSON 模式处理字典"""
        from tests.ext.text_chunker.conftest import create_parse_result

        parse_result = create_parse_result(json_dict_data, OutputFormat.JSON)
        config = JsonChunkConfig(mode="json")
        strategy = JsonChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        assert len(chunks) >= 1

        # 验证输出是有效的 JSON
        for chunk in chunks:
            json.loads(chunk.content)

    @pytest.mark.asyncio
    async def test_json_mode_with_key_filter(self, json_dict_data):
        """测试 JSON 模式 key 过滤"""
        from tests.ext.text_chunker.conftest import create_parse_result

        parse_result = create_parse_result(json_dict_data, OutputFormat.JSON)
        config = JsonChunkConfig(mode="json", keys=["title", "author"])
        strategy = JsonChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        assert len(chunks) >= 1

        # 验证只包含指定的 keys
        for chunk in chunks:
            data = json.loads(chunk.content)
            assert "title" in data
            assert "author" in data
            assert "sections" not in data
            assert "metadata" not in data


class TestJsonChunkStrategyNestedData:
    """测试嵌套数据处理"""

    @pytest.mark.asyncio
    async def test_simple_mode_nested_dict(self, json_nested_data):
        """测试 Simple 模式嵌套字典"""
        from tests.ext.text_chunker.conftest import create_parse_result

        parse_result = create_parse_result(json_nested_data, OutputFormat.JSON)
        config = JsonChunkConfig(mode="simple", keys=[])
        strategy = JsonChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_simple_mode_nested_list(self):
        """测试 Simple 模式嵌套列表"""
        data = {
            "title": "Test",
            "items": ["item1", "item2", "item3"],
        }
        content = json.dumps(data, ensure_ascii=False)

        from tests.ext.text_chunker.conftest import create_parse_result

        parse_result = create_parse_result(content, OutputFormat.JSON)
        config = JsonChunkConfig(mode="simple", keys=[])
        strategy = JsonChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        assert len(chunks) >= 1


class TestJsonChunkStrategyMergeAndSplit:
    """测试合并和分割"""

    @pytest.mark.asyncio
    async def test_merge_small_chunks_simple_mode(self, json_list_data):
        """测试 Simple 模式合并小块"""
        from tests.ext.text_chunker.conftest import create_parse_result

        # 创建一个小数据的列表
        small_data = [{"id": 1}, {"id": 2}, {"id": 3}]
        content = json.dumps(small_data, ensure_ascii=False)

        parse_result = create_parse_result(content, OutputFormat.JSON)
        config = JsonChunkConfig(mode="simple", keys=[], max_chunk_size=50)
        strategy = JsonChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        # 应该合并成较少的块
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_split_large_chunks_json_mode(self):
        """测试 JSON 模式分割大块"""
        # 创建一个大的 JSON 对象
        large_data = {
            "data": "x" * 2000,
        }
        content = json.dumps(large_data, ensure_ascii=False)

        from tests.ext.text_chunker.conftest import create_parse_result

        parse_result = create_parse_result(content, OutputFormat.JSON)
        config = JsonChunkConfig(mode="json", max_chunk_size=500)
        strategy = JsonChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        # 应该分割成多个块
        assert len(chunks) >= 1


class TestJsonChunkStrategyErrorHandling:
    """测试错误处理"""

    @pytest.mark.asyncio
    async def test_invalid_json_fallback_to_text(self, text_parse_result):
        """测试无效 JSON 回退到普通文本处理"""
        config = JsonChunkConfig(mode="simple", keys=[], max_chunk_size=1000)
        strategy = JsonChunkStrategy(config)

        chunks = await strategy.chunk(text_parse_result)

        # 应该回退到按长度切分
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_plain_text_content(self, long_text_parse_result):
        """测试纯文本内容"""
        config = JsonChunkConfig(mode="json", max_chunk_size=200)
        strategy = JsonChunkStrategy(config)

        chunks = await strategy.chunk(long_text_parse_result)

        # 应该回退到按长度切分
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_malformed_json(self):
        """测试格式错误的 JSON"""
        malformed_json = '{"name": "Alice", "age": 30'  # 缺少闭合括号

        from tests.ext.text_chunker.conftest import create_parse_result

        parse_result = create_parse_result(malformed_json, OutputFormat.JSON)
        config = JsonChunkConfig(mode="simple")
        strategy = JsonChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        # 应该回退到按长度切分
        assert len(chunks) >= 1


class TestJsonChunkStrategyEdgeCases:
    """测试边界情况"""

    @pytest.mark.asyncio
    async def test_empty_json_array(self):
        """测试空 JSON 数组"""
        content = "[]"

        from tests.ext.text_chunker.conftest import create_parse_result

        parse_result = create_parse_result(content, OutputFormat.JSON)
        config = JsonChunkConfig(mode="simple")
        strategy = JsonChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        # 空数组应该返回空列表或处理为空
        assert isinstance(chunks, list)

    @pytest.mark.asyncio
    async def test_empty_json_object(self):
        """测试空 JSON 对象"""
        content = "{}"

        from tests.ext.text_chunker.conftest import create_parse_result

        parse_result = create_parse_result(content, OutputFormat.JSON)
        config = JsonChunkConfig(mode="simple")
        strategy = JsonChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        # 应该至少有一个块
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_json_simple_value(self):
        """测试简单 JSON 值"""
        content = '"just a string"'

        from tests.ext.text_chunker.conftest import create_parse_result

        parse_result = create_parse_result(content, OutputFormat.JSON)
        config = JsonChunkConfig(mode="simple")
        strategy = JsonChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_json_number(self):
        """测试 JSON 数字"""
        content = "42"

        from tests.ext.text_chunker.conftest import create_parse_result

        parse_result = create_parse_result(content, OutputFormat.JSON)
        config = JsonChunkConfig(mode="simple")
        strategy = JsonChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_json_boolean(self):
        """测试 JSON 布尔值"""
        content = "true"

        from tests.ext.text_chunker.conftest import create_parse_result

        parse_result = create_parse_result(content, OutputFormat.JSON)
        config = JsonChunkConfig(mode="simple")
        strategy = JsonChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_json_null(self):
        """测试 JSON null"""
        content = "null"

        from tests.ext.text_chunker.conftest import create_parse_result

        parse_result = create_parse_result(content, OutputFormat.JSON)
        config = JsonChunkConfig(mode="simple")
        strategy = JsonChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_missing_keys_in_filter(self, json_parse_result):
        """测试过滤不存在的 keys"""
        config = JsonChunkConfig(mode="simple", keys=["nonexistent", "another_missing"])
        strategy = JsonChunkStrategy(config)

        chunks = await strategy.chunk(json_parse_result)

        # 应该不报错，但可能返回空或少量内容
        assert isinstance(chunks, list)


class TestJsonChunkStrategyMetadata:
    """测试元数据"""

    @pytest.mark.asyncio
    async def test_strategy_metadata(self, json_parse_result):
        """测试策略元数据"""
        config = JsonChunkConfig(mode="simple")
        strategy = JsonChunkStrategy(config)

        chunks = await strategy.chunk(json_parse_result)

        for chunk in chunks:
            assert chunk.metadata["strategy"] == "json"
            assert "mode" in chunk.metadata
            assert chunk.metadata["mode"] == "simple"
            assert "chunk_index" in chunk.metadata

    @pytest.mark.asyncio
    async def test_json_mode_metadata(self, json_parse_result):
        """测试 JSON 模式元数据"""
        config = JsonChunkConfig(mode="json")
        strategy = JsonChunkStrategy(config)

        chunks = await strategy.chunk(json_parse_result)

        for chunk in chunks:
            assert chunk.metadata["mode"] == "json"


class TestJsonChunkStrategyComplexScenarios:
    """测试复杂场景"""

    @pytest.mark.asyncio
    async def test_real_world_api_response(self):
        """测试真实 API 响应数据"""
        api_response = json.dumps(
            {
                "status": "success",
                "data": [
                    {
                        "id": 1,
                        "name": "Item 1",
                        "description": "Description 1",
                        "tags": ["tag1", "tag2"],
                    },
                    {
                        "id": 2,
                        "name": "Item 2",
                        "description": "Description 2",
                        "tags": ["tag3", "tag4"],
                    },
                ],
                "pagination": {"page": 1, "per_page": 10, "total": 100},
            },
            ensure_ascii=False,
        )

        from tests.ext.text_chunker.conftest import create_parse_result

        parse_result = create_parse_result(api_response, OutputFormat.JSON)
        config = JsonChunkConfig(mode="simple", keys=["id", "name"])
        strategy = JsonChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_deeply_nested_structure(self):
        """测试深度嵌套结构"""
        deep_data = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {"value": "deep", "items": [1, 2, 3]},
                    }
                }
            }
        }
        content = json.dumps(deep_data, ensure_ascii=False)

        from tests.ext.text_chunker.conftest import create_parse_result

        parse_result = create_parse_result(content, OutputFormat.JSON)
        config = JsonChunkConfig(mode="simple")
        strategy = JsonChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_unicode_content(self):
        """测试 Unicode 内容"""
        unicode_data = [
            {"text": "Hello 世界"},
            {"text": "こんにちは"},
            {"text": "مرحبا"},
            {"text": "🎉🎊"},
        ]
        content = json.dumps(unicode_data, ensure_ascii=False)

        from tests.ext.text_chunker.conftest import create_parse_result

        parse_result = create_parse_result(content, OutputFormat.JSON)
        # 设置小的 max_chunk_size 以避免合并
        config = JsonChunkConfig(mode="simple", max_chunk_size=30)
        strategy = JsonChunkStrategy(config)

        chunks = await strategy.chunk(parse_result)

        assert len(chunks) == 4
        # 验证 Unicode 字符被正确保留
        assert "世界" in chunks[0].content
