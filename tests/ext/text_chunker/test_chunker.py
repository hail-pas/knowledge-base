"""
测试 TextChunker 主类

测试核心功能：
- 初始化和策略列表
- 配置 schema 获取
- 预设管理
- 自动策略选择
- 手动策略选择
- 无效输入处理
"""

import pytest

from ext.document_parser.core.parse_result import OutputFormat
from ext.text_chunker import TextChunker


class TestTextChunkerInitialization:
    """测试 TextChunker 初始化"""

    def test_initialization(self, text_chunker):
        """测试初始化成功"""
        assert text_chunker is not None
        assert isinstance(text_chunker, TextChunker)

    def test_list_strategies(self, text_chunker):
        """测试列出所有策略"""
        strategies = text_chunker.list_strategies()

        assert isinstance(strategies, list)
        assert len(strategies) == 4

        expected_strategies = ["length", "delimiter", "heading", "json"]
        assert set(strategies) == set(expected_strategies)

    def test_get_strategy_config_schema_valid(self, text_chunker):
        """测试获取有效策略的配置 schema"""
        for strategy in ["length", "delimiter", "heading", "json"]:
            schema = text_chunker.get_strategy_config_schema(strategy)

            assert schema is not None
            assert isinstance(schema, dict)
            assert "properties" in schema

    def test_get_strategy_config_schema_invalid(self, text_chunker):
        """测试获取无效策略的配置 schema"""
        schema = text_chunker.get_strategy_config_schema("invalid_strategy")

        assert schema is None


class TestTextChunkerPresets:
    """测试预设管理"""

    def test_list_presets(self, text_chunker):
        """测试列出所有预设"""
        presets = text_chunker.list_presets()

        assert isinstance(presets, dict)
        assert len(presets) > 0

        assert OutputFormat.TEXT in presets
        assert OutputFormat.MARKDOWN in presets
        assert OutputFormat.JSON in presets

        for format_type, preset in presets.items():
            assert "strategy" in preset
            assert "config" in preset

    def test_get_preset_strategy_text(self, text_chunker):
        """测试获取 TEXT 格式的预设策略"""
        preset = text_chunker.get_preset_strategy(OutputFormat.TEXT)

        assert preset["strategy"] == "length"
        assert "chunk_size" in preset["config"]
        assert "overlap" in preset["config"]

    def test_get_preset_strategy_markdown(self, text_chunker):
        """测试获取 MARKDOWN 格式的预设策略"""
        preset = text_chunker.get_preset_strategy(OutputFormat.MARKDOWN)

        assert preset["strategy"] == "heading"
        assert "max_chunk_size" in preset["config"]
        assert "preserve_headings" in preset["config"]

    def test_get_preset_strategy_json(self, text_chunker):
        """测试获取 JSON 格式的预设策略"""
        preset = text_chunker.get_preset_strategy(OutputFormat.JSON)

        assert preset["strategy"] == "json"
        assert "mode" in preset["config"]

    def test_validate_strategy_for_format_recommended(self, text_chunker):
        """测试推荐策略验证"""
        is_recommended, msg = text_chunker.validate_strategy_for_format("heading", OutputFormat.MARKDOWN)

        assert is_recommended is True
        assert "recommended" in msg.lower()

    def test_validate_strategy_for_format_compatible(self, text_chunker):
        """测试兼容策略验证"""
        is_compatible, msg = text_chunker.validate_strategy_for_format("length", OutputFormat.MARKDOWN)

        assert is_compatible is True
        assert "compatible" in msg.lower() or "recommended" in msg.lower()

    def test_validate_strategy_for_format_incompatible(self, text_chunker):
        """测试不兼容策略验证"""
        is_compatible, msg = text_chunker.validate_strategy_for_format("json", OutputFormat.MARKDOWN)

        assert is_compatible is False
        assert "warning" in msg.lower() or "not work well" in msg.lower()


class TestTextChunkerAutoStrategy:
    """测试自动策略选择"""

    @pytest.mark.asyncio
    async def test_auto_strategy_text(self, text_chunker, text_parse_result):
        """测试 TEXT 格式自动选择 length 策略"""
        chunks = await text_chunker.chunk(text_parse_result, strategy="auto")

        assert chunks is not None
        assert isinstance(chunks, list)

    @pytest.mark.asyncio
    async def test_auto_strategy_markdown(self, text_chunker, markdown_parse_result):
        """测试 MARKDOWN 格式自动选择 heading 策略"""
        chunks = await text_chunker.chunk(markdown_parse_result, strategy="auto")

        assert chunks is not None
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_auto_strategy_json(self, text_chunker, json_parse_result):
        """测试 JSON 格式自动选择 json 策略"""
        chunks = await text_chunker.chunk(json_parse_result, strategy="auto")

        assert chunks is not None
        assert isinstance(chunks, list)

    @pytest.mark.asyncio
    async def test_auto_strategy_uses_preset_config(self, text_chunker, markdown_parse_result):
        """测试自动策略使用预设配置"""
        chunks = await text_chunker.chunk(markdown_parse_result, strategy="auto")

        assert chunks is not None
        assert len(chunks) > 0


class TestTextChunkerManualStrategy:
    """测试手动策略选择"""

    @pytest.mark.asyncio
    async def test_manual_strategy_length(self, text_chunker, text_parse_result):
        """测试手动指定 length 策略"""
        chunks = await text_chunker.chunk(text_parse_result, strategy="length", config={"chunk_size": 10})

        assert chunks is not None
        assert isinstance(chunks, list)

    @pytest.mark.asyncio
    async def test_manual_strategy_delimiter(self, text_chunker, text_parse_result):
        """测试手动指定 delimiter 策略"""
        chunks = await text_chunker.chunk(text_parse_result, strategy="delimiter", config={"delimiters": [" "]})

        assert chunks is not None
        assert isinstance(chunks, list)

    @pytest.mark.asyncio
    async def test_manual_strategy_heading(self, text_chunker, markdown_parse_result):
        """测试手动指定 heading 策略"""
        chunks = await text_chunker.chunk(markdown_parse_result, strategy="heading")

        assert chunks is not None
        assert isinstance(chunks, list)

    @pytest.mark.asyncio
    async def test_manual_strategy_json(self, text_chunker, json_parse_result):
        """测试手动指定 json 策略"""
        chunks = await text_chunker.chunk(json_parse_result, strategy="json")

        assert chunks is not None
        assert isinstance(chunks, list)

    @pytest.mark.asyncio
    async def test_manual_strategy_with_custom_config(self, text_chunker, long_text_parse_result):
        """测试自定义配置"""
        chunks = await text_chunker.chunk(
            long_text_parse_result, strategy="length", config={"chunk_size": 100, "overlap": 20}
        )

        assert chunks is not None
        assert len(chunks) > 0

        for chunk in chunks:
            assert len(chunk.content) <= 100


class TestTextChunkerOverrideRecommended:
    """测试覆盖推荐策略"""

    @pytest.mark.asyncio
    async def test_override_markdown_with_length(self, text_chunker, markdown_parse_result):
        """测试在 Markdown 上使用 length 策略（不推荐但兼容）"""
        chunks = await text_chunker.chunk(markdown_parse_result, strategy="length", config={"chunk_size": 100})

        assert chunks is not None
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_override_text_with_delimiter(self, text_chunker, text_with_delimiters, create_parse_result):
        """测试在 TEXT 上使用 delimiter 策略"""

        parse_result = create_parse_result(text_with_delimiters, OutputFormat.TEXT)
        chunks = await text_chunker.chunk(parse_result, strategy="delimiter", config={"delimiters": ["\n\n"]})

        assert chunks is not None
        assert len(chunks) > 0


class TestTextChunkerErrorHandling:
    """测试错误处理"""

    @pytest.mark.asyncio
    async def test_invalid_strategy(self, text_chunker, text_parse_result):
        """测试无效策略名称"""
        with pytest.raises(ValueError, match="Unsupported strategy"):
            await text_chunker.chunk(text_parse_result, strategy="invalid_strategy")

    @pytest.mark.asyncio
    async def test_invalid_config_type(self, text_chunker, text_parse_result):
        """测试无效的配置类型"""
        with pytest.raises(ValueError, match="Invalid config"):
            await text_chunker.chunk(text_parse_result, strategy="length", config={"chunk_size": "invalid"})

    @pytest.mark.asyncio
    async def test_invalid_config_value(self, text_chunker, text_parse_result):
        """测试无效的配置值"""
        with pytest.raises(ValueError, match="Invalid config"):
            await text_chunker.chunk(text_parse_result, strategy="length", config={"chunk_size": -1})

    @pytest.mark.asyncio
    async def test_empty_content(self, text_chunker, empty_parse_result):
        """测试空内容"""
        chunks = await text_chunker.chunk(empty_parse_result, strategy="length")

        assert chunks is not None
        assert isinstance(chunks, list)


class TestTextChunkerConfigMerging:
    """测试配置合并"""

    @pytest.mark.asyncio
    async def test_auto_with_user_config_override(self, text_chunker, markdown_parse_result):
        """测试自动策略 + 用户配置覆盖"""
        chunks = await text_chunker.chunk(
            markdown_parse_result,
            strategy="auto",
            config={"max_chunk_size": 500},
        )

        assert chunks is not None
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_no_config_uses_defaults(self, text_chunker, text_parse_result):
        """测试不提供配置时使用默认值"""
        chunks = await text_chunker.chunk(text_parse_result, strategy="length")

        assert chunks is not None
        assert isinstance(chunks, list)


class TestTextChunkerChunkResult:
    """测试切块结果结构"""

    @pytest.mark.asyncio
    async def test_chunk_result_structure(self, text_chunker, long_text_parse_result):
        """测试切块结果结构"""
        chunks = await text_chunker.chunk(long_text_parse_result, strategy="length", config={"chunk_size": 100})

        assert len(chunks) > 0

        for chunk in chunks:
            assert hasattr(chunk, "content")
            assert hasattr(chunk, "pages")
            assert hasattr(chunk, "start")
            assert hasattr(chunk, "end")
            assert hasattr(chunk, "metadata")

            assert isinstance(chunk.content, str)
            assert isinstance(chunk.pages, list)
            assert len(chunk.pages) > 0

    @pytest.mark.asyncio
    async def test_chunk_result_metadata(self, text_chunker, long_text_parse_result):
        """测试切块结果元数据"""
        chunks = await text_chunker.chunk(long_text_parse_result, strategy="length")

        for chunk in chunks:
            assert "chunk_index" in chunk.metadata
            assert "strategy" in chunk.metadata
