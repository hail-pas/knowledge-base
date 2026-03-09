"""
文本切块器主类

提供统一的文本切块接口
"""

from loguru import logger
from typing import Any

from ext.document_parser.core.parse_result import OutputFormat, ParseResult
from ext.text_chunker.config.strategy_config import (
    DelimiterChunkConfig,
    HeadingChunkConfig,
    JsonChunkConfig,
    LengthChunkConfig,
)
from ext.text_chunker.config.strategy_presets import (
    STRATEGY_PRESETS,
    get_preset,
    list_presets,
    validate_strategy_for_format,
)
from ext.text_chunker.core.chunk_result import ChunkResult
from ext.text_chunker.strategies.delimiter_based import DelimiterChunkStrategy
from ext.text_chunker.strategies.heading_based import HeadingChunkStrategy
from ext.text_chunker.strategies.json_based import JsonChunkStrategy
from ext.text_chunker.strategies.length_based import LengthChunkStrategy


class TextChunker:
    """文本切块器

    提供多种文本切块策略，支持：
    - auto: 自动选择策略（根据文档格式）
    - length: 按长度切块（字符或token）
    - delimiter: 按分隔符切块
    - heading: 按标题层级切块
    - json: JSON数据切块
    """

    def __init__(self):
        """初始化切块器"""
        self._strategy_map = {
            "length": (LengthChunkStrategy, LengthChunkConfig),
            "delimiter": (DelimiterChunkStrategy, DelimiterChunkConfig),
            "heading": (HeadingChunkStrategy, HeadingChunkConfig),
            "json": (JsonChunkStrategy, JsonChunkConfig),
        }
        self._presets = STRATEGY_PRESETS

    async def chunk(
        self,
        parse_result: ParseResult,
        strategy: str = "auto",
        config: dict[str, Any] | None = None,
    ) -> list[ChunkResult]:
        """
        执行文本切块

        Args:
            parse_result: 文档解析结果
            strategy: 切块策略名称 ("auto", "length", "delimiter", "heading", "json")
                      默认为 "auto"，根据文档格式自动选择
            config: 策略配置参数（可选，使用预设配置或默认值如果未提供）

        Returns:
            切块结果列表

        Raises:
            ValueError: 不支持的切块策略或配置参数错误

        Example:
            >>> chunker = TextChunker()
            >>> # 使用自动策略（推荐）
            >>> chunks = await chunker.chunk(parse_result)
            >>> # 显式指定策略
            >>> chunks = await chunker.chunk(parse_result, strategy="length")
            >>> # 自定义配置
            >>> chunks = await chunker.chunk(
            ...     parse_result,
            ...     strategy="length",
            ...     config={"chunk_size": 500, "overlap": 100}
            ... )
            >>> # Markdown文档使用非推荐策略
            >>> chunks = await chunker.chunk(parse_result, strategy="delimiter")
        """
        # 处理 auto 策略
        if strategy == "auto":
            preset = get_preset(parse_result.format)
            actual_strategy = preset.strategy
            # 如果没有提供config，使用预设配置
            if config is None:
                config = preset.config.copy()
            # 获取格式字符串（处理枚举和字符串两种情况）
            format_str = parse_result.format if isinstance(parse_result.format, str) else parse_result.format.value
            logger.info(f"Auto-selected strategy '{actual_strategy}' for {format_str} format")
        else:
            actual_strategy = strategy

        # 验证策略与格式的兼容性
        is_recommended, validation_msg = validate_strategy_for_format(actual_strategy, parse_result.format)
        if not is_recommended:
            logger.warning(validation_msg)
        else:
            logger.debug(validation_msg)

        # 检查策略是否支持
        if actual_strategy not in self._strategy_map:
            raise ValueError(
                f"Unsupported strategy: {actual_strategy}. Available strategies: {list(self._strategy_map.keys())}"
            )

        strategy_class, config_class = self._strategy_map[actual_strategy]

        # 解析配置
        if config is None:
            config_obj = config_class()
        else:
            try:
                config_obj = config_class(**config)
            except Exception as e:
                raise ValueError(f"Invalid config for strategy '{actual_strategy}': {e}")

        # 创建策略实例并执行切块
        logger.info(f"Starting chunking with strategy: {actual_strategy}, config: {config_obj.model_dump()}")
        strategy_instance = strategy_class(config_obj)
        chunks = await strategy_instance.chunk(parse_result)

        logger.info(f"Chunking completed: {len(chunks)} chunks generated")
        return chunks

    def list_strategies(self) -> list[str]:
        """
        列出所有可用的切块策略

        Returns:
            策略名称列表
        """
        return list(self._strategy_map.keys())

    def get_strategy_config_schema(self, strategy: str) -> dict[str, Any] | None:
        """
        获取指定策略的配置schema

        Args:
            strategy: 策略名称

        Returns:
            配置schema字典，如果策略不存在则返回None
        """
        if strategy not in self._strategy_map:
            return None

        _, config_class = self._strategy_map[strategy]
        return config_class.model_json_schema()

    def get_preset_strategy(self, format: OutputFormat) -> dict[str, Any]:
        """
        获取指定格式的预设策略配置

        Args:
            format: 文档格式

        Returns:
            包含 strategy 和 config 的字典
        """
        preset = get_preset(format)
        return {"strategy": preset.strategy, "config": preset.config}

    def list_presets(self) -> dict[OutputFormat, dict[str, Any]]:
        """
        列出所有可用的策略预设

        Returns:
            格式到预设配置的映射字典
        """
        return list_presets()

    def validate_strategy_for_format(self, strategy: str, format: OutputFormat) -> tuple[bool, str]:
        """
        验证策略是否适合指定的文档格式

        Args:
            strategy: 策略名称
            format: 文档格式

        Returns:
            (is_recommended, message) 推荐状态和提示信息
        """
        return validate_strategy_for_format(strategy, format)
