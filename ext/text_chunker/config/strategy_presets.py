"""
策略预设配置

为不同文档格式定义推荐的切块策略和配置
"""

from ext.document_parser.core.parse_result import OutputFormat
from ext.text_chunker.config.strategy_config import (
    DelimiterChunkConfig,
    HeadingChunkConfig,
    JsonChunkConfig,
    LengthChunkConfig,
)


class StrategyPreset:
    """策略预设"""

    def __init__(self, strategy: str, config: dict | None = None):
        """
        初始化预设

        Args:
            strategy: 策略名称
            config: 策略配置参数
        """
        self.strategy = strategy
        self.config = config or {}


# 文档格式到策略预设的映射
STRATEGY_PRESETS: dict[OutputFormat, StrategyPreset] = {
    OutputFormat.MARKDOWN: StrategyPreset(
        strategy="heading",
        config={
            "max_chunk_size": 2000,
            "overlap_paragraphs": 1,
            "preserve_headings": True,
            "min_paragraph_size": 100,
            "heading_patterns": ["^#{1,6}\\s", "^第[一二三四五六七八九十]+[章节篇]"],
        },
    ),
    OutputFormat.TEXT: StrategyPreset(
        strategy="length",
        config={
            "chunk_size": 1000,
            "overlap": 200,
            "mode": "chars",
        },
    ),
    OutputFormat.JSON: StrategyPreset(
        strategy="json",
        config={
            "mode": "simple",
            "keys": [],
            "max_chunk_size": 1000,
        },
    ),
}


def get_preset(format: OutputFormat | str) -> StrategyPreset:
    """
    获取指定格式的策略预设

    Args:
        format: 文档格式 (OutputFormat 枚举或字符串)

    Returns:
        策略预设对象，如果格式未定义则返回默认预设
    """
    # 处理字符串格式的输入
    if isinstance(format, str):
        # 将字符串转换为 OutputFormat 枚举
        format_map = {
            "text": OutputFormat.TEXT,
            "markdown": OutputFormat.MARKDOWN,
            "json": OutputFormat.JSON,
        }
        format = format_map.get(format.lower(), OutputFormat.TEXT)

    return STRATEGY_PRESETS.get(format, STRATEGY_PRESETS[OutputFormat.TEXT])


def list_presets() -> dict[OutputFormat, dict]:
    """
    列出所有可用的策略预设

    Returns:
        格式到预设配置的映射字典
    """
    return {
        format_type: {
            "strategy": preset.strategy,
            "config": preset.config,
        }
        for format_type, preset in STRATEGY_PRESETS.items()
    }


def validate_strategy_for_format(strategy: str, format: OutputFormat | str) -> tuple[bool, str]:
    """
    验证策略是否适合指定的文档格式

    Args:
        strategy: 策略名称
        format: 文档格式 (OutputFormat 枚举或字符串)

    Returns:
        (is_recommended, message) 推荐状态和提示信息
    """
    # 处理字符串格式的输入
    if isinstance(format, str):
        # 将字符串转换为 OutputFormat 枚举
        format_map = {
            "text": OutputFormat.TEXT,
            "markdown": OutputFormat.MARKDOWN,
            "json": OutputFormat.JSON,
        }
        format = format_map.get(format.lower(), OutputFormat.TEXT)
        format_str = format.value
    else:
        format_str = format.value

    preset = get_preset(format)

    if strategy == preset.strategy:
        return True, f"Strategy '{strategy}' is recommended for {format_str} format"

    # 定义策略与格式的兼容性矩阵
    compatibility = {
        (OutputFormat.MARKDOWN, "heading"): True,
        (OutputFormat.MARKDOWN, "length"): True,
        (OutputFormat.MARKDOWN, "delimiter"): True,
        (OutputFormat.MARKDOWN, "json"): False,
        (OutputFormat.TEXT, "length"): True,
        (OutputFormat.TEXT, "delimiter"): True,
        (OutputFormat.TEXT, "heading"): True,
        (OutputFormat.TEXT, "json"): False,
        (OutputFormat.JSON, "json"): True,
        (OutputFormat.JSON, "length"): True,
        (OutputFormat.JSON, "delimiter"): True,
        (OutputFormat.JSON, "heading"): False,
    }

    is_compatible = compatibility.get((format, strategy), False)

    if is_compatible:
        return (
            True,
            f"Strategy '{strategy}' is compatible with {format_str} format (though '{preset.strategy}' is recommended)",
        )
    else:
        return (
            False,
            f"Warning: Strategy '{strategy}' may not work well with {format_str} format. Recommended: '{preset.strategy}'",
        )
