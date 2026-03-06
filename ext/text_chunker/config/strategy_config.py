"""
切块策略配置模型

定义各种切块策略的配置参数
"""
from typing import Literal
from pydantic import BaseModel, Field


class LengthChunkConfig(BaseModel):
    """按长度切块配置

    支持按字符数或token数进行切块
    """

    chunk_size: int = Field(default=1000, ge=1, description="切块大小（字符或token数）")
    overlap: int = Field(default=200, ge=0, description="重叠字符数")
    mode: str = Field(default="chars", description="计数模式: 'chars' 或 'tokens'")
    encoding: str = Field(default="cl100k_base", description="token编码名称（仅当mode='tokens'时有效）")


class HeadingChunkConfig(BaseModel):
    """按标题层级切块配置

    按照文档标题层级进行切块，保留父级标题
    """

    max_chunk_size: int = Field(default=2000, ge=1, description="单个切块最大字符数")
    overlap_paragraphs: int = Field(default=1, ge=0, description="重叠段落数")
    preserve_headings: bool = Field(default=True, description="是否保留所有父级标题")
    min_paragraph_size: int = Field(default=100, ge=0, description="最小段落大小")
    heading_patterns: list[str] = Field(
        default=["^#{1,6}\\s", "^第[一二三四五六七八九十]+[章节篇]"],
        description="标题识别正则模式列表",
    )


class DelimiterChunkConfig(BaseModel):
    """按分隔符切块配置

    按照自定义分隔符进行切块，支持普通字符串和正则表达式分隔符
    """

    delimiters: list[str] = Field(
        default=["\\n\\n", "\\n"], description="分隔符列表（支持前缀标记区分正则表达式，如 'regex:\\d+\\.\\s'）"
    )
    regex_prefix: str = Field(default="regex:", description="正则表达式前缀标记")
    keep_delimiter: bool = Field(default=False, description="是否在切块中保留分隔符（支持正则分隔符）")
    max_chunk_size: int = Field(default=1000, ge=1, description="单个切块最大字符数")
    overlap: int = Field(default=0, ge=0, description="重叠字符数")
    fallback_to_length: bool = Field(default=True, description="超长时是否回退到按长度切块")


class JsonChunkConfig(BaseModel):
    """JSON数据切块配置

    用于处理结构化的JSON数据
    """

    mode: Literal["simple", "json"] = Field(default="simple", description="切块模式: 'simple' (键值对) 或 'json' (保持JSON格式)")
    keys: list[str] = Field(default_factory=list, description="需要提取的key列表（空表示全部）")
    key_separator: str = Field(default=",", description="键之间分隔符")
    value_separator: str = Field(default=":", description="键值分隔符")
    item_joiner: str = Field(default="\\n", description="条目间连接符")
    max_chunk_size: int = Field(default=1000, ge=1, description="单个切块最大字符数")
