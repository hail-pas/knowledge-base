from typing import Any

from pydantic import Field, BaseModel


class TextPosition(BaseModel):
    """文本位置坐标

    表示文档中某个具体位置的页码和页内偏移
    """

    page_number: int = Field(description="页码（从1开始）")
    char_index: int = Field(description="该页内的字符索引（从0开始）")


class ChunkOverlap(BaseModel):
    """重叠区域信息

    用于记录chunk之间重叠内容的位置信息
    """

    start: TextPosition = Field(description="重叠区域起始位置")
    end: TextPosition = Field(description="重叠区域结束位置")


class ChunkResult(BaseModel):
    """单个切块结果

    包含切块内容及其在文档中的位置信息
    """

    content: str = Field(description="切块内容")
    pages: list[int] = Field(description="所跨页码列表，如 [1, 2, 3]")
    start: TextPosition = Field(description="起始位置（页码+页内偏移）")
    end: TextPosition = Field(description="结束位置（页码+页内偏移）")
    overlap_start: TextPosition | None = Field(default=None, description="overlap区域起始位置（如果有）")
    overlap_end: TextPosition | None = Field(default=None, description="overlap区域结束位置（如果有）")
    metadata: dict[str, Any] = Field(default_factory=dict, description="额外的元数据信息")
