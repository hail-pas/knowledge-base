"""
切块策略基类

定义所有切块策略的通用接口和辅助方法
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from ext.text_chunker.core.chunk_result import ChunkResult
from ext.document_parser.core.parse_result import ParseResult
from ext.text_chunker.core.coordinate_mapper import CoordinateMapper

ConfigT = TypeVar("ConfigT")


class BaseChunkStrategy(ABC, Generic[ConfigT]):
    """切块策略抽象基类

    所有具体的切块策略都应继承此类并实现 chunk 方法

    Type Parameters:
        ConfigT: 配置类型
    """

    def __init__(self, config: ConfigT) -> None:
        """
        初始化策略

        Args:
            config: 策略配置
        """
        self.config = config
        self._mapper: CoordinateMapper | None = None

    def _set_mapper(self, mapper: CoordinateMapper) -> None:
        """
        设置坐标映射器

        Args:
            mapper: 坐标映射器实例
        """
        self._mapper = mapper

    @abstractmethod
    async def chunk(self, parse_result: ParseResult) -> list[ChunkResult]:
        """
        执行切块逻辑

        Args:
            parse_result: 文档解析结果

        Returns:
            切块结果列表
        """

    def _build_chunk(
        self,
        content: str,
        global_start: int,
        global_end: int,
        overlap_start: int | None = None,
        overlap_end: int | None = None,
        metadata: dict | None = None,
    ) -> ChunkResult:
        """
        构建 ChunkResult（统一入口）

        Args:
            content: 切块内容
            global_start: 全局起始索引
            global_end: 全局结束索引（不包含）
            overlap_start: overlap区域全局起始索引（可选）
            overlap_end: overlap区域全局结束索引（可选）
            metadata: 额外元数据（可选）

        Returns:
            ChunkResult 实例

        Raises:
            RuntimeError: 坐标映射器未设置
        """
        if self._mapper is None:
            raise RuntimeError("CoordinateMapper not set, call _set_mapper first")

        # 转换起始和结束位置
        start = self._mapper.global_to_page(global_start)
        # end 是不包含的，所以要减1
        end = self._mapper.global_to_page(max(global_end - 1, 0))

        # 计算所跨页码
        pages = self._mapper.get_pages_for_range(global_start, global_end)

        # 转换overlap位置（如果存在）
        overlap_start_pos = None
        overlap_end_pos = None
        if overlap_start is not None and overlap_end is not None and overlap_start < overlap_end:
            try:
                overlap_start_pos = self._mapper.global_to_page(overlap_start)
                overlap_end_pos = self._mapper.global_to_page(max(overlap_end - 1, 0))
            except ValueError:
                # overlap转换失败，忽略
                pass

        return ChunkResult(
            content=content,
            pages=pages,
            start=start,
            end=end,
            overlap_start=overlap_start_pos,
            overlap_end=overlap_end_pos,
            metadata=metadata or {},
        )
