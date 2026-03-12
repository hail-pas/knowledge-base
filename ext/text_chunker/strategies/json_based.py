"""
JSON数据切块策略

用于处理结构化的JSON数据
"""

import json
from typing import Any

from loguru import logger

from ext.text_chunker.strategies.base import BaseChunkStrategy
from ext.document_parser.core.parse_result import ParseResult
from ext.text_chunker.config.strategy_config import JsonChunkConfig


class JsonChunkStrategy(BaseChunkStrategy[JsonChunkConfig]):
    """JSON数据切块策略

    支持两种模式：
    1. simple: 将JSON转换为键值对文本
    2. json: 保持JSON格式
    """

    async def chunk(self, parse_result: ParseResult) -> list:
        """
        执行JSON数据切块

        Args:
            parse_result: 文档解析结果

        Returns:
            切块结果列表
        """
        from ext.text_chunker.core.coordinate_mapper import CoordinateMapper

        self._set_mapper(CoordinateMapper(parse_result))

        mode = self.config.mode
        text = parse_result.content

        # 尝试解析JSON
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            # 如果不是有效JSON，按普通文本处理
            return await self._chunk_as_text(parse_result)

        # 根据模式处理
        if mode == "simple":
            chunks_text = self._process_simple_mode(data)
        elif mode == "json":
            chunks_text = self._process_json_mode(data)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # 构建ChunkResult列表
        results = []
        global_pos = 0
        chunk_index = 0
        content_length = len(parse_result.content)

        for chunk_text in chunks_text:
            if not chunk_text:
                continue

            start = global_pos
            end = global_pos + len(chunk_text)
            global_pos = end

            # 确保位置在有效范围内（对于JSON转换，长度可能不同）
            actual_end = min(end, content_length)
            if start >= content_length:
                # 如果起始位置超出范围，重置为0
                start = 0
                actual_end = min(len(chunk_text), content_length)

            chunk = self._build_chunk(
                content=chunk_text,
                global_start=start,
                global_end=actual_end,
                metadata={"chunk_index": chunk_index, "strategy": "json", "mode": mode},
            )
            results.append(chunk)
            chunk_index += 1

        logger.info(f"Chunked JSON into {len(results)} chunks using mode: {mode}")
        return results

    def _process_simple_mode(self, data: Any) -> list[str]:
        """
        简单模式：将JSON转换为键值对文本

        Args:
            data: JSON数据

        Returns:
            文本切块列表
        """
        chunks = []

        if isinstance(data, list):
            # list[dict] 的情况
            for idx, item in enumerate(data):
                if isinstance(item, dict):
                    text = self._dict_to_text(item, f"Item {idx + 1}")
                    chunks.append(text)
                else:
                    # 简单值
                    text = f"Item {idx + 1}: {str(item)}"
                    chunks.append(text)
        elif isinstance(data, dict):
            # dict 的情况
            text = self._dict_to_text(data, "Root")
            chunks.append(text)
        else:
            # 简单值
            text = str(data)
            chunks.append(text)

        # 检查是否需要合并小块
        if self.config.max_chunk_size > 0:
            chunks = self._merge_small_chunks(chunks)

        return chunks

    def _dict_to_text(self, data: dict, prefix: str = "") -> str:
        """
        将字典转换为键值对文本

        Args:
            data: 字典数据
            prefix: 前缀（如 "Item 1"）

        Returns:
            格式化文本
        """
        lines = []

        if prefix:
            lines.append(f"{prefix}\n")

        # 过滤keys
        keys = self.config.keys if self.config.keys else data.keys()

        for key in keys:
            if key not in data:
                continue

            value = data[key]

            if isinstance(value, dict):
                # 嵌套字典，递归处理
                nested_prefix = f"{prefix}{self.config.key_separator}{key}" if prefix else key
                nested_text = self._dict_to_text(value, nested_prefix)
                lines.append(nested_text)
            elif isinstance(value, list):
                # 列表值
                str_value = self.config.item_joiner.join(str(v) for v in value)
                lines.append(f"{key}{self.config.value_separator}{str_value}")
            else:
                # 简单值
                lines.append(f"{key}{self.config.value_separator}{value}")

        return self.config.item_joiner.join(lines)

    def _process_json_mode(self, data: Any) -> list[str]:
        """
        JSON模式：保持JSON格式

        Args:
            data: JSON数据

        Returns:
            JSON字符串切块列表
        """
        chunks = []

        if isinstance(data, list):
            # list[dict] 的情况
            for item in data:
                json_str = json.dumps(item, ensure_ascii=False, indent=2)
                chunks.append(json_str)
        elif isinstance(data, dict):
            # dict 的情况，按keys分组
            keys = self.config.keys if self.config.keys else list(data.keys())

            # 如果配置了keys，只处理指定的keys
            if self.config.keys:
                filtered_data = {k: data[k] for k in keys if k in data}
                json_str = json.dumps(filtered_data, ensure_ascii=False, indent=2)
                chunks.append(json_str)
            else:
                # 没有指定keys，整个dict作为一个chunk
                json_str = json.dumps(data, ensure_ascii=False, indent=2)
                chunks.append(json_str)
        else:
            # 简单值
            json_str = json.dumps(data, ensure_ascii=False)
            chunks.append(json_str)

        # 检查是否需要分割超长的JSON
        if self.config.max_chunk_size > 0:
            chunks = self._split_large_chunks(chunks)

        return chunks

    def _merge_small_chunks(self, chunks: list[str]) -> list[str]:
        """
        合并过小的切块

        Args:
            chunks: 原始切块列表

        Returns:
            合并后的切块列表
        """
        if not chunks:
            return []

        merged = []
        current_chunk = chunks[0]

        for chunk in chunks[1:]:
            test_chunk = current_chunk + self.config.item_joiner + chunk

            if len(test_chunk) <= self.config.max_chunk_size:
                current_chunk = test_chunk
            else:
                merged.append(current_chunk)
                current_chunk = chunk

        merged.append(current_chunk)

        return merged

    def _split_large_chunks(self, chunks: list[str]) -> list[str]:
        """
        分割过大的切块

        Args:
            chunks: 原始切块列表

        Returns:
            分割后的切块列表
        """
        result = []

        for chunk in chunks:
            if len(chunk) <= self.config.max_chunk_size:
                result.append(chunk)
            else:
                # 尝试按行分割
                lines = chunk.split("\n")
                current_chunk = ""

                for line in lines:
                    test_chunk = current_chunk + ("\n" if current_chunk else "") + line

                    if len(test_chunk) <= self.config.max_chunk_size:
                        current_chunk = test_chunk
                    else:
                        if current_chunk:
                            result.append(current_chunk)
                        current_chunk = line

                if current_chunk:
                    result.append(current_chunk)

        return result

    async def _chunk_as_text(self, parse_result: ParseResult) -> list:
        """
        将非JSON内容作为普通文本处理

        Args:
            parse_result: 文档解析结果

        Returns:
            切块结果列表
        """
        from ext.text_chunker.config.strategy_config import LengthChunkConfig
        from ext.text_chunker.strategies.length_based import LengthChunkStrategy

        logger.info("Content is not valid JSON, treating as plain text")

        config = LengthChunkConfig(chunk_size=self.config.max_chunk_size, overlap=0)
        strategy = LengthChunkStrategy(config)
        return await strategy.chunk(parse_result)
