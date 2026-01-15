"""
文件处理工作流任务

实现完整的文件处理工作流，包括：
- fetch_file: 获取文件
- load_file: 加载文件到内存
- replace_content: 替换内容
- summary: 生成摘要
- split_text: 分割文本
- index_into_milvus: 索引到 Milvus
- index_into_es: 索引到 Elasticsearch

工作流结构：
fetch_file -> load_file -> replace_content -> summary
                           -> split_text -> index_into_milvus
                                         -> index_into_es
"""

import asyncio
import hashlib
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from ext.workflow import ActivityTaskTemplate
from ext.workflow.template import activity_task


# =============================================================================
# Pydantic Models for Input/Output
# =============================================================================


class FetchFileInput(BaseModel):
    """获取文件任务的输入"""
    file_path: str = Field(description="文件路径")
    file_source_config: Dict[str, Any] = Field(
        default_factory=dict, description="文件源配置"
    )


class FetchFileOutput(BaseModel):
    """获取文件任务的输出"""
    file_content: str = Field(description="文件内容")
    file_path: str = Field(description="文件路径")
    file_size: int = Field(description="文件大小（字节）")
    file_hash: str = Field(description="文件哈希（MD5）")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    status: str = Field(description="状态")


class LoadFileInput(BaseModel):
    """加载文件任务的输入（来自 fetch_file 的输出）"""
    file_content: str = Field(description="文件内容")
    file_path: str = Field(description="文件路径")
    file_size: int = Field(description="文件大小")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class LoadFileOutput(BaseModel):
    """加载文件任务的输出"""
    file_path: str = Field(description="文件路径")
    file_content: str = Field(description="文件内容")
    file_size: int = Field(description="文件大小")
    line_count: int = Field(description="行数")
    word_count: int = Field(description="单词数")
    char_count: int = Field(description="字符数")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    status: str = Field(description="状态")


class ReplaceContentInput(BaseModel):
    """替换内容任务的输入"""
    file_content: str = Field(description="文件内容")
    file_path: str = Field(description="文件路径")
    replace_rules: List[Dict[str, str]] = Field(
        default_factory=list, description="替换规则列表"
    )


class ReplaceContentOutput(BaseModel):
    """替换内容任务的输出"""
    original_content: str = Field(description="原始内容")
    replaced_content: str = Field(description="替换后的内容")
    replace_count: int = Field(description="替换次数")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    status: str = Field(description="状态")


class SplitTextInput(BaseModel):
    """分割文本任务的输入"""
    file_content: str = Field(description="文件内容")
    chunk_size: int = Field(default=1000, description="每个分块的大小")
    chunk_overlap: int = Field(default=100, description="分块重叠大小")
    separators: List[str] = Field(
        default=["\n\n", "\n", "。", "！", "？", ".", "!", "?"], description="分隔符"
    )


class SplitTextOutput(BaseModel):
    """分割文本任务的输出"""
    chunks: List[str] = Field(description="文本分块列表")
    chunk_count: int = Field(description="分块数量")
    avg_chunk_length: int = Field(description="平均分块长度")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    status: str = Field(description="状态")


class IndexInput(BaseModel):
    """索引任务的输入"""
    chunks: List[str] = Field(description="文本分块列表")
    chunk_count: int = Field(description="分块数量")
    index_name: str = Field(description="索引名称")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class IndexOutput(BaseModel):
    """索引任务的输出"""
    ids: List[str] = Field(description="文档 ID 列表")
    indexed_count: int = Field(description="已索引数量")
    index_name: str = Field(description="索引名称")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    status: str = Field(description="状态")


class SummaryInput(BaseModel):
    """摘要生成任务的输入"""
    file_content: str = Field(description="文件内容")
    max_length: int = Field(default=200, description="摘要最大长度")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class SummaryOutput(BaseModel):
    """摘要生成任务的输出"""
    summary: str = Field(description="摘要内容")
    summary_length: int = Field(description="摘要长度")
    original_length: int = Field(description="原始内容长度")
    compression_ratio: float = Field(description="压缩比")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    status: str = Field(description="状态")


# =============================================================================
# Task Implementations
# =============================================================================


class FetchFileTask(ActivityTaskTemplate):
    """获取文件任务

    从文件源获取文件内容
    """

    async def execute(self) -> Dict[str, Any]:
        """执行文件获取逻辑"""
        # 解析输入
        input_data = FetchFileInput(**self.input)
        file_path = input_data.file_path

        logger.info(f"[{self.activity_name}] Fetching file: {file_path}")

        # 模拟异步操作
        await asyncio.sleep(1)

        # 实际文件获取逻辑（这里使用本地文件系统）
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            file_content = f.read()

        file_size = len(file_content.encode("utf-8"))
        file_hash = hashlib.md5(file_content.encode("utf-8")).hexdigest()

        logger.info(
            f"[{self.activity_name}] File fetched: size={file_size}, hash={file_hash}"
        )

        # 返回结果
        output = FetchFileOutput(
            file_content=file_content,
            file_path=file_path,
            file_size=file_size,
            file_hash=file_hash,
            metadata={
                "mime_type": "text/plain",
                "encoding": "utf-8",
                "fetched_at": datetime.now().isoformat(),
            },
            status="fetched",
        )

        return output.model_dump()


class LoadFileTask(ActivityTaskTemplate):
    """加载文件任务

    将文件加载到内存并进行基本分析
    """

    async def execute(self) -> Dict[str, Any]:
        """执行文件加载逻辑"""
        # 获取上游输出
        upstream_outputs = await self.get_upstream_outputs()
        fetch_output_data = upstream_outputs.get("fetch_file", {})

        # 解析输入
        input_data = LoadFileInput(**fetch_output_data)
        file_content = input_data.file_content
        file_path = input_data.file_path
        file_size = input_data.file_size

        logger.info(f"[{self.activity_name}] Loading file: {file_path}")

        # 模拟异步操作
        await asyncio.sleep(1)

        # 分析文件内容
        lines = file_content.split("\n")
        line_count = len(lines)
        words = file_content.split()
        word_count = len(words)
        char_count = len(file_content)

        logger.info(
            f"[{self.activity_name}] File loaded: lines={line_count}, "
            f"words={word_count}, chars={char_count}"
        )

        # 返回结果
        output = LoadFileOutput(
            file_path=file_path,
            file_content=file_content,
            file_size=file_size,
            line_count=line_count,
            word_count=word_count,
            char_count=char_count,
            metadata=input_data.metadata,
            status="loaded",
        )

        return output.model_dump()


class ReplaceContentTask(ActivityTaskTemplate):
    """替换内容任务

    根据规则替换文件内容
    """

    async def execute(self) -> Dict[str, Any]:
        """执行内容替换逻辑"""
        # 获取上游输出
        upstream_outputs = await self.get_upstream_outputs()
        load_output_data = upstream_outputs.get("load_file", {})

        # 解析输入
        file_content = load_output_data.get("file_content", "")
        file_path = load_output_data.get("file_path", "")

        # 获取替换规则（从 self.input 或配置）
        replace_rules = self.input.get(
            "replace_rules",
            [
                {"pattern": "old", "replacement": "new"},
                {"pattern": "foo", "replacement": "bar"},
            ],
        )

        logger.info(f"[{self.activity_name}] Replacing content in: {file_path}")
        logger.info(f"[{self.activity_name}] Replace rules: {len(replace_rules)}")

        # 模拟异步操作
        await asyncio.sleep(0.5)

        # 执行替换
        original_content = file_content
        replaced_content = file_content
        replace_count = 0

        for rule in replace_rules:
            pattern = rule.get("pattern", "")
            replacement = rule.get("replacement", "")
            count_before = replaced_content.count(pattern)
            replaced_content = replaced_content.replace(pattern, replacement)
            replace_count += count_before

        logger.info(f"[{self.activity_name}] Replaced {replace_count} occurrences")

        # 返回结果
        output = ReplaceContentOutput(
            original_content=original_content,
            replaced_content=replaced_content,
            replace_count=replace_count,
            metadata={
                "file_path": file_path,
                "replaced_at": datetime.now().isoformat(),
            },
            status="replaced",
        )

        return output.model_dump()


class SplitTextTask(ActivityTaskTemplate):
    """分割文本任务

    将文本分割成多个块，用于向量化和索引
    """

    async def execute(self) -> Dict[str, Any]:
        """执行文本分割逻辑"""
        # 获取上游输出
        upstream_outputs = await self.get_upstream_outputs()
        load_output_data = upstream_outputs.get("load_file", {})

        # 解析输入
        file_content = load_output_data.get("file_content", "")

        # 获取分割参数
        chunk_size = self.input.get("chunk_size", 1000)
        chunk_overlap = self.input.get("chunk_overlap", 100)
        separators = self.input.get(
            "separators", ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " "]
        )

        logger.info(f"[{self.activity_name}] Splitting text into chunks")
        logger.info(
            f"[{self.activity_name}] chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}"
        )

        # 模拟异步操作
        await asyncio.sleep(0.5)

        # 分割文本
        chunks = self._split_text(file_content, chunk_size, chunk_overlap, separators)

        chunk_count = len(chunks)
        avg_chunk_length = sum(len(chunk) for chunk in chunks) // chunk_count if chunk_count > 0 else 0

        logger.info(f"[{self.activity_name}] Split into {chunk_count} chunks")
        logger.info(f"[{self.activity_name}] Average chunk length: {avg_chunk_length}")

        # 返回结果
        output = SplitTextOutput(
            chunks=chunks,
            chunk_count=chunk_count,
            avg_chunk_length=avg_chunk_length,
            metadata={
                "original_length": len(file_content),
                "split_at": datetime.now().isoformat(),
            },
            status="splitted",
        )

        return output.model_dump()

    def _split_text(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        separators: List[str],
    ) -> List[str]:
        """分割文本为多个块

        Args:
            text: 原始文本
            chunk_size: 每个块的大小
            chunk_overlap: 块之间的重叠大小
            separators: 分隔符列表，按优先级排序

        Returns:
            文本块列表
        """
        chunks = []
        current_position = 0
        text_length = len(text)

        while current_position < text_length:
            # 计算块的结束位置
            end_position = min(current_position + chunk_size, text_length)

            # 如果还有剩余文本，尝试在分隔符处分割
            if end_position < text_length:
                for sep in separators:
                    # 在当前块中查找最后一个分隔符
                    last_sep = text.rfind(sep, current_position, end_position)
                    if last_sep != -1:
                        # 在分隔符处分割
                        end_position = last_sep + len(sep)
                        break

            # 提取文本块
            chunk = text[current_position:end_position].strip()
            if chunk:
                chunks.append(chunk)

            # 移动到下一个块（考虑重叠）
            current_position = end_position - chunk_overlap
            if current_position < 0:
                current_position = 0

        return chunks


class IndexIntoMilvusTask(ActivityTaskTemplate):
    """索引到 Milvus 任务

    将文本块索引到 Milvus 向量数据库
    """

    async def execute(self) -> Dict[str, Any]:
        """执行 Milvus 索引逻辑"""
        # 获取上游输出
        upstream_outputs = await self.get_upstream_outputs()
        split_output_data = upstream_outputs.get("split_text", {})

        # 解析输入
        chunks = split_output_data.get("chunks", [])
        chunk_count = split_output_data.get("chunk_count", 0)

        index_name = self.input.get("index_name", "file_index")

        logger.info(f"[{self.activity_name}] Indexing {chunk_count} chunks into Milvus")
        logger.info(f"[{self.activity_name}] Index name: {index_name}")

        # 模拟异步操作
        await asyncio.sleep(1)

        # 模拟索引过程（实际实现应该调用 Milvus SDK）
        ids = [f"doc_{i}_{datetime.now().timestamp()}" for i in range(chunk_count)]

        logger.info(f"[{self.activity_name}] Indexed {len(ids)} documents")

        # 返回结果
        output = IndexOutput(
            ids=ids,
            indexed_count=len(ids),
            index_name=index_name,
            metadata={
                "index_type": "milvus",
                "dimension": 768,  # 假设使用 768 维向量
                "indexed_at": datetime.now().isoformat(),
            },
            status="indexed",
        )

        return output.model_dump()


class IndexIntoEsTask(ActivityTaskTemplate):
    """索引到 Elasticsearch 任务

    将文本块索引到 Elasticsearch 搜索引擎
    """

    async def execute(self) -> Dict[str, Any]:
        """执行 ES 索引逻辑"""
        # 获取上游输出
        upstream_outputs = await self.get_upstream_outputs()
        split_output_data = upstream_outputs.get("split_text", {})

        # 解析输入
        chunks = split_output_data.get("chunks", [])
        chunk_count = split_output_data.get("chunk_count", 0)

        index_name = self.input.get("index_name", "file_index")

        logger.info(f"[{self.activity_name}] Indexing {chunk_count} chunks into Elasticsearch")
        logger.info(f"[{self.activity_name}] Index name: {index_name}")

        # 模拟异步操作
        await asyncio.sleep(1)

        # 模拟索引过程（实际实现应该调用 Elasticsearch SDK）
        ids = [f"doc_{i}_{datetime.now().timestamp()}" for i in range(chunk_count)]

        logger.info(f"[{self.activity_name}] Indexed {len(ids)} documents")

        # 返回结果
        output = IndexOutput(
            ids=ids,
            indexed_count=len(ids),
            index_name=index_name,
            metadata={
                "index_type": "elasticsearch",
                "shards": 3,
                "replicas": 1,
                "indexed_at": datetime.now().isoformat(),
            },
            status="indexed",
        )

        return output.model_dump()


class SummaryTask(ActivityTaskTemplate):
    """生成摘要任务

    生成文件内容的摘要
    """

    async def execute(self) -> Dict[str, Any]:
        """执行摘要生成逻辑"""
        # 获取上游输出
        upstream_outputs = await self.get_upstream_outputs()
        replace_output_data = upstream_outputs.get("replace_content", {})

        # 解析输入
        file_content = replace_output_data.get("replaced_content", "")

        max_length = self.input.get("max_length", 200)

        logger.info(f"[{self.activity_name}] Generating summary")
        logger.info(f"[{self.activity_name}] Max length: {max_length}")

        # 模拟异步操作
        await asyncio.sleep(0.5)

        # 生成摘要（简单实现：取前 N 个字符）
        original_length = len(file_content)
        summary = file_content[:max_length] + "..." if original_length > max_length else file_content
        summary_length = len(summary)
        compression_ratio = summary_length / original_length if original_length > 0 else 0

        logger.info(f"[{self.activity_name}] Summary generated: {summary_length} chars")
        logger.info(f"[{self.activity_name}] Compression ratio: {compression_ratio:.2%}")

        # 返回结果
        output = SummaryOutput(
            summary=summary,
            summary_length=summary_length,
            original_length=original_length,
            compression_ratio=compression_ratio,
            metadata={
                "max_length": max_length,
                "generated_at": datetime.now().isoformat(),
            },
            status="summarized",
        )

        return output.model_dump()


# =============================================================================
# Register Tasks to Celery
# =============================================================================

# 使用装饰器注册任务
fetch_file_task = activity_task(FetchFileTask)
load_file_task = activity_task(LoadFileTask)
replace_content_task = activity_task(ReplaceContentTask)
split_text_task = activity_task(SplitTextTask)
index_into_milvus_task = activity_task(IndexIntoMilvusTask)
index_into_es_task = activity_task(IndexIntoEsTask)
summary_task = activity_task(SummaryTask)


# =============================================================================
# Workflow Configuration
# =============================================================================

# 文件处理工作流配置
FILE_PROCESS_WORKFLOW = {
    "fetch_file": {
        "input": {"file_path": "/tmp/sample.txt"},
        "execute_params": {
            "task_name": fetch_file_task.name,
            "max_retries": 3,
        },
        "depends_on": [],
    },
    "load_file": {
        "execute_params": {
            "task_name": load_file_task.name,
            "max_retries": 2,
        },
        "depends_on": ["fetch_file"],
    },
    "replace_content": {
        "execute_params": {
            "task_name": replace_content_task.name,
            "max_retries": 2,
        },
        "depends_on": ["load_file"],
        "input": {
            "replace_rules": [
                {"pattern": "old", "replacement": "new"},
                {"pattern": "foo", "replacement": "bar"},
            ]
        },
    },
    "split_text": {
        "execute_params": {
            "task_name": split_text_task.name,
            "max_retries": 2,
        },
        "depends_on": ["load_file"],
        "input": {
            "chunk_size": 500,
            "chunk_overlap": 50,
        },
    },
    "index_into_milvus": {
        "execute_params": {
            "task_name": index_into_milvus_task.name,
            "max_retries": 3,
        },
        "depends_on": ["split_text"],
        "input": {
            "index_name": "file_index_milvus",
        },
    },
    "index_into_es": {
        "execute_params": {
            "task_name": index_into_es_task.name,
            "max_retries": 3,
        },
        "depends_on": ["split_text"],
        "input": {
            "index_name": "file_index_es",
        },
    },
    "summary": {
        "execute_params": {
            "task_name": summary_task.name,
            "max_retries": 2,
        },
        "depends_on": ["replace_content"],
        "input": {
            "max_length": 150,
        },
    },
}


# =============================================================================
# Task Registry for Easy Access
# =============================================================================

TASK_REGISTRY = {
    "fetch_file": fetch_file_task,
    "load_file": load_file_task,
    "replace_content": replace_content_task,
    "split_text": split_text_task,
    "index_into_milvus": index_into_milvus_task,
    "index_into_es": index_into_es_task,
    "summary": summary_task,
}


def get_task_by_name(task_name: str):
    """根据任务名称获取任务

    Args:
        task_name: 任务名称

    Returns:
        Celery 任务实例，如果不存在则返回 None
    """
    return TASK_REGISTRY.get(task_name)


__all__ = [
    # Tasks
    "fetch_file_task",
    "load_file_task",
    "replace_content_task",
    "split_text_task",
    "index_into_milvus_task",
    "index_into_es_task",
    "summary_task",
    # Workflow Config
    "FILE_PROCESS_WORKFLOW",
    # Task Registry
    "TASK_REGISTRY",
    "get_task_by_name",
]
