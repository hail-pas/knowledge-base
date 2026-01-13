import asyncio
from temporalio import activity
from typing import Dict, Any
from pydantic import BaseModel

class FileInput(BaseModel):
    file_path: str

class FileOutput(BaseModel):
    file_content: str
    file_path: str
    file_size: int
    status: str

class LoadOutput(BaseModel):
    file_path: str
    file_content: str
    loaded_chunks: list[str]
    status: str

class SplitOutput(BaseModel):
    split_texts: list[str]
    text_count: int
    status: str


class IndexOutput(BaseModel):
    ids: list[str]
    indexed_count: int
    index_type: str
    status: str

class SummaryOutput(BaseModel):
    summary: str
    summary_length: int
    status: str

@activity.defn
async def fetch_file(input_data: dict) -> dict:
    """模拟获取文件"""
    input_data = FileInput(**input_data)
    file_path = input_data.file_path
    print(f"[Activity] Fetching file: {file_path}")

    await asyncio.sleep(12)  # 模拟异步操作

    # 模拟获取文件，返回文件内容和元数据
    return FileOutput(**{
        "file_content": "This is the content of the file",
        "file_path": file_path,
        "file_size": 1024,
        "status": "fetched"
    }).model_dump()

@activity.defn
async def load_file(input_data: dict) -> dict:
    """模拟加载文件"""
    input_data = FileOutput(**input_data)
    file_path = input_data.file_path
    file_content = input_data.file_content
    print(f"[Activity] Loading file: {file_path}")

    await asyncio.sleep(12)  # 模拟异步操作

    # 模拟加载文件到内存
    return LoadOutput(**{
        "file_path": file_path,
        "file_content": file_content,
        "loaded_chunks": ["chunk1", "chunk2", "chunk3"],
        "status": "loaded"
    }).model_dump()

@activity.defn
async def split_text(input_data: dict) -> Dict:
    """模拟分割文本"""
    input_data = LoadOutput(**input_data)
    file_content = input_data.file_content
    print(f"[Activity] Splitting text content")

    await asyncio.sleep(12)  # 模拟异步操作

    # 模拟将文本分割成多个段落
    split_texts = [
        "First paragraph of text",
        "Second paragraph of text",
        "Third paragraph of text"
    ]

    return SplitOutput(**{
        "split_texts": split_texts,
        "text_count": len(split_texts),
        "status": "split"
    }).model_dump()

@activity.defn
async def index_into_milvus(input_data: dict) -> Dict:
    """模拟索引到 Milvus"""
    input_data = SplitOutput(**input_data)
    split_texts = input_data.split_texts
    print(f"[Activity] Indexing {len(split_texts)} texts into Milvus")

    await asyncio.sleep(12)  # 模拟异步操作
    # 模拟将文本索引到 Milvus 向量数据库
    milvus_ids = [f"milvus_id_{i}" for i in range(len(split_texts))]

    return IndexOutput(**{
        "ids": milvus_ids,
        "indexed_count": len(split_texts),
        "index_type": "milvus",
        "status": "indexed_milvus"
    }).model_dump()

@activity.defn
async def index_into_es(input_data: dict) -> dict:
    """模拟索引到 Elasticsearch"""
    input_data = SplitOutput(**input_data)
    split_texts = input_data.split_texts
    print(f"[Activity] Indexing {len(split_texts)} texts into Elasticsearch")

    await asyncio.sleep(12)  # 模拟异步操作
    # 模拟将文本索引到 Elasticsearch
    es_ids = [f"es_id_{i}" for i in range(len(split_texts))]

    return IndexOutput(**{
        "ids": es_ids,
        "indexed_count": len(split_texts),
        "index_type": "elasticsearch",
        "status": "indexed_es"
    }).model_dump()

@activity.defn
async def summary(input_data: Dict) -> dict:
    """模拟生成摘要"""
    input_data = LoadOutput(**input_data)
    file_content = input_data.file_content
    print(f"[Activity] Generating summary from Milvus and ES results")
    await asyncio.sleep(12)  # 模拟异步操作

    # 模拟生成摘要，综合 Milvus 和 ES 的索引结果
    summary_text = f"Summary: {file_content}"

    return SummaryOutput(
        summary=summary_text,
        summary_length=len(summary_text),
        status="summarized"
    ).model_dump()
