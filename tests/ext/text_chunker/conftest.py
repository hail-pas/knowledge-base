"""
Text Chunker 模块的 conftest.py

定义测试所需的 fixtures
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ext.document_parser.core.parse_result import OutputFormat, ParseResult, PageResult
from ext.text_chunker import TextChunker


@pytest.fixture
def text_chunker():
    """TextChunker 实例"""
    return TextChunker()


@pytest.fixture
def short_text():
    """短文本"""
    return "This is a short text for testing."


@pytest.fixture
def long_text():
    """长文本（适合切块测试）"""
    return "This is a test sentence. " * 100


@pytest.fixture
def multi_page_text():
    """多页文本"""
    page1 = "Content of page 1. " * 50
    page2 = "Content of page 2. " * 50
    page3 = "Content of page 3. " * 50
    return page1, page2, page3


@pytest.fixture
def markdown_with_headings():
    """包含标题的 Markdown 文本"""
    return """# 第一章：引言

这是第一章的详细内容。

## 1.1 背景

背景信息...

## 1.2 目标

目标描述...

# 第二章：方法

这是第二章的详细内容。

## 2.1 研究方法

研究方法描述...

## 2.2 数据分析

数据分析内容...

# 第三章：结论

结论内容..."""


@pytest.fixture
def chinese_document():
    """中文文档（包含章节标题）"""
    return """第一章：引言

这是第一章的内容。

第二章：方法

这是第二章的内容。

第三章：结论

这是第三章的内容。"""


@pytest.fixture
def json_list_data():
    """JSON 列表数据"""
    import json

    data = [
        {"name": "Alice", "age": 30, "city": "New York"},
        {"name": "Bob", "age": 25, "city": "Los Angeles"},
        {"name": "Charlie", "age": 35, "city": "Chicago"},
        {"name": "Diana", "age": 28, "city": "Boston"},
    ]
    return json.dumps(data, ensure_ascii=False)


@pytest.fixture
def json_dict_data():
    """JSON 字典数据"""
    import json

    data = {
        "title": "Test Document",
        "author": "Test Author",
        "sections": ["Section 1", "Section 2", "Section 3"],
        "metadata": {"created": "2024-01-01", "tags": ["test", "example"]},
    }
    return json.dumps(data, ensure_ascii=False)


@pytest.fixture
def json_nested_data():
    """嵌套 JSON 数据"""
    import json

    data = {
        "user": {
            "name": "Alice",
            "profile": {"age": 30, "city": "New York"},
            "posts": [
                {"title": "Post 1", "content": "Content 1"},
                {"title": "Post 2", "content": "Content 2"},
            ],
        }
    }
    return json.dumps(data, ensure_ascii=False)


@pytest.fixture
def text_with_delimiters():
    """包含分隔符的文本"""
    return """Section 1: Introduction

This is the introduction section.

Section 2: Methodology

This is the methodology section.

Section 3: Results

These are the results."""


def create_parse_result(content: str, format: OutputFormat, page_count: int = 1) -> ParseResult:
    """辅助函数：创建 ParseResult"""

    if page_count == 1:
        pages = [PageResult(page_number=1, content=content)]
    else:
        pages = []
        content_len = len(content)
        page_size = content_len // page_count

        for i in range(page_count):
            start = i * page_size
            end = start + page_size if i < page_count - 1 else content_len
            page_content = content[start:end]
            pages.append(PageResult(page_number=i + 1, content=page_content))

    return ParseResult(
        content=content,
        format=format,
        pages=pages,
        page_count=page_count,
        metadata={},
        confidence=1.0,
        engine_used="test",
    )


@pytest.fixture
def text_parse_result(short_text):
    """TEXT 格式的 ParseResult"""
    return create_parse_result(short_text, OutputFormat.TEXT)


@pytest.fixture
def long_text_parse_result(long_text):
    """长文本的 ParseResult"""
    return create_parse_result(long_text, OutputFormat.TEXT)


@pytest.fixture
def multi_page_parse_result(multi_page_text):
    """多页 ParseResult"""
    content = "\n\n".join(multi_page_text)
    return create_parse_result(content, OutputFormat.TEXT, page_count=3)


@pytest.fixture
def markdown_parse_result(markdown_with_headings):
    """MARKDOWN 格式的 ParseResult"""
    return create_parse_result(markdown_with_headings, OutputFormat.MARKDOWN)


@pytest.fixture
def json_parse_result(json_list_data):
    """JSON 格式的 ParseResult"""
    return create_parse_result(json_list_data, OutputFormat.JSON)


@pytest.fixture
def empty_parse_result():
    """空的 ParseResult"""
    return create_parse_result("", OutputFormat.TEXT)


@pytest.fixture
def single_char_parse_result():
    """单字符 ParseResult"""
    return create_parse_result("A", OutputFormat.TEXT)
