"""
Text Chunker 使用示例

演示如何使用 TextChunker 进行文本切块
"""

import asyncio
from datetime import datetime

from ext.document_parser.core.parse_result import ParseResult, PageResult, OutputFormat
from ext.text_chunker import TextChunker


async def example_length_based():
    """示例1: 按长度切块"""
    print("\n=== 示例1: 按长度切块 ===\n")

    # 模拟一个解析结果
    content = "这是一段测试文本。" * 100  # 创建较长的文本
    parse_result = ParseResult(
        content=content,
        format=OutputFormat.TEXT,
        pages=[
            PageResult(page_number=1, content=content[: len(content) // 2]),
            PageResult(page_number=2, content=content[len(content) // 2 :]),
        ],
        page_count=2,
        metadata={},
        confidence=1.0,
        engine_used="test",
    )

    # 创建切块器
    chunker = TextChunker()

    # 按字符切块
    chunks = await chunker.chunk(parse_result, strategy="length", config={"chunk_size": 200, "overlap": 50})

    print(f"生成了 {len(chunks)} 个切块:")
    for i, chunk in enumerate(chunks[:3]):  # 只显示前3个
        print(f"\n切块 {i + 1}:")
        print(f"  内容长度: {len(chunk.content)}")
        print(f"  页码: {chunk.pages}")
        print(f"  起始位置: 第{chunk.start.page_number}页, 第{chunk.start.char_index}字符")
        print(f"  结束位置: 第{chunk.end.page_number}页, 第{chunk.end.char_index}字符")
        if chunk.overlap_start and chunk.overlap_end:
            print(
                f"  Overlap: {chunk.overlap_start.page_number}:{chunk.overlap_start.char_index} -> {chunk.overlap_end.page_number}:{chunk.overlap_end.char_index}"
            )


async def example_delimiter_based():
    """示例2: 按分隔符切块"""
    print("\n=== 示例2: 按分隔符切块 ===\n")

    content = """第一章：引言

这是第一章的内容。

第二章：方法

这是第二章的内容。

第三章：结论

这是第三章的内容。"""

    parse_result = ParseResult(
        content=content,
        format=OutputFormat.TEXT,
        pages=[PageResult(page_number=1, content=content)],
        page_count=1,
        metadata={},
        confidence=1.0,
        engine_used="test",
    )

    chunker = TextChunker()

    # 按双换行符切分（段落）
    chunks = await chunker.chunk(
        parse_result, strategy="delimiter", config={"delimiters": ["\\n\\n"], "keep_delimiter": False}
    )

    print(f"生成了 {len(chunks)} 个切块:")
    for i, chunk in enumerate(chunks):
        print(f"\n切块 {i + 1}:")
        print(f"  内容: {chunk.content[:50]}...")
        print(f"  页码: {chunk.pages}")


async def example_token_based():
    """示例3: 按token切块"""
    print("\n=== 示例3: 按Token切块 ===\n")

    content = "This is a test document for token-based chunking. " * 50
    parse_result = ParseResult(
        content=content,
        format=OutputFormat.TEXT,
        pages=[
            PageResult(page_number=1, content=content),
        ],
        page_count=1,
        metadata={},
        confidence=1.0,
        engine_used="test",
    )

    chunker = TextChunker()

    # 按token切块
    chunks = await chunker.chunk(
        parse_result,
        strategy="length",
        config={"chunk_size": 100, "overlap": 20, "mode": "tokens"},
    )

    print(f"生成了 {len(chunks)} 个切块 (基于token)")
    for i, chunk in enumerate(chunks[:2]):
        print(f"\n切块 {i + 1}:")
        print(f"  内容长度: {len(chunk.content)} 字符")


async def example_heading_based():
    """示例4: 按标题层级切块"""
    print("\n=== 示例4: 按标题层级切块 ===\n")

    content = """# 第一章：引言

这是第一章的详细内容。

## 1.1 背景

背景信息...

## 1.2 目标

目标描述...

# 第二章：方法

这是第二章的详细内容。

## 2.1 研究方法

研究方法描述...

# 第三章：结论

结论内容..."""

    parse_result = ParseResult(
        content=content,
        format=OutputFormat.MARKDOWN,
        pages=[
            PageResult(page_number=1, content=content),
        ],
        page_count=1,
        metadata={},
        confidence=1.0,
        engine_used="test",
    )

    chunker = TextChunker()

    chunks = await chunker.chunk(
        parse_result,
        strategy="heading",
        config={"max_chunk_size": 500, "preserve_headings": True},
    )

    print(f"生成了 {len(chunks)} 个切块 (按标题)")
    for i, chunk in enumerate(chunks):
        print(f"\n切块 {i + 1}:")
        print(f"  内容:\n{chunk.content[:100]}...")


async def example_auto_strategy():
    """示例5: 自动策略选择"""
    print("\n=== 示例5: 自动策略选择 ===\n")

    content = """# 第一章：引言

这是第一章的详细内容。

## 1.1 背景

背景信息...

## 1.2 目标

目标描述...

# 第二章：方法

这是第二章的详细内容。"""

    parse_result = ParseResult(
        content=content,
        format=OutputFormat.MARKDOWN,
        pages=[
            PageResult(page_number=1, content=content),
        ],
        page_count=1,
        metadata={},
        confidence=1.0,
        engine_used="test",
    )

    chunker = TextChunker()

    # 使用自动策略（推荐）
    chunks = await chunker.chunk(parse_result)

    print(f"生成了 {len(chunks)} 个切块 (自动选择heading策略)")
    for i, chunk in enumerate(chunks):
        print(f"\n切块 {i + 1}:")
        print(f"  内容:\n{chunk.content[:80]}...")


async def example_override_recommended():
    """示例6: 覆盖推荐策略"""
    print("\n=== 示例6: 覆盖推荐策略 ===\n")

    content = """# 第一章：引言

这是第一章的详细内容。

## 1.1 背景

背景信息...

## 1.2 目标

目标描述...

# 第二章：方法

这是第二章的详细内容。"""

    parse_result = ParseResult(
        content=content,
        format=OutputFormat.MARKDOWN,
        pages=[
            PageResult(page_number=1, content=content),
        ],
        page_count=1,
        metadata={},
        confidence=1.0,
        engine_used="test",
    )

    chunker = TextChunker()

    # Markdown文档推荐使用heading策略，但这里强制使用length策略
    chunks = await chunker.chunk(
        parse_result,
        strategy="length",
        config={"chunk_size": 100, "overlap": 20},
    )

    print(f"生成了 {len(chunks)} 个切块 (Markdown使用length策略)")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n切块 {i + 1}:")
        print(f"  内容:\n{chunk.content[:80]}...")


async def example_custom_delimiter_for_markdown():
    """示例7: Markdown使用自定义分隔符"""
    print("\n=== 示例7: Markdown使用自定义分隔符 ===\n")

    content = """# 第一章：引言

这是第一章的详细内容。

## 1.1 背景

背景信息...

## 1.2 目标

目标描述...

# 第二章：方法

这是第二章的详细内容。"""

    parse_result = ParseResult(
        content=content,
        format=OutputFormat.MARKDOWN,
        pages=[
            PageResult(page_number=1, content=content),
        ],
        page_count=1,
        metadata={},
        confidence=1.0,
        engine_used="test",
    )

    chunker = TextChunker()

    # 使用自定义分隔符按 ## 切分（二级标题）
    chunks = await chunker.chunk(
        parse_result,
        strategy="delimiter",
        config={"delimiters": ["##"], "keep_delimiter": True},
    )

    print(f"生成了 {len(chunks)} 个切块 (按 ## 分隔符切分)")
    for i, chunk in enumerate(chunks):
        print(f"\n切块 {i + 1}:")
        print(f"  内容:\n{chunk.content[:80]}...")


async def example_regex_delimiter():
    """示例8: 正则表达式分隔符"""
    print("\n=== 示例8: 正则表达式分隔符 ===\n")

    content = """第一章：引言

这是第一章的内容。

第二章：方法

这是第二章的内容。

第三章：结论

这是第三章的内容。"""

    parse_result = ParseResult(
        content=content,
        format=OutputFormat.TEXT,
        pages=[PageResult(page_number=1, content=content)],
        page_count=1,
        metadata={},
        confidence=1.0,
        engine_used="test",
    )

    chunker = TextChunker()

    # 使用正则表达式按"第X章："分隔
    chunks = await chunker.chunk(
        parse_result,
        strategy="delimiter",
        config={"delimiters": ["regex:第[一二三四五六七八九十]+章："], "keep_delimiter": True},
    )

    print(f"生成了 {len(chunks)} 个切块 (使用正则表达式分隔符)")
    for i, chunk in enumerate(chunks):
        print(f"\n切块 {i + 1}:")
        print(f"  内容: {chunk.content[:60]}...")


async def example_list_presets():
    """示例8: 列出所有预设"""
    print("\n=== 示例8: 列出所有预设 ===\n")

    chunker = TextChunker()

    presets = chunker.list_presets()

    print("可用的策略预设:")
    for format_type, preset in presets.items():
        print(f"\n格式: {format_type.value}")
        print(f"  推荐策略: {preset['strategy']}")
        print(f"  默认配置: {preset['config']}")


async def example_json_based():
    """示例9: JSON数据切块"""
    print("\n=== 示例9: JSON数据切块 ===\n")

    import json

    data = [
        {"name": "Alice", "age": 30, "city": "New York"},
        {"name": "Bob", "age": 25, "city": "Los Angeles"},
        {"name": "Charlie", "age": 35, "city": "Chicago"},
    ]

    content = json.dumps(data, ensure_ascii=False)

    parse_result = ParseResult(
        content=content,
        format=OutputFormat.JSON,
        pages=[PageResult(page_number=1, content=content)],
        page_count=1,
        metadata={},
        confidence=1.0,
        engine_used="test",
    )

    chunker = TextChunker()

    # Simple模式
    chunks = await chunker.chunk(
        parse_result,
        strategy="json",
        config={"mode": "simple", "keys": ["name", "age"]},
    )

    print(f"生成了 {len(chunks)} 个切块 (JSON simple模式)")
    for i, chunk in enumerate(chunks):
        print(f"\n切块 {i + 1}:")
        print(f"  内容:\n{chunk.content}")

    # JSON模式
    print("\n--- JSON格式模式 ---")
    chunks = await chunker.chunk(parse_result, strategy="json", config={"mode": "json"})

    print(f"生成了 {len(chunks)} 个切块 (JSON格式模式)")
    for i, chunk in enumerate(chunks):
        print(f"\n切块 {i + 1}:")
        print(f"  内容:\n{chunk.content[:100]}...")


async def main():
    """运行所有示例"""
    print("=" * 60)
    print("Text Chunker 使用示例")
    print("=" * 60)

    await example_length_based()
    await example_delimiter_based()
    await example_token_based()
    await example_heading_based()
    await example_json_based()
    await example_auto_strategy()
    await example_override_recommended()
    await example_custom_delimiter_for_markdown()
    await example_regex_delimiter()
    await example_list_presets()

    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
