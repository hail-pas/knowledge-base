# Document Parser Module

简化的多引擎文档解析模块，支持 PDF、Word、Excel、PPT、HTML、Markdown、图片等多种格式的智能解析。

## 特性

- **多引擎支持**: 同一文件类型支持多个解析引擎，自动选择最佳结果
- **手动指定引擎**: 可手动指定使用哪个解析引擎
- **后处理接口**: 支持内容清洗、去重、敏感信息处理
- **结构化数据**: 支持表格数据的结构化提取
- **分页支持**: PDF、PPT 等支持按页访问内容

## 快速开始

### 基础使用

```python
import asyncio
from ext.document_parser import DocumentParser

async def main():
    parser = DocumentParser()
    result = await parser.parse("document.pdf")
    
    print(f"使用引擎: {result.engine_used}")
    print(f"内容: {result.content}")

asyncio.run(main())
```

### 手动指定引擎

```python
parser = DocumentParser()
result = await parser.parse("document.pdf", engine="pdfplumber")
```

### 带后处理

```python
from ext.document_parser import TextCleaner, EmailSanitizer, PhoneSanitizer

parser = DocumentParser()
result = await parser.parse(
    "document.pdf",
    processors=[
        TextCleaner(remove_extra_whitespace=True),
        EmailSanitizer(),
        PhoneSanitizer(),
    ]
)
```

## 支持的文件格式

| 文件类型 | 主引擎 | 次引擎 |
|---------|--------|--------|
| PDF | pymupdf, pdfplumber | paddleocr |
| Word (.docx) | docx | - |
| Excel (.xlsx) | xlsx | - |
| PowerPoint (.pptx) | pptx | - |
| HTML | trafilatura | - |
| Markdown | markdown | - |
| CSV | csv | - |
| JSON | json | - |
| 图片 | paddleocr | tesseract |

## 后处理器

### TextCleaner - 文本清洗

```python
from ext.document_parser import TextCleaner

cleaner = TextCleaner(
    remove_extra_whitespace=True,
    normalize_quotes=True,
    remove_control_chars=True,
)
```

### ContentDeduplicator - 内容去重

```python
from ext.document_parser import ContentDeduplicator

deduplicator = ContentDeduplicator()
```

### 敏感信息处理

```python
from ext.document_parser import EmailSanitizer, PhoneSanitizer, IDCardSanitizer

email_sanitizer = EmailSanitizer()
phone_sanitizer = PhoneSanitizer()
id_card_sanitizer = IDCardSanitizer()
```

## 扩展：添加自定义引擎

```python
from ext.document_parser import BaseEngine, register_engine
from ext.document_parser.core.parse_result import ParseResult, OutputFormat

class MyCustomEngine(BaseEngine):
    engine_name = "my_custom"
    supported_formats = [".custom"]
    
    async def parse(self, file_path, options=None):
        return ParseResult(
            content="解析内容",
            format=OutputFormat.TEXT,
            confidence=0.90,
            engine_used="my_custom"
        )

register_engine("my_custom", MyCustomEngine)
```

## 扩展：添加自定义处理器

```python
from ext.document_parser import BaseProcessor
from ext.document_parser.core.parse_result import ParseResult

class MyProcessor(BaseProcessor):
    async def process(self, result: ParseResult) -> ParseResult:
        result.content = result.content.upper()
        return result
```

## 架构说明

```
ext/document_parser/
├── core/
│   ├── engine_base.py      # 引擎基类
│   ├── parse_result.py     # 解析结果
│   └── parser.py           # 解析器（简化版）
├── engines/                # 各种引擎实现
│   ├── pdf/
│   ├── ocr/
│   ├── office/
│   ├── web/
│   └── structured/
├── processors/             # 后处理器
│   ├── base.py
│   ├── cleaners.py
│   ├── deduplicator.py
│   └── sanitizers.py
└── config/
    └── engine_registry.py  # 引擎注册表
```

## 依赖

```
pymupdf>=1.23.0
pdfplumber>=0.10.0
paddleocr>=2.7.0
pytesseract>=0.3.10
pdf2image>=1.16.0
python-docx>=1.0.0
openpyxl>=3.1.0
python-pptx>=0.6.0
trafilatura>=1.6.0
pandas>=2.0.0
pillow>=10.0.0
```

## 许可证

MIT License
