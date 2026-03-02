from pathlib import Path
from loguru import logger
from typing import Optional, List

from ext.document_parser.core.parse_result import ParseResult, OutputFormat
from ext.document_parser.config.engine_registry import get_engine, list_engines
from ext.document_parser.processors.base import BaseProcessor


class DocumentParser:
    """简化的文档解析器"""

    def __init__(self):
        self._selection_rules = {
            ".pdf": ["pymupdf", "pdfplumber", "paddleocr"],
            ".docx": ["docx"],
            ".xlsx": ["xlsx"],
            ".pptx": ["pptx"],
            ".html": ["trafilatura"],
            ".htm": ["trafilatura"],
            ".md": ["markdown"],
            ".markdown": ["markdown"],
            ".csv": ["csv"],
            ".json": ["json"],
            ".png": ["paddleocr", "tesseract"],
            ".jpg": ["paddleocr"],
            ".jpeg": ["paddleocr"],
            ".txt": ["text"],
        }

    async def parse(
        self,
        file_path: str,
        engine: Optional[str] = None,
        output_format: OutputFormat = OutputFormat.TEXT,
        processors: Optional[List[BaseProcessor]] = None,
        options: Optional[dict] = None,
    ) -> ParseResult:
        if engine:
            engine_names = [engine]
            logger.info(f"使用指定引擎: {engine}")
        else:
            # Handle URLs first - check if file_path is a URL
            if file_path.startswith(("http://", "https://")):
                engine_instance = get_engine("url")
                if not engine_instance:
                    raise ValueError("URLEngine not registered")
                logger.info(f"检测到URL，使用URLEngine: {file_path}")
                result = await engine_instance.parse(file_path, options)
                if processors:
                    for processor in processors:
                        result = await processor.process(result)
                return result

            ext = Path(file_path).suffix.lower()
            engine_names = self._selection_rules.get(ext, [])

            if not engine_names:
                raise ValueError(f"不支持的文件类型: {ext}, file_path: {file_path}")

            logger.info(f"自动选择引擎: {engine_names}")

        last_error = None
        for engine_name in engine_names:
            try:
                engine_instance = get_engine(engine_name)
                if not engine_instance:
                    logger.warning(f"引擎 {engine_name} 未注册，跳过")
                    continue

                result = await engine_instance.parse(file_path, options)

                if not result.content:  # or len(result.content.strip()) < 10:
                    logger.warning(f"引擎 {engine_name} 解析结果为空，尝试下一个")
                    continue

                print(">>>" * 20, result.format, type(result.format))

                if processors:
                    for processor in processors:
                        result = await processor.process(result)

                if output_format == OutputFormat.MARKDOWN and result.format == OutputFormat.TEXT:
                    result.content = self._to_markdown(result.content)
                    result.format = OutputFormat.MARKDOWN

                logger.info(f"解析成功，使用引擎: {engine_name}")
                return result

            except Exception as e:
                last_error = e
                logger.warning(f"引擎 {engine_name} 失败: {e}")
                continue

        raise Exception(f"所有解析引擎都失败了: {last_error}")

    def _to_markdown(self, text: str) -> str:
        lines = text.split("\n")
        markdown_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                markdown_lines.append("")
            elif len(stripped) < 50 and stripped[0].isupper() and stripped[0].isalpha():
                markdown_lines.append(f"## {stripped}")
            else:
                markdown_lines.append(stripped)

        return "\n".join(markdown_lines)

    def list_supported_formats(self) -> dict[str, list[str]]:
        return self._selection_rules.copy()

    def list_available_engines(self) -> list[str]:
        return list_engines()
