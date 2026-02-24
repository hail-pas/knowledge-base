from ext.document_parser.core.engine_base import BaseEngine


_engines: dict[str, type] = {}


def _register_default_engines():
    from ext.document_parser.engines.pdf.pymupdf import PyMUPDFEngine
    from ext.document_parser.engines.pdf.pdfplumber import PDFPlumberEngine
    from ext.document_parser.engines.ocr.paddleocr import PaddleOCREngine
    from ext.document_parser.engines.ocr.tesseract import TesseractOCREngine
    from ext.document_parser.engines.office.engines import DocxEngine, XLSXEngine, PPTXEngine
    from ext.document_parser.engines.web.engines import TrafilaturaEngine, MarkdownEngine
    from ext.document_parser.engines.web.url import URLEngine
    from ext.document_parser.engines.structured.engines import CSVEngine, JSONEngine
    from ext.document_parser.engines.amarkitdown.amarkitdown import MarkitdownEngine

    _engines.update(
        {
            "pymupdf": PyMUPDFEngine,
            "pdfplumber": PDFPlumberEngine,
            "paddleocr": PaddleOCREngine,
            "tesseract": TesseractOCREngine,
            "docx": DocxEngine,
            "xlsx": XLSXEngine,
            "pptx": PPTXEngine,
            "trafilatura": TrafilaturaEngine,
            "markdown": MarkdownEngine,
            "csv": CSVEngine,
            "json": JSONEngine,
            "markitdown": MarkitdownEngine,
            "url": URLEngine,
        }
    )


_instances: dict[str, BaseEngine] = {}


def register_engine(name: str, engine_class: type) -> None:
    _engines[name] = engine_class


def list_engines() -> list[str]:
    if not _engines:
        _register_default_engines()
    return list(_engines.keys())


def get_engine(name: str, **kwargs) -> BaseEngine | None:
    if not _engines:
        _register_default_engines()

    if name not in _engines:
        return None

    if name not in _instances:
        engine_class = _engines[name]
        _instances[name] = engine_class(**kwargs)

    return _instances[name]


def clear_cache() -> None:
    _instances.clear()
