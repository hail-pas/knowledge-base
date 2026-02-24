from ext.document_parser.core.engine_base import BaseEngine


_engines: dict[str, type] = {}


def _register_default_engines():

    default_engines = {}

    try:
        from ext.document_parser.engines.pdf.pymupdf import PyMUPDFEngine
        default_engines["pymupdf"] = PyMUPDFEngine
    except ImportError:
        pass

    try:
        from ext.document_parser.engines.pdf.pdfplumber import PDFPlumberEngine
        default_engines["pdfplumber"] = PDFPlumberEngine
    except ImportError:
        pass

    try:
        from ext.document_parser.engines.ocr.paddleocr import PaddleOCREngine
        default_engines["paddleocr"] = PaddleOCREngine
    except ImportError:
        pass

    try:
        from ext.document_parser.engines.ocr.tesseract import TesseractOCREngine
        default_engines["tesseract"] = TesseractOCREngine
    except ImportError:
        pass

    try:
        from ext.document_parser.engines.office.engines import DocxEngine, XLSXEngine, PPTXEngine
        default_engines["docx"] = DocxEngine
        default_engines["xlsx"] = XLSXEngine
        default_engines["pptx"] = PPTXEngine
    except ImportError:
        pass

    try:
        from ext.document_parser.engines.web.engines import TrafilaturaEngine, MarkdownEngine
        default_engines["trafilatura"] = TrafilaturaEngine
        default_engines["markdown"] = MarkdownEngine
    except ImportError:
        pass

    try:
        from ext.document_parser.engines.web.url import URLEngine
        default_engines["url"] = URLEngine
    except ImportError:
        pass

    try:
        from ext.document_parser.engines.structured.engines import CSVEngine, JSONEngine
        default_engines["csv"] = CSVEngine
        default_engines["json"] = JSONEngine
    except ImportError:
        pass

    try:
        from ext.document_parser.engines.amarkitdown.amarkitdown import MarkitdownEngine
        default_engines["markitdown"] = MarkitdownEngine
    except ImportError:
        pass

    _engines.update(default_engines)


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
