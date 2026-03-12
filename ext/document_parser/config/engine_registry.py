from ext.document_parser.core.engine_base import BaseEngine

_engines: dict[str, type] = {}


def _register_default_engines() -> None:
    default_engines = {}

    try:
        from ext.document_parser.engines.pdf.pymupdf import PyMUPDFEngine

        default_engines[PyMUPDFEngine.engine_name] = PyMUPDFEngine
    except ImportError:
        pass

    try:
        from ext.document_parser.engines.pdf.pdfplumber import PDFPlumberEngine

        default_engines[PDFPlumberEngine.engine_name] = PDFPlumberEngine
    except ImportError:
        pass

    try:
        from ext.document_parser.engines.ocr.paddleocr import PaddleOCREngine

        default_engines[PaddleOCREngine.engine_name] = PaddleOCREngine
    except ImportError:
        pass

    try:
        from ext.document_parser.engines.ocr.tesseract import TesseractOCREngine

        default_engines[TesseractOCREngine.engine_name] = TesseractOCREngine
    except ImportError:
        pass

    try:
        from ext.document_parser.engines.office.engines import (
            DocxEngine,
            PPTXEngine,
            XLSXEngine,
        )

        default_engines[DocxEngine.engine_name] = DocxEngine
        default_engines[XLSXEngine.engine_name] = XLSXEngine
        default_engines[PPTXEngine.engine_name] = PPTXEngine
    except ImportError:
        pass

    try:
        from ext.document_parser.engines.web.engines import (
            MarkdownEngine,
            TrafilaturaEngine,
        )

        default_engines[TrafilaturaEngine.engine_name] = TrafilaturaEngine
        default_engines[MarkdownEngine.engine_name] = MarkdownEngine
    except ImportError:
        pass

    try:
        from ext.document_parser.engines.web.url import URLEngine

        default_engines[URLEngine.engine_name] = URLEngine
    except ImportError:
        pass

    try:
        from ext.document_parser.engines.structured.engines import CSVEngine, JSONEngine

        default_engines[CSVEngine.engine_name] = CSVEngine
        default_engines[JSONEngine.engine_name] = JSONEngine
    except ImportError:
        pass

    try:
        from ext.document_parser.engines.amarkitdown.amarkitdown import MarkitdownEngine

        default_engines[MarkitdownEngine.engine_name] = MarkitdownEngine
    except ImportError:
        pass

    try:
        from ext.document_parser.engines.plain import TextEngine

        default_engines[TextEngine.engine_name] = TextEngine
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
