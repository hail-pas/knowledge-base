"""
Document Parser 模块的 conftest.py

定义测试所需的 fixtures
"""

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ext.document_parser import DocumentParser


# 测试文件目录
SAMPLE_FILES_DIR = Path(__file__).parent.parent.parent.parent / "local" / "parse_files"


@pytest.fixture
def sample_files_dir():
    """测试文件目录路径"""
    return SAMPLE_FILES_DIR


@pytest.fixture
def sample_files(sample_files_dir):
    """按类型分组的测试文件字典"""
    files = {
        "pdf": [],
        "docx": [],
        "xlsx": [],
        "pptx": [],
        "html": [],
        "csv": [],
        "json": [],
        "txt": [],
        "png": [],
        "md": [],
    }

    if not sample_files_dir.exists():
        return files

    for file_path in sample_files_dir.iterdir():
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()
        file_name = file_path.name

        if ext == ".pdf":
            files["pdf"].append(file_path)
        elif ext == ".docx":
            files["docx"].append(file_path)
        elif ext == ".xlsx":
            files["xlsx"].append(file_path)
        elif ext == ".pptx":
            files["pptx"].append(file_path)
        elif ext in [".html", ".htm"]:
            files["html"].append(file_path)
        elif ext == ".csv":
            files["csv"].append(file_path)
        elif ext == ".json":
            files["json"].append(file_path)
        elif ext == ".txt":
            files["txt"].append(file_path)
        elif ext == ".png":
            files["png"].append(file_path)
        elif ext in [".md", ".markdown"]:
            files["md"].append(file_path)

    return files


@pytest.fixture
def parser():
    """DocumentParser 实例"""
    return DocumentParser()


@pytest.fixture
def skip_if_missing():
    """跳过缺失文件的测试装饰器工厂"""

    def decorator(file_type):
        return pytest.mark.skipif(
            not SAMPLE_FILES_DIR.exists() or len(list(SAMPLE_FILES_DIR.glob(f"*.{file_type}"))) == 0,
            reason=f"No sample files found for type: {file_type}",
        )

    return decorator


@pytest.fixture
def has_pdf_files(sample_files):
    """检查是否有 PDF 测试文件"""
    return len(sample_files.get("pdf", [])) > 0


@pytest.fixture
def has_docx_files(sample_files):
    """检查是否有 DOCX 测试文件"""
    return len(sample_files.get("docx", [])) > 0


@pytest.fixture
def has_xlsx_files(sample_files):
    """检查是否有 XLSX 测试文件"""
    return len(sample_files.get("xlsx", [])) > 0


@pytest.fixture
def has_pptx_files(sample_files):
    """检查是否有 PPTX 测试文件"""
    return len(sample_files.get("pptx", [])) > 0


@pytest.fixture
def has_html_files(sample_files):
    """检查是否有 HTML 测试文件"""
    return len(sample_files.get("html", [])) > 0


@pytest.fixture
def has_csv_files(sample_files):
    """检查是否有 CSV 测试文件"""
    return len(sample_files.get("csv", [])) > 0


@pytest.fixture
def has_json_files(sample_files):
    """检查是否有 JSON 测试文件"""
    return len(sample_files.get("json", [])) > 0


@pytest.fixture
def has_txt_files(sample_files):
    """检查是否有 TXT 测试文件"""
    return len(sample_files.get("txt", [])) > 0


@pytest.fixture
def has_png_files(sample_files):
    """检查是否有 PNG 测试文件"""
    return len(sample_files.get("png", [])) > 0


@pytest.fixture
def has_md_files(sample_files):
    """检查是否有 MD 测试文件"""
    return len(sample_files.get("md", [])) > 0


@pytest.fixture
def has_poppler():
    """检查是否安装了 poppler（用于 pdf2image）"""
    return shutil.which("pdftoppm") is not None or shutil.which("pdftocairo") is not None


@pytest.fixture
def has_tesseract():
    """检查是否安装了 tesseract"""
    return shutil.which("tesseract") is not None


@pytest.fixture
def has_paddleocr():
    """检查 paddleocr 是否可用"""
    try:
        import paddleocr
        return True
    except ImportError:
        return False
