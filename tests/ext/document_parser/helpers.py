from __future__ import annotations

import shutil
from pathlib import Path

import pytest


TEST_FILES_DIR = Path(__file__).resolve().parents[2] / "files"


def sample_file(name: str) -> Path:
    return TEST_FILES_DIR / name


def require_module(module_name: str):
    return pytest.importorskip(module_name)


def require_command(command_name: str) -> str:
    command_path = shutil.which(command_name)
    if command_path is None:
        pytest.skip(f"missing system command: {command_name}")
    return command_path
