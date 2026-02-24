"""
File Source 模块的 conftest.py

定义测试所需的 fixtures
"""

import os
import tempfile
from pathlib import Path

import pytest


# Local File Source 配置
@pytest.fixture
def local_temp_dir():
    """创建临时目录用于 Local File Source 测试"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def local_provider_config(local_temp_dir):
    """Local Provider 配置"""
    return {
        "access_key": None,
        "secret_key": None,
        "endpoint": None,
        "region": None,
        "storage_location": str(local_temp_dir),
        "use_ssl": False,
        "verify_ssl": False,
        "timeout": 30,
        "max_retries": 3,
        "concurrent_limit": 10,
        "max_connections": 10,
        "extra_config": {},
    }


# Aliyun OSS 配置（从环境变量获取）
ALIYUN_ACCESS_KEY_ID = os.getenv("ALIYUN_ACCESS_KEY_ID")
ALIYUN_ACCESS_KEY_SECRET = os.getenv("ALIYUN_ACCESS_KEY_SECRET")
ALIYUN_ENDPOINT = os.getenv("ALIYUN_ENDPOINT")
ALIYUN_REGION = os.getenv("ALIYUN_REGION")
ALIYUN_BUCKET_NAME = os.getenv("ALIYUN_BUCKET_NAME")
ALIYUN_PRIVATE_KEY_PATH = os.getenv("ALIYUN_PRIVATE_KEY_PATH")
ALIYUN_PUBLIC_KEY_PATH = os.getenv("ALIYUN_PUBLIC_KEY_PATH")


@pytest.fixture
def aliyun_oss_provider_config():
    """Aliyun OSS Provider 配置"""
    extra_config = {"is_encrypted_bucket": True, "enable_crc": True}

    # 如果提供了 RSA 密钥对路径，读取密钥内容
    if ALIYUN_PRIVATE_KEY_PATH and ALIYUN_PUBLIC_KEY_PATH:
        if Path(ALIYUN_PRIVATE_KEY_PATH).exists() and Path(ALIYUN_PUBLIC_KEY_PATH).exists():
            with open(ALIYUN_PRIVATE_KEY_PATH, "r") as f:
                extra_config["private_key_content"] = f.read() # type: ignore
            with open(ALIYUN_PUBLIC_KEY_PATH, "r") as f:
                extra_config["public_key_content"] = f.read() # type: ignore

    return {
        "access_key": ALIYUN_ACCESS_KEY_ID,
        "secret_key": ALIYUN_ACCESS_KEY_SECRET,
        "endpoint": ALIYUN_ENDPOINT,
        "region": ALIYUN_REGION,
        "storage_location": ALIYUN_BUCKET_NAME,
        "use_ssl": True,
        "verify_ssl": False,
        "timeout": 30,
        "max_retries": 3,
        "concurrent_limit": 10,
        "max_connections": 10,
        "extra_config": extra_config,
    }


def pytest_configure(config):
    """注册自定义 marker"""
    config.addinivalue_line(
        "markers",
        "skip_if_no_aliyun_config: skips test if aliyun config not set"
    )

    config.addinivalue_line(
        "markers",
        "skip_if_no_s3_config: skips test if s3 config not set"
    )

def pytest_runtest_setup(item):
    """自动处理 marker"""
    if item.get_closest_marker("skip_if_no_aliyun_config"):
        if not all([ALIYUN_ACCESS_KEY_ID, ALIYUN_ACCESS_KEY_SECRET, ALIYUN_ENDPOINT, ALIYUN_REGION, ALIYUN_BUCKET_NAME]):
            pytest.skip("aliyun config not set")


    if item.get_closest_marker("skip_if_no_s3_config"):
        if not all([ALIYUN_ACCESS_KEY_ID, ALIYUN_ACCESS_KEY_SECRET, ALIYUN_ENDPOINT, ALIYUN_REGION, ALIYUN_BUCKET_NAME]):
            pytest.skip("s3 config not set")


@pytest.fixture
def s3_provider_config():
    """S3 Provider 配置（使用阿里云 S3 兼容）"""
    return {
        "access_key": ALIYUN_ACCESS_KEY_ID,
        "secret_key": ALIYUN_ACCESS_KEY_SECRET,
        "endpoint": ALIYUN_ENDPOINT,
        "region": ALIYUN_REGION,
        "storage_location": ALIYUN_BUCKET_NAME,
        "use_ssl": True,
        "verify_ssl": False,
        "timeout": 30,
        "max_retries": 3,
        "concurrent_limit": 10,
        "max_connections": 10,
        "extra_config": {
            "addressing_style": "virtual",
            "payload_transfer_threshold": 0,
            "config": {"request_checksum_calculation": "when_required"},
        },
    }


@pytest.fixture
def sample_file_content():
    """示例文件内容"""
    return b"Hello, World! This is a test file for file source providers."


@pytest.fixture
def sample_file_name():
    """示例文件名"""
    return "test_file.txt"


@pytest.fixture
def sample_file_content_type():
    """示例文件 MIME 类型"""
    return "text/plain"
