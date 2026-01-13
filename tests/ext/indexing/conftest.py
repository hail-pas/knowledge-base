"""Indexing 模块测试共享配置和 Fixtures"""
import os
import numpy as np
import pytest


# ============================================================================
# Elasticsearch Fixtures
# ============================================================================

@pytest.fixture
def es_config():
    """从环境变量获取 Elasticsearch 配置"""
    return {
        "host": os.getenv("ES_HOST", "localhost"),
        "port": int(os.getenv("ES_PORT", "9200")),
        "username": os.getenv("ES_USERNAME"),
        "password": os.getenv("ES_PASSWORD"),
        "secure": os.getenv("ES_SECURE", "false").lower() == "true",
        "timeout": 30,
        "verify_certs": False,  # 测试环境可以禁用证书验证
    }


# ============================================================================
# Milvus Fixtures
# ============================================================================

@pytest.fixture
def milvus_config():
    """从环境变量获取 Milvus 配置"""
    return {
        "host": os.getenv("MILVUS_HOST", "localhost"),
        "port": int(os.getenv("MILVUS_PORT", "19530")),
        "username": os.getenv("MILVUS_USERNAME"),
        "password": os.getenv("MILVUS_PASSWORD"),
        "secure": os.getenv("MILVUS_SECURE", "false").lower() == "true",
        "db_name": os.getenv("MILVUS_DATABASE", "default"),
        "timeout": 30,
    }


# ============================================================================
# 通用 Fixtures
# ============================================================================

@pytest.fixture
def sample_embedding():
    """生成示例 embedding 向量（768维）"""
    return np.random.rand(768).tolist()


# ============================================================================
# 测试跳过辅助函数
# ============================================================================

def should_skip_es_test(es_config: dict) -> bool:
    """判断是否应该跳过 Elasticsearch 测试"""
    return not es_config["host"] or es_config["host"] == "localhost"


def should_skip_milvus_test(milvus_config: dict) -> bool:
    """判断是否应该跳过 Milvus 测试"""
    return not milvus_config["host"] or milvus_config["host"] == "localhost"


# ============================================================================
# pytest 配置
# ============================================================================

def pytest_configure(config):
    """pytest 配置钩子"""
    config.addinivalue_line(
        "markers", "elasticsearch: mark test as requiring Elasticsearch"
    )
    config.addinivalue_line(
        "markers", "milvus: mark test as requiring Milvus"
    )
