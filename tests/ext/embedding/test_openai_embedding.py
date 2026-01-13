"""OpenAI Embedding 单元测试"""
import os
from typing import List

import pytest
import numpy as np

from ext.embedding.providers.openai import OpenAIEmbedding
from ext.embedding.base import EmbeddingResult
from ext.embedding.exceptions import EmbeddingConfigError, EmbeddingAPIError
from ext.embedding.factory import EmbeddingModelFactory
from ext.ext_tortoise.models.knowledge_base import EmbeddingModelConfig
from ext.ext_tortoise.enums import EmbeddingModelTypeEnum


@pytest.fixture
def embedding_config():
    """从环境变量获取 OpenAI Embedding 配置"""
    config = {
        "api_key": os.getenv("OPENAI_EMBEDDING_API_KEY", ""),
        "base_url": os.getenv("OPENAI_EMBEDDING_BASE_URL"),
        "timeout": 60,
    }
    return config


@pytest.fixture
def model_params():
    """从环境变量获取模型参数"""
    return {
        "model_name_or_path": os.getenv("OPENAI_EMBEDDING_MODEL_NAME", "text-embedding-3-small"),
        "dimension": int(os.getenv("OPENAI_EMBEDDING_DIMENSION", "768")),
        "max_batch_size": int(os.getenv("OPENAI_EMBEDDING_MAX_BATCH_SIZE", "512")),
        "max_token_per_text": int(os.getenv("OPENAI_EMBEDDING_MAX_TOKEN_PER_TEXT", "512")),
    }


@pytest.mark.asyncio
class TestOpenAIEmbedding:
    """OpenAI Embedding 测试类"""

    async def test_init_success(self, embedding_config, model_params):
        """测试成功初始化 OpenAIEmbedding"""
        config = embedding_config.copy()
        # 确保有 api_key，如果没有则跳过测试
        if not config["api_key"]:
            pytest.skip("OPENAI_EMBEDDING_API_KEY not set")

        # 清理 None 值
        config = {k: v for k, v in config.items() if v is not None}

        model = OpenAIEmbedding(
            model_name_or_path=model_params["model_name_or_path"],
            dimension=model_params["dimension"],
            max_batch_size=model_params["max_batch_size"],
            max_token_per_text=model_params["max_token_per_text"],
            config=config,
        )

        assert model.model_name_or_path == model_params["model_name_or_path"]
        assert model.dimension == model_params["dimension"]
        assert model.max_batch_size == model_params["max_batch_size"]
        assert model.max_token_per_text == model_params["max_token_per_text"]
        assert model.api_key == config["api_key"]
        assert model.base_url == config.get("base_url", OpenAIEmbedding.DEFAULT_BASE_URL)

        # 关闭客户端连接
        await model.close()

    async def test_init_missing_api_key(self, model_params):
        """测试缺少 api_key 时抛出异常"""
        config = {}  # 缺少 api_key

        with pytest.raises(EmbeddingConfigError, match="Missing required config keys"):
            OpenAIEmbedding(
                model_name_or_path=model_params["model_name_or_path"],
                dimension=model_params["dimension"],
                config=config,
            )

    async def test_embed_single_text(self, embedding_config, model_params):
        """测试生成单个文本的 embedding"""
        config = embedding_config.copy()
        if not config["api_key"]:
            pytest.skip("OPENAI_EMBEDDING_API_KEY not set")
        config = {k: v for k, v in config.items() if v is not None}

        model = OpenAIEmbedding(
            model_name_or_path=model_params["model_name_or_path"],
            dimension=model_params["dimension"],
            config=config,
        )

        text = "This is a test sentence for embedding."
        result = await model.embed(text)

        # 验证返回结果类型
        assert isinstance(result, EmbeddingResult)

        # 验证 embedding 数据
        assert isinstance(result.embedding, list)
        assert len(result.embedding) == model_params["dimension"]
        assert isinstance(result.embedding[0], float)

        # 验证元数据
        assert result.text == text
        assert result.index == 0
        assert result.model == model_params["model_name_or_path"]

        # 验证向量归一化（可选，OpenAI 的 embedding 通常是归一化的）
        embedding_array = np.array(result.embedding)
        norm = np.linalg.norm(embedding_array)
        # OpenAI 的 embedding 通常是 L2 归一化的，范数应该接近 1
        assert 0.9 < norm < 1.1, f"Embedding norm {norm} is not close to 1"

        await model.close()

    async def test_embed_batch_texts(self, embedding_config, model_params):
        """测试批量生成文本的 embedding"""
        config = embedding_config.copy()
        if not config["api_key"]:
            pytest.skip("OPENAI_EMBEDDING_API_KEY not set")
        config = {k: v for k, v in config.items() if v is not None}

        model = OpenAIEmbedding(
            model_name_or_path=model_params["model_name_or_path"],
            dimension=model_params["dimension"],
            config=config,
        )

        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence.",
            "Fourth test sentence.",
            "Fifth test sentence.",
        ]

        results = await model.embed_batch(texts)

        # 验证返回结果数量
        assert len(results) == len(texts)

        # 验证每个结果
        for i, result in enumerate(results):
            assert isinstance(result, EmbeddingResult)
            assert isinstance(result.embedding, list)
            assert len(result.embedding) == model_params["dimension"]
            assert result.text == texts[i]
            assert result.index == i
            assert result.model == model_params["model_name_or_path"]

            # 验证向量归一化
            embedding_array = np.array(result.embedding)
            norm = np.linalg.norm(embedding_array)
            assert 0.9 < norm < 1.1, f"Embedding {i} norm {norm} is not close to 1"

        await model.close()

    async def test_embed_batch_empty_list(self, embedding_config, model_params):
        """测试空列表批量 embedding"""
        config = embedding_config.copy()
        if not config["api_key"]:
            pytest.skip("OPENAI_EMBEDDING_API_KEY not set")
        config = {k: v for k, v in config.items() if v is not None}

        model = OpenAIEmbedding(
            model_name_or_path=model_params["model_name_or_path"],
            dimension=model_params["dimension"],
            config=config,
        )

        results = await model.embed_batch([])

        assert results == []

        await model.close()

    async def test_get_embeddings_convenience(self, embedding_config, model_params):
        """测试便捷方法 get_embeddings"""
        config = embedding_config.copy()
        if not config["api_key"]:
            pytest.skip("OPENAI_EMBEDDING_API_KEY not set")
        config = {k: v for k, v in config.items() if v is not None}

        model = OpenAIEmbedding(
            model_name_or_path=model_params["model_name_or_path"],
            dimension=model_params["dimension"],
            config=config,
        )

        texts = ["First sentence", "Second sentence"]
        embeddings = await model.get_embeddings(texts)

        # 验证返回类型
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(texts)

        # 验证每个 embedding
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert len(embedding) == model_params["dimension"]

        await model.close()

    async def test_get_dimension(self, embedding_config, model_params):
        """测试获取向量维度"""
        config = embedding_config.copy()
        if not config["api_key"]:
            pytest.skip("OPENAI_EMBEDDING_API_KEY not set")
        config = {k: v for k, v in config.items() if v is not None}

        model = OpenAIEmbedding(
            model_name_or_path=model_params["model_name_or_path"],
            dimension=model_params["dimension"],
            config=config,
        )

        assert model.get_dimension() == model_params["dimension"]

        await model.close()

    async def test_context_manager(self, embedding_config, model_params):
        """测试异步上下文管理器"""
        config = embedding_config.copy()
        if not config["api_key"]:
            pytest.skip("OPENAI_EMBEDDING_API_KEY not set")
        config = {k: v for k, v in config.items() if v is not None}

        async with OpenAIEmbedding(
            model_name_or_path=model_params["model_name_or_path"],
            dimension=model_params["dimension"],
            config=config,
        ) as model:
            result = await model.embed("Test context manager")
            assert isinstance(result, EmbeddingResult)
            assert len(result.embedding) == model_params["dimension"]

    async def test_embedding_similarity(self, embedding_config, model_params):
        """测试相似文本的 embedding 相似度更高"""
        config = embedding_config.copy()
        if not config["api_key"]:
            pytest.skip("OPENAI_EMBEDDING_API_KEY not set")
        config = {k: v for k, v in config.items() if v is not None}

        model = OpenAIEmbedding(
            model_name_or_path=model_params["model_name_or_path"],
            dimension=model_params["dimension"],
            config=config,
        )

        # 相似文本对
        similar_texts = [
            "The cat sits on the mat.",
            "A cat is sitting on the mat.",
        ]

        # 不相似的文本对
        dissimilar_texts = [
            "The cat sits on the mat.",
            "I love programming in Python.",
        ]

        # 获取 embeddings
        results_similar = await model.embed_batch(similar_texts)
        results_dissimilar = await model.embed_batch(dissimilar_texts)

        # 计算余弦相似度
        def cosine_similarity(emb1: List[float], emb2: List[float]) -> float:
            vec1 = np.array(emb1)
            vec2 = np.array(emb2)
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

        sim_similar = cosine_similarity(
            results_similar[0].embedding,
            results_similar[1].embedding
        )
        sim_dissimilar = cosine_similarity(
            results_dissimilar[0].embedding,
            results_dissimilar[1].embedding
        )

        # 相似文本的相似度应该高于不相似的文本
        assert sim_similar > sim_dissimilar

        await model.close()

    async def test_factory_create_from_config_and_embed(self, embedding_config, model_params):
        """测试使用工厂方法从 EmbeddingModelConfig 创建实例并嵌入文本"""
        config_data = embedding_config.copy()
        if not config_data["api_key"]:
            pytest.skip("OPENAI_EMBEDDING_API_KEY not set")
        config_data = {k: v for k, v in config_data.items() if v is not None}

        # 创建一个 EmbeddingModelConfig 对象（临时对象，未保存到数据库）
        embedding_model_config = EmbeddingModelConfig(
            name="test_openai_config",
            type=EmbeddingModelTypeEnum.openai,
            model_name_or_path=model_params["model_name_or_path"],
            dimension=model_params["dimension"],
            max_batch_size=model_params["max_batch_size"],
            max_token_per_request=model_params["max_batch_size"] * model_params["max_token_per_text"],
            max_token_per_text=model_params["max_token_per_text"],
            config=config_data,
            is_enabled=True,
            is_default=False,
            description="Test config for OpenAI embedding"
        )

        # 使用工厂创建 OpenAIEmbedding 实例
        model = await EmbeddingModelFactory.create(embedding_model_config, use_cache=False)

        # 验证模型类型
        assert isinstance(model, OpenAIEmbedding)
        assert model.model_name_or_path == model_params["model_name_or_path"]
        assert model.dimension == model_params["dimension"]

        # 测试嵌入一条文本
        text = "Testing factory method with EmbeddingModelConfig."
        result = await model.embed(text)

        # 验证返回结果类型
        assert isinstance(result, EmbeddingResult)
        assert result.text == text
        assert result.index == 0
        assert len(result.embedding) == model_params["dimension"]

        # 验证向量归一化
        import numpy as np
        embedding_array = np.array(result.embedding)
        norm = np.linalg.norm(embedding_array)
        assert 0.9 < norm < 1.1, f"Embedding norm {norm} is not close to 1"

        # 关闭客户端连接
        await model.close()
