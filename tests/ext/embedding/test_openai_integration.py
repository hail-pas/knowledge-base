"""
OpenAI Embedding Provider 集成测试

使用真实 API 进行测试，配置从环境变量读取
"""

import pytest
import asyncio

# 导入 providers 模块以触发自动注册
import ext.embedding.providers

from ext.embedding import EmbeddingModelFactory
from ext.ext_tortoise.models.knowledge_base import EmbeddingModelConfig
from tests.ext.embedding.conftest import (
    openai_embedding_config,
    openai_embedding_config_with_extra,
    sample_texts,
    long_text,
    skip_if_no_api_key,
)


@skip_if_no_api_key
class TestOpenAIEmbeddingIntegration:
    """OpenAI Embedding 集成测试（需要真实 API）"""

    @pytest.mark.asyncio
    async def test_create_model_from_config(self, openai_embedding_config):
        """测试从配置创建模型"""
        model = await EmbeddingModelFactory.create(openai_embedding_config, use_cache=False)

        assert model is not None
        assert model.model_name == openai_embedding_config.model_name
        assert model.dimension == openai_embedding_config.dimension

    @pytest.mark.asyncio
    async def test_embed_single_text(self, openai_embedding_config):
        """测试生成单个 embedding"""
        model = await EmbeddingModelFactory.create(openai_embedding_config, use_cache=False)

        text = "Hello, world!"
        embeddings = await model.embed_batch([text])

        assert len(embeddings) == 1
        assert isinstance(embeddings[0], list)
        assert len(embeddings[0]) == openai_embedding_config.dimension
        assert all(isinstance(x, (float, int)) for x in embeddings[0])

    @pytest.mark.asyncio
    async def test_embed_batch_texts(self, openai_embedding_config, sample_texts):
        """测试批量生成 embeddings"""
        model = await EmbeddingModelFactory.create(openai_embedding_config, use_cache=False)

        embeddings = await model.embed_batch(sample_texts)

        assert len(embeddings) == len(sample_texts)
        for embedding in embeddings:
            assert len(embedding) == openai_embedding_config.dimension

    @pytest.mark.asyncio
    async def test_embed_with_extra_config(self, openai_embedding_config_with_extra):
        """测试使用额外配置"""
        model = await EmbeddingModelFactory.create(openai_embedding_config_with_extra, use_cache=False)

        text = "Test with extra config"
        embeddings = await model.embed_batch([text])

        assert len(embeddings) == 1

    @pytest.mark.asyncio
    async def test_long_text(self, openai_embedding_config, long_text, caplog):
        """测试超长文本处理"""
        import logging

        # 设置日志级别
        caplog.set_level(logging.WARNING)

        model = await EmbeddingModelFactory.create(openai_embedding_config, use_cache=False)

        # 生成 embedding
        embeddings = await model.embed_batch([long_text])

        # 检查是否成功
        assert len(embeddings) == 1

    @pytest.mark.asyncio
    async def test_model_caching(self, openai_embedding_config):
        """测试模型缓存"""
        # 第一次创建
        model1 = await EmbeddingModelFactory.create(openai_embedding_config, use_cache=True)

        # 第二次创建（应该从缓存获取）
        model2 = await EmbeddingModelFactory.create(openai_embedding_config, use_cache=True)

        assert model1 is model2  # 应该是同一个实例

        # 清除缓存
        EmbeddingModelFactory.clear_cache(openai_embedding_config.id)

        # 再次创建（应该创建新实例）
        model3 = await EmbeddingModelFactory.create(openai_embedding_config, use_cache=True)

        assert model1 is not model3  # 应该是新实例

    @pytest.mark.asyncio
    async def test_large_batch(self, openai_embedding_config):
        """测试大批量处理"""
        model = await EmbeddingModelFactory.create(openai_embedding_config, use_cache=False)

        # 创建大量文本（超过默认批大小）
        large_texts = [f"Text {i}" for i in range(200)]

        embeddings = await model.embed_batch(large_texts)

        assert len(embeddings) == len(large_texts)
        for embedding in embeddings:
            assert len(embedding) == openai_embedding_config.dimension

    @pytest.mark.asyncio
    async def test_empty_texts(self, openai_embedding_config):
        """测试空文本列表"""
        model = await EmbeddingModelFactory.create(openai_embedding_config, use_cache=False)

        embeddings = await model.embed_batch([])

        assert embeddings == []

    @pytest.mark.asyncio
    async def test_error_handling_invalid_api_key(self, openai_embedding_config):
        """测试错误处理：无效的 API key"""
        # 创建新配置对象
        from ext.ext_tortoise.models.knowledge_base import EmbeddingModelConfig

        # 清理可能存在的旧配置
        await EmbeddingModelConfig.filter(name="test-invalid-api-key").delete()

        config = await EmbeddingModelConfig.create(
            name="test-invalid-api-key",
            type=openai_embedding_config.type,
            model_name=openai_embedding_config.model_name,
            dimension=openai_embedding_config.dimension,
            api_key="invalid-api-key",
            base_url=openai_embedding_config.base_url,
            max_chunk_length=openai_embedding_config.max_chunk_length,
            batch_size=openai_embedding_config.batch_size,
            max_retries=openai_embedding_config.max_retries,
            timeout=openai_embedding_config.timeout,
            rate_limit=openai_embedding_config.rate_limit,
            extra_config=openai_embedding_config.extra_config,
            is_enabled=openai_embedding_config.is_enabled,
            is_default=openai_embedding_config.is_default,
            description="测试无效API key",
        )

        model = await EmbeddingModelFactory.create(config, use_cache=False)

        # 应该抛出 RuntimeError
        try:
            await model.embed_batch(["Test"])
            assert False, "Expected RuntimeError was not raised"
        except RuntimeError as e:
            # 验证错误信息
            assert "embedding请求失败" in str(e) or "401" in str(e) or "Cannot find embeddings at path" in str(e)
        finally:
            # 清理测试数据
            await EmbeddingModelConfig.filter(name="test-invalid-api-key").delete()

    @pytest.mark.asyncio
    async def test_dimension_consistency(self, openai_embedding_config):
        """测试维度一致性"""
        model = await EmbeddingModelFactory.create(openai_embedding_config, use_cache=False)

        text = "Test dimension consistency"
        embeddings = await model.embed_batch([text])

        # 检查返回的embedding维度是否与配置一致
        assert len(embeddings[0]) == openai_embedding_config.dimension

    @pytest.mark.asyncio
    async def test_multiple_requests(self, openai_embedding_config):
        """测试多次请求"""
        model = await EmbeddingModelFactory.create(openai_embedding_config, use_cache=False)

        # 进行多次请求
        for i in range(3):
            text = f"Test request {i}"
            embeddings = await model.embed_batch([text])

            assert len(embeddings) == 1
            assert len(embeddings[0]) == openai_embedding_config.dimension
