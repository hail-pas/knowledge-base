"""
测试 embedding 模块的基本功能

不依赖真实 API 的单元测试
"""

from ext.embedding import EmbeddingModelFactory
from ext.embedding.base import BaseEmbeddingModel
from ext.embedding.providers.openai import OpenAIEmbeddingModel
from ext.embedding.providers.types import OpenAIExtraConfig
from ext.ext_tortoise.enums import EmbeddingModelTypeEnum
from ext.ext_tortoise.models.knowledge_base import EmbeddingModelConfig


class TestEmbeddingFactory:
    """测试 Embedding 工厂"""

    def test_register_provider(self):
        """测试注册 provider"""
        assert EmbeddingModelFactory.has_provider(EmbeddingModelTypeEnum.openai)

    def test_get_registered_model_types(self):
        """测试获取已注册的模型类型"""
        types = EmbeddingModelFactory.get_registered_model_types()
        assert len(types) > 0
        assert EmbeddingModelTypeEnum.openai in types

    def test_get_cache_info(self):
        """测试获取缓存信息"""
        info = EmbeddingModelFactory.get_cache_info()
        assert "cached_count" in info
        assert "cached_ids" in info
        assert "registered_models" in info


class TestOpenAIEmbeddingModel:
    """测试 OpenAI Embedding 模型"""

    def test_generic_type_inference(self, openai_embedding_config):
        """测试泛型类型自动推断"""
        # 检查泛型参数
        if hasattr(OpenAIEmbeddingModel, "__orig_bases__"):
            bases = OpenAIEmbeddingModel.__orig_bases__  # type: ignore
            assert len(bases) > 0
            # 检查是否包含泛型参数
            assert any(hasattr(base, "__args__") for base in bases)

    def test_create_model(self, openai_embedding_config):
        """测试创建模型实例"""
        model = OpenAIEmbeddingModel(
            model_name=openai_embedding_config.model_name,
            model_type=openai_embedding_config.type.value,
            dimension=openai_embedding_config.dimension,
            api_key=openai_embedding_config.api_key,
            base_url=openai_embedding_config.base_url,
            max_chunk_length=openai_embedding_config.max_chunk_length,
            batch_size=openai_embedding_config.batch_size,
            max_retries=openai_embedding_config.max_retries,
            timeout=openai_embedding_config.timeout,
            rate_limit=openai_embedding_config.rate_limit,
            extra_config=openai_embedding_config.extra_config,
        )

        assert model is not None
        assert model.model_name == openai_embedding_config.model_name
        assert model.dimension == openai_embedding_config.dimension
        assert model.base_url == openai_embedding_config.base_url

    def test_extra_config_type(self, openai_embedding_config):
        """测试 extra_config 类型转换"""
        model = OpenAIEmbeddingModel(
            model_name=openai_embedding_config.model_name,
            model_type=openai_embedding_config.type.value,
            dimension=openai_embedding_config.dimension,
            api_key=openai_embedding_config.api_key,
            base_url=openai_embedding_config.base_url,
            max_chunk_length=openai_embedding_config.max_chunk_length,
            batch_size=openai_embedding_config.batch_size,
            max_retries=openai_embedding_config.max_retries,
            timeout=openai_embedding_config.timeout,
            rate_limit=openai_embedding_config.rate_limit,
            extra_config=openai_embedding_config.extra_config,
        )

        # 检查类型
        assert isinstance(model.extra_config, OpenAIExtraConfig)
        # 检查字段
        assert model.extra_config.encoding_format == "float"

    def test_extra_config_with_extra_fields(self, openai_embedding_config_with_extra):
        """测试包含额外字段的 extra_config"""
        model = OpenAIEmbeddingModel(
            model_name=openai_embedding_config_with_extra.model_name,
            model_type=openai_embedding_config_with_extra.type.value,
            dimension=openai_embedding_config_with_extra.dimension,
            api_key=openai_embedding_config_with_extra.api_key,
            base_url=openai_embedding_config_with_extra.base_url,
            max_chunk_length=openai_embedding_config_with_extra.max_chunk_length,
            batch_size=openai_embedding_config_with_extra.batch_size,
            max_retries=openai_embedding_config_with_extra.max_retries,
            timeout=openai_embedding_config_with_extra.timeout,
            rate_limit=openai_embedding_config_with_extra.rate_limit,
            extra_config=openai_embedding_config_with_extra.extra_config,
        )

        # 检查类型
        assert isinstance(model.extra_config, OpenAIExtraConfig)
        # 检查字段
        assert model.extra_config.encoding_format == "float"
        assert model.extra_config.user == "test-user-123"

    def test_build_endpoint_url(self, openai_embedding_config):
        """测试端点URL构建"""
        model = OpenAIEmbeddingModel(
            model_name=openai_embedding_config.model_name,
            model_type=openai_embedding_config.type.value,
            dimension=openai_embedding_config.dimension,
            api_key=openai_embedding_config.api_key,
            base_url=openai_embedding_config.base_url,
            max_chunk_length=openai_embedding_config.max_chunk_length,
            batch_size=openai_embedding_config.batch_size,
            max_retries=openai_embedding_config.max_retries,
            timeout=openai_embedding_config.timeout,
            rate_limit=openai_embedding_config.rate_limit,
            extra_config=openai_embedding_config.extra_config,
        )

        endpoint_url = model.build_endpoint_url()
        # 检查端点是否包含正确的部分
        assert endpoint_url.endswith("/v1/embeddings")
        assert openai_embedding_config.base_url in endpoint_url

    def test_build_auth_headers(self, openai_embedding_config):
        """测试认证头构建"""
        model = OpenAIEmbeddingModel(
            model_name=openai_embedding_config.model_name,
            model_type=openai_embedding_config.type.value,
            dimension=openai_embedding_config.dimension,
            api_key=openai_embedding_config.api_key,
            base_url=openai_embedding_config.base_url,
            max_chunk_length=openai_embedding_config.max_chunk_length,
            batch_size=openai_embedding_config.batch_size,
            max_retries=openai_embedding_config.max_retries,
            timeout=openai_embedding_config.timeout,
            rate_limit=openai_embedding_config.rate_limit,
            extra_config=openai_embedding_config.extra_config,
        )

        auth_headers = model.build_auth_headers()
        assert "Authorization" in auth_headers
        assert auth_headers["Authorization"] == f"Bearer {openai_embedding_config.api_key}"

    def test_build_request_body(self, openai_embedding_config, sample_texts):
        """测试请求体构建"""
        model = OpenAIEmbeddingModel(
            model_name=openai_embedding_config.model_name,
            model_type=openai_embedding_config.type.value,
            dimension=openai_embedding_config.dimension,
            api_key=openai_embedding_config.api_key,
            base_url=openai_embedding_config.base_url,
            max_chunk_length=openai_embedding_config.max_chunk_length,
            batch_size=openai_embedding_config.batch_size,
            max_retries=openai_embedding_config.max_retries,
            timeout=openai_embedding_config.timeout,
            rate_limit=openai_embedding_config.rate_limit,
            extra_config=openai_embedding_config.extra_config,
        )

        request_body = model.build_request_body(sample_texts)
        assert "input" in request_body
        assert request_body["input"] == sample_texts
        assert request_body["model"] == openai_embedding_config.model_name
        assert request_body["encoding_format"] == "float"

    def test_parse_response(self, openai_embedding_config):
        """测试响应解析"""
        model = OpenAIEmbeddingModel(
            model_name=openai_embedding_config.model_name,
            model_type=openai_embedding_config.type.value,
            dimension=openai_embedding_config.dimension,
            api_key=openai_embedding_config.api_key,
            base_url=openai_embedding_config.base_url,
            max_chunk_length=openai_embedding_config.max_chunk_length,
            batch_size=openai_embedding_config.batch_size,
            max_retries=openai_embedding_config.max_retries,
            timeout=openai_embedding_config.timeout,
            rate_limit=openai_embedding_config.rate_limit,
            extra_config=openai_embedding_config.extra_config,
        )

        test_response = {
            "object": "list",
            "data": [
                {"embedding": [0.1, 0.2, 0.3], "index": 0},
                {"embedding": [0.4, 0.5, 0.6], "index": 1},
            ],
        }

        embeddings = model.parse_response(test_response)
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 3
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]

    def test_split_into_batches(self, openai_embedding_config, sample_texts):
        """测试分批逻辑"""
        model = OpenAIEmbeddingModel(
            model_name=openai_embedding_config.model_name,
            model_type=openai_embedding_config.type.value,
            dimension=openai_embedding_config.dimension,
            api_key=openai_embedding_config.api_key,
            base_url=openai_embedding_config.base_url,
            max_chunk_length=openai_embedding_config.max_chunk_length,
            batch_size=2,  # 设置为小批大小
            max_retries=openai_embedding_config.max_retries,
            timeout=openai_embedding_config.timeout,
            rate_limit=openai_embedding_config.rate_limit,
            extra_config=openai_embedding_config.extra_config,
        )

        batches = model.split_into_batches(sample_texts)
        assert len(batches) == 3
        assert len(batches[0]) == 2
        assert len(batches[2]) == 1

    def test_extract_error_message(self, openai_embedding_config):
        """测试错误信息提取"""
        model = OpenAIEmbeddingModel(
            model_name=openai_embedding_config.model_name,
            model_type=openai_embedding_config.type.value,
            dimension=openai_embedding_config.dimension,
            api_key=openai_embedding_config.api_key,
            base_url=openai_embedding_config.base_url,
            max_chunk_length=openai_embedding_config.max_chunk_length,
            batch_size=openai_embedding_config.batch_size,
            max_retries=openai_embedding_config.max_retries,
            timeout=openai_embedding_config.timeout,
            rate_limit=openai_embedding_config.rate_limit,
            extra_config=openai_embedding_config.extra_config,
        )

        test_error = {"error": {"message": "Invalid request", "type": "invalid_request_error"}}
        error_message = model.extract_error_message(test_error)
        assert error_message == "Invalid request"

    def test_should_retry(self, openai_embedding_config):
        """测试重试逻辑"""
        model = OpenAIEmbeddingModel(
            model_name=openai_embedding_config.model_name,
            model_type=openai_embedding_config.type.value,
            dimension=openai_embedding_config.dimension,
            api_key=openai_embedding_config.api_key,
            base_url=openai_embedding_config.base_url,
            max_chunk_length=openai_embedding_config.max_chunk_length,
            batch_size=openai_embedding_config.batch_size,
            max_retries=openai_embedding_config.max_retries,
            timeout=openai_embedding_config.timeout,
            rate_limit=openai_embedding_config.rate_limit,
            extra_config=openai_embedding_config.extra_config,
        )

        # 429 应该重试
        assert model.should_retry(429, 0) is True
        assert model.should_retry(429, 3) is False  # 超过重试次数

        # 500 应该重试
        assert model.should_retry(500, 0) is True

        # 400 不应该重试
        assert model.should_retry(400, 0) is False

    def test_get_retry_delay(self, openai_embedding_config):
        """测试重试延迟"""
        model = OpenAIEmbeddingModel(
            model_name=openai_embedding_config.model_name,
            model_type=openai_embedding_config.type.value,
            dimension=openai_embedding_config.dimension,
            api_key=openai_embedding_config.api_key,
            base_url=openai_embedding_config.base_url,
            max_chunk_length=openai_embedding_config.max_chunk_length,
            batch_size=openai_embedding_config.batch_size,
            max_retries=openai_embedding_config.max_retries,
            timeout=openai_embedding_config.timeout,
            rate_limit=openai_embedding_config.rate_limit,
            extra_config=openai_embedding_config.extra_config,
        )

        # 指数退避：1, 2, 4, 8...
        assert model.get_retry_delay(0) == 1
        assert model.get_retry_delay(1) == 2
        assert model.get_retry_delay(2) == 4
        assert model.get_retry_delay(3) == 8
