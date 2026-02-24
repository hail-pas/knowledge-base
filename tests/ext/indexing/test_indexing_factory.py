"""IndexModelFactory 测试用例

测试动态创建 IndexModel 类的功能
"""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock
from pydantic import Field

from ext.indexing import IndexModelFactory, BaseIndexModel
from ext.indexing.base import BaseProvider
from ext.indexing.types import DenseSearchClause, FilterClause
from ext.ext_tortoise.enums import EmbeddingModelTypeEnum
from ext.ext_tortoise.models.knowledge_base import EmbeddingModelConfig

from datetime import datetime


class MockProvider(BaseProvider):
    """模拟 Provider"""

    def __init__(self):
        from ext.indexing.providers.types import ElasticsearchConfig

        self.config = None
        self.extra_config = ElasticsearchConfig()
        self._client = None

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def create_collection(self, model_class, drop_existing=False):
        pass

    async def drop_collection(self, model_class):
        pass

    async def get(self, model_class, ids):
        return []

    async def filter(self, model_class, filter_clause, limit=10, offset=0, sort=None):
        return []

    async def insert(self, model_class, documents):
        return None

    async def update(self, model_class, documents):
        return []

    async def delete(self, model_class, ids):
        pass

    async def delete_by_query(self, model_class, filter_clause):
        pass

    async def count(self, model_class, filter_clause):
        return 0

    async def search(self, model_class, query_clause, filter_clause=None, limit=10, offset=0):
        return []

    async def search_cursor(self, model_class, query_clause, filter_clause=None, page_size=100, cursor=None):
        return [], None

    async def bulk_upsert(self, model_class, documents):
        return None

    async def health_check(self):
        return True


class ChunkIndex(BaseIndexModel):
    """测试用的基础 IndexModel"""

    content: str
    document_id: int

    class Meta:
        index_name = "chunks"
        dense_vector_field = "dense_vector"
        dense_vector_dimension = None
        auto_generate_id = True
        provider = None


@pytest.fixture
async def embedding_configs():
    """创建测试用的 embedding 配置"""

    configs = {}

    configs["1536"] = EmbeddingModelConfig(
        name="test-model-1536",
        type=EmbeddingModelTypeEnum.openai,
        model_name="text-embedding-3-small",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        dimension=1536,
        max_chunk_length=8192,
        batch_size=100,
        max_retries=3,
        timeout=60,
        rate_limit=60,
    )

    configs["3072"] = EmbeddingModelConfig(
        name="test-model-3072",
        type=EmbeddingModelTypeEnum.openai,
        model_name="text-embedding-3-large",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        dimension=3072,
        max_chunk_length=8192,
        batch_size=100,
        max_retries=3,
        timeout=60,
        rate_limit=60,
    )

    configs["1024"] = EmbeddingModelConfig(
        name="test-model-1024",
        type=EmbeddingModelTypeEnum.openai,
        model_name="text-embedding-ada-002",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        dimension=1024,
        max_chunk_length=8192,
        batch_size=100,
        max_retries=3,
        timeout=60,
        rate_limit=60,
    )

    return configs


@pytest.fixture
def mock_provider():
    """创建模拟 Provider"""
    return MockProvider()


class TestIndexModelFactory:
    """测试 IndexModelFactory"""

    def test_create_dynamic_class_with_dimension_1536(self, mock_provider, embedding_configs):
        """测试创建维度为 1536 的动态模型"""
        base_model = ChunkIndex
        base_model.Meta.provider = mock_provider

        emb_config = embedding_configs["1536"]

        dynamic_model = IndexModelFactory.create_for_embedding(base_model, emb_config)

        assert dynamic_model.Meta.dense_vector_dimension == 1536
        assert dynamic_model.Meta.index_name == "chunks"
        assert dynamic_model.Meta.provider is not None

    def test_create_dynamic_class_with_dimension_3072(self, mock_provider, embedding_configs):
        """测试创建维度为 3072 的动态模型"""
        base_model = ChunkIndex
        base_model.Meta.provider = mock_provider

        emb_config = embedding_configs["3072"]

        dynamic_model = IndexModelFactory.create_for_embedding(base_model, emb_config)

        assert dynamic_model.Meta.dense_vector_dimension == 3072

    def test_caching_mechanism(self, mock_provider, embedding_configs):
        """测试缓存机制 - 相同配置返回同一类"""
        base_model = ChunkIndex
        base_model.Meta.provider = mock_provider

        emb_config = embedding_configs["1536"]

        model1 = IndexModelFactory.create_for_embedding(base_model, emb_config)
        model2 = IndexModelFactory.create_for_embedding(base_model, emb_config)

        assert model1 is model2

    def test_different_dimensions_create_different_classes(self, mock_provider, embedding_configs):
        """测试不同维度创建不同的类"""
        base_model = ChunkIndex
        base_model.Meta.provider = mock_provider

        emb_config_1536 = embedding_configs["1536"]
        emb_config_3072 = embedding_configs["3072"]

        model_1536 = IndexModelFactory.create_for_embedding(base_model, emb_config_1536)
        model_3072 = IndexModelFactory.create_for_embedding(base_model, emb_config_3072)

        assert model_1536 is not model_3072
        assert model_1536.Meta.dense_vector_dimension == 1536
        assert model_3072.Meta.dense_vector_dimension == 3072

    def test_inheritance_from_base_model(self, mock_provider, embedding_configs):
        """测试动态模型继承自基础模型"""
        base_model = ChunkIndex
        base_model.Meta.provider = mock_provider

        emb_config = embedding_configs["1024"]

        dynamic_model = IndexModelFactory.create_for_embedding(base_model, emb_config)

        assert issubclass(dynamic_model, BaseIndexModel)
        assert issubclass(dynamic_model, ChunkIndex)

    def test_clear_cache(self, mock_provider, embedding_configs):
        """测试清除缓存"""
        base_model = ChunkIndex
        base_model.Meta.provider = mock_provider

        emb_config = embedding_configs["1536"]

        model1 = IndexModelFactory.create_for_embedding(base_model, emb_config)
        IndexModelFactory.clear_cache()

        model2 = IndexModelFactory.create_for_embedding(base_model, emb_config)

        assert model1 is not model2

    def test_get_registered_models(self, mock_provider, embedding_configs):
        """测试获取已注册的模型列表"""
        IndexModelFactory.clear_cache()

        base_model = ChunkIndex
        base_model.Meta.provider = mock_provider

        emb_config_1536 = embedding_configs["1536"]
        emb_config_3072 = embedding_configs["3072"]

        IndexModelFactory.create_for_embedding(base_model, emb_config_1536)
        IndexModelFactory.create_for_embedding(base_model, emb_config_3072)

        registered = IndexModelFactory.get_registered_models()

        assert "ChunkIndex_1536" in registered
        assert "ChunkIndex_3072" in registered

    def test_get_model(self, mock_provider, embedding_configs):
        """测试通过基础模型名和维度获取模型"""
        IndexModelFactory.clear_cache()

        base_model = ChunkIndex
        base_model.Meta.provider = mock_provider

        emb_config = embedding_configs["1536"]

        IndexModelFactory.create_for_embedding(base_model, emb_config)

        retrieved_model = IndexModelFactory.get_model("ChunkIndex", 1536)

        assert retrieved_model is not None
        assert retrieved_model.Meta.dense_vector_dimension == 1536

    def test_get_model_not_exists(self):
        """测试获取不存在的模型返回 None"""
        result = IndexModelFactory.get_model("NonExistentModel", 1234)
        assert result is None

    def test_invalid_dimension_raises_error(self, mock_provider):
        """测试无效维度抛出错误"""
        from ext.ext_tortoise.models.knowledge_base import EmbeddingModelConfig

        base_model = ChunkIndex
        base_model.Meta.provider = mock_provider

        with pytest.raises(ValueError, match="Invalid dimension"):
            emb_config = EmbeddingModelConfig(
                name="test",
                type=EmbeddingModelTypeEnum.openai,
                model_name="test",
                api_key="test",
                base_url="test",
                dimension=0,
            )
            IndexModelFactory.create_for_embedding(base_model, emb_config)

        with pytest.raises(ValueError, match="Invalid dimension"):
            emb_config = EmbeddingModelConfig(
                name="test",
                type=EmbeddingModelTypeEnum.openai,
                model_name="test",
                api_key="test",
                base_url="test",
                dimension=-1,
            )
            IndexModelFactory.create_for_embedding(base_model, emb_config)

    def test_provider_not_bound_raises_error(self, embedding_configs):
        """测试未绑定 provider 抛出错误"""
        base_model = ChunkIndex
        base_model.Meta.provider = None

        emb_config = embedding_configs["1536"]

        with pytest.raises(RuntimeError, match="Meta.provider is not bound"):
            IndexModelFactory.create_for_embedding(base_model, emb_config)

    def test_wrong_config_type_raises_error(self, mock_provider):
        """测试传入错误的配置类型抛出错误"""
        base_model = ChunkIndex
        base_model.Meta.provider = mock_provider

        with pytest.raises(TypeError, match="must be EmbeddingModelConfig"):
            IndexModelFactory.create_for_embedding(base_model, "not-a-config")

    def test_dynamic_model_can_be_instantiated(self, mock_provider, embedding_configs):
        """测试动态模型可以正常实例化"""
        base_model = ChunkIndex
        base_model.Meta.provider = mock_provider

        emb_config = embedding_configs["1536"]

        dynamic_model = IndexModelFactory.create_for_embedding(base_model, emb_config)

        instance = dynamic_model(
            content="test content",
            document_id=123,
        )

        assert instance.content == "test content"
        assert instance.document_id == 123
        assert dynamic_model.Meta.dense_vector_dimension == 1536

    def test_model_key_generation(self):
        """测试模型 key 生成逻辑"""
        key = IndexModelFactory._get_model_key("ChunkIndex", 1536)
        assert key == "ChunkIndex_1536"

        key = IndexModelFactory._get_model_key("DocumentIndex", 3072)
        assert key == "DocumentIndex_3072"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
