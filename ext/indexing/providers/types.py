"""Provider 配置类型定义"""

from pydantic import BaseModel, Field
from typing import Optional


class ProviderConfig(BaseModel):
    """Provider 配置基类（所有 Provider 特定配置的父类）"""

    partition_value: Optional[str] = Field(default=None, description="Partition 值")


class ElasticsearchConfig(ProviderConfig):
    """Elasticsearch 特定配置"""

    number_of_shards: int = Field(default=3, description="分片数")
    number_of_replicas: int = Field(default=2, description="副本数")

    vector_similarity: str = Field(default="cosine", description="向量相似度：cosine/l2/ip")

    text_analyzer: str = Field(default="ik_smart", description="文本分析器")
    search_analyzer: str = Field(default="ik_smart", description="文本分析器")


class MilvusConfig(ProviderConfig):
    """Milvus 特定配置"""

    db_name: str = Field(default="default", description="数据库名称")

    index_type: str = Field(default="HNSW", description="索引类型：HNSW/IVF_FLAT")
    metric_type: str = Field(default="COSINE", description="距离度量：COSINE/L2/IP")

    M: int = Field(default=16, description="HNSW M 参数")
    ef_construction: int = Field(default=64, description="HNSW ef_construction 参数")

    inverted_index_algo: str = Field(default="DAAT_MAXSCORE", description="距离度量：COSINE/L2/IP")
