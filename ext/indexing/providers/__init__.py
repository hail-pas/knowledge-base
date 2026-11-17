"""Backend adapters are imported lazily by IndexingProviderFactory.

Import explicit implementations from their concrete modules when required:
`from ext.indexing.providers.postgresql import PostgreSQLProvider`.
"""
from ext.indexing.base import BaseProvider
from ext.indexing.providers.types import ProviderConfig, ElasticsearchConfig, MilvusConfig

__all__ = ["BaseProvider", "ProviderConfig", "ElasticsearchConfig", "MilvusConfig"]
