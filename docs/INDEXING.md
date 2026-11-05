# Indexing Architecture

## Default backend

The default indexing provider is PostgreSQL. It reuses the knowledge-base PostgreSQL DSN when no enabled default `IndexingBackendConfig(type="postgresql")` exists. An explicit backend record can override the connection and `extra_config`.

The PostgreSQL provider requires the server-side `vector` extension to be installed and the application role to have permission to run `CREATE EXTENSION IF NOT EXISTS vector` (or for an administrator to install it beforehand).

## Separation of concerns

- **Index model** describes typed document fields and logical index identity.
- **Provider** owns physical storage, schema creation, CRUD, filtering, and backend-specific query execution.
- **Search clause** describes dense, sparse, or hybrid retrieval without exposing backend query syntax.
- **Embedding configuration** supplies the vector dimension. Dynamic model classes are cached by base model, dimension, and provider identity.

A new storage backend should implement `BaseProvider` and register itself with `IndexingProviderFactory`. It should not add backend-specific branches to document workflows.

A new retrieval mode should add a backend-neutral query clause and a clear capability contract. Backend-specific optimizations can remain inside a provider; portable composition such as reciprocal-rank fusion belongs in the shared retrieval layer.

## PostgreSQL implementation

- Dense vectors use pgvector `vector(n)` through 2,000 dimensions and `halfvec(n)` from 2,001 through 4,000 dimensions.
- Dense search supports cosine distance, L2 distance, and inner product.
- HNSW is the default approximate index; IVFFlat can be selected in provider `extra_config`.
- Keyword search uses PostgreSQL full-text search (`to_tsvector`, `plainto_tsquery`, `ts_rank_cd`) with a GIN expression index on the selected text field.
- Hybrid search retrieves candidates from dense and keyword search independently and fuses ranks with weighted Reciprocal Rank Fusion (RRF). Raw dense and FTS scores are not directly added because their scales are not comparable.
- JSON-like fields are stored as JSONB; scalar fields use explicit PostgreSQL scalar types.

Example provider `extra_config`:

```json
{
  "database": "knowledge_base",
  "schema": "public",
  "vector_metric": "cosine",
  "vector_index": "hnsw",
  "text_search_config": "simple",
  "pool_min_size": 1,
  "pool_max_size": 10
}
```

## Operational considerations

1. PostgreSQL's built-in `simple` text configuration does not perform Chinese word segmentation. For Chinese keyword relevance, deploy a suitable PostgreSQL text-search parser such as zhparser and configure an appropriate text-search configuration, or use Elasticsearch as the keyword-search backend.
2. Approximate vector indexes trade recall for latency. Benchmark recall and query latency with representative filters and tenant distributions before tuning HNSW/IVFFlat parameters.
3. pgvector applies filters during SQL query execution, but approximate-index filtering can reduce the number of returned candidates. Validate filtered recall and tune iterative scans / candidate limits when needed.
4. Changing embedding dimensions creates a distinct physical index name; it does not migrate or re-embed existing vectors. Re-index data before switching production traffic to a new embedding model.
5. Graph retrieval is not implemented by this provider. The query type can be extended later, but graph storage/traversal should be introduced as a distinct capability rather than simulated through vector search.
