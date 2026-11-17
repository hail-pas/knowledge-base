# Indexing architecture and switching

## Design

```text
Collection / Document services
  └── CollectionIndexModelHelper
       ├── IndexingProviderFactory  (backend resolution, lazy client, cache, lifecycle)
       └── IndexModelFactory       (provider-scoped model, dimension, schema setup)
            └── BaseIndexModel     (validated data, CRUD, batch writes)
                 └── IndexSearch  (dense/sparse dispatch, weighted RRF, cursor)
                      └── BaseProvider
                           ├── PostgreSQLProvider
                           ├── ElasticsearchProvider
                           └── MilvusProvider
```

The domain layer never selects a backend implementation. Backends are imported
only when selected; optional clients need not be installed for unused providers.
A concrete model class is scoped to exactly one provider instance and one vector
dimension, and has independent `Meta` configuration. Concurrent schema
initialization is coalesced once per model identity.

## Features

- Dense vector, sparse text and hybrid search.
- Hybrid mode always executes two backend-native searches and fuses them through
  weighted reciprocal-rank fusion (RRF, k=60). Raw scores are **not** combined,
  since Elasticsearch BM25, Milvus and pgvector use incompatible scales.
- Filtering supports equality, membership, range and nested AND/OR.
- Sorting accepts `field`, `-field`, `field:asc` or `field:desc`.
- One bounded batch writer implements insert, update and upsert semantics.
  Documents with generated IDs are inserted; an existing ID is upserted.
- Portable search cursors encode an offset (not a snapshot). They are suited to
  stable result sets; live index mutations may shift pages. Deep paging can be
  expensive, and backend limits still apply.
- Empty-filter delete-by-query is forbidden.

## Backend selection

**Collection override:** `collection.indexing_backend_config_id` takes
priority. A backend must exist, be enabled and not be deleted.

**Default:** if the collection has no explicit backend, the currently enabled
`is_default=True` configuration is resolved at request time. Exactly one
global default is allowed. If no default exists, the application PostgreSQL
connection is used as the fallback. Connections are pooled by the effective
connection settings, including credentials and backend-specific options.

**Change default:** `PUT /indexing-backend/{id}/default` (super-admin only).
It updates defaults transactionally. This is effective without restarting the
indexing extension, but only after existing unbound collections with documents
have been pinned to their current backend or migrated. Unbound non-empty
collections block switching to avoid silently routing their searches to a new,
empty backend.

**Pin a collection:** `PUT /collections/{id}` (use the project's collection
update route) with `indexing_backend_config_id` equal to the current default.
This is allowed even for non-empty collections because it does not change the
underlying backend.

## Switching a populated collection to a different engine

This operation is **not** a configuration-only update:

1. Create and health-check the target indexing backend configuration.
2. Quiesce index writes to that collection.
3. Rebuild content sparse, content dense and FAQ indexes on the target backend
   using the same embedding dimension, chunk metadata and source IDs.
4. Verify per-family counts, representative filtered queries, dense/sparse/hybrid
   retrieval, and destructive operations against a test dataset.
5. Update `indexing_backend_config_id`, resume writes, observe results.
6. Retain the previous index until the rollback window closes; delete it only
   after validation.

The Collection update endpoint intentionally rejects a direct backend change
for populated collections. A fully automated cross-engine data migration is
not provided by this module; do not bypass the check or silently claim that
changing a backend moves index data.

## Operational checks

```bash
uv run python -m compileall -q ext/indexing
uv run pytest -q --confcutdir=tests/ext/indexing \
  tests/ext/indexing/test_contract_refactor.py \
  tests/ext/indexing/test_provider_contracts.py \
  tests/ext/indexing/test_lifecycle_refactor.py \
  tests/ext/indexing/test_indexing_factory.py
```

The PR CI runs these pure contract tests without a database. Full integration
tests in `tests/ext/indexing/test_{postgresql,elasticsearch,milvus}.py`
require configured backend services and should be run before production rollout.

## Limitations and design decisions

- Hybrid fusion is intentionally identical across backends but **candidate
  recall** can differ due to text analyzers, indexing parameters and distances.
- The connection cache retains older versioned providers until shutdown so
  in-flight requests remain safe; close them via
  `await IndexingProviderFactory.clear_cache()` at maintenance boundaries.
- Schema creation is idempotent per model and process; explicitly invalidate
  `IndexModelFactory.clear_cache()` after out-of-band schema deletion.
- Offset-based filtering/searching is not a substitute for backend-specific
  snapshot/scan APIs for massive, concurrently changing data.
