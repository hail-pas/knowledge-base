# Indexing backend selection and lifecycle

The indexing module keeps the existing `BaseIndexModel`, `IndexModelFactory`,
`IndexingProviderFactory`, and `CollectionIndexModelHelper` public APIs.

## Backend resolution

1. If a collection has `indexing_backend_config_id`, its enabled backend is used.
2. Otherwise the active default backend (`is_enabled=True`, `is_default=True`)
   is used at application registration.
3. If no default backend exists, the existing knowledge-base PostgreSQL
   connection is used as a compatibility fallback.

Supported registered providers: PostgreSQL, Elasticsearch, and Milvus.

### Switching a collection

Change the collection's `indexing_backend_config_id` to an enabled backend
configuration, then rebuild/reindex its data on that backend. The helper
resolves the backend per collection and caches provider connections by effective
connection settings. No global `Meta.provider` mutation is needed for a
collection-specific backend.

**Important:** Switching the configuration does **not** migrate stored index
data automatically. Reindex before routing production traffic to the new
backend, and verify document counts and retrieval quality. Keep the original
index until validation and cutover are complete.

### Changing the default

Set exactly one enabled, non-deleted indexing backend configuration as default
and restart/re-register the indexing extension to update the default model
bindings. Explicit collection bindings are unaffected.

## Schema lifecycle

Dynamic models are scoped to their provider instance and embedding dimension.
Their schema is initialized once per process per model identity; concurrent
initializers share the same lock. Use `IndexModelFactory.clear_cache()` to
invalidate the in-process model/schema cache when manually recreating a schema.
Provider connections are closed by
`await IndexingProviderFactory.clear_cache()` during shutdown.

## Compatibility and limitations

- Existing dense, sparse, hybrid retrieval and CRUD contracts are preserved.
- A backend change is a data migration, not just a configuration update.
- Existing requests may finish on an old provider during a configuration
  change; coordinate cutover and clean up old connections at a safe point.
- Set only one default backend at a time; multiple defaults should be prevented
  at the configuration-management layer.
