# Knowledge Base V2 Architecture

## 1. Service responsibility

Knowledge Base owns the complete knowledge-base lifecycle:

- Collection and file-source management.
- Document ingestion from uploads and external URIs.
- Parsing, normalization, chunking, metadata extraction and deduplication.
- Embedding generation and sparse/dense/hybrid indexing.
- Workflow execution, retries, status tracking and document maintenance.
- LLM, embedding, index backend and storage provider configuration.
- Future retrieval-augmented chat (RAG Chat, TODO).

Identity lifecycle is explicitly outside this service. Pero is the only user center.

## 2. External identity contract

1. A client sends `Authorization: Bearer <access_token>`.
2. The Knowledge Base authentication dependency calls Pero:
   `GET {user_center.base_url}/api/oauth2/userinfo`.
3. Pero validates the access token and returns OIDC claims. The `openid` scope is required.
4. Knowledge Base reads only `sub`, validates it as a UUID, and creates a minimal `CurrentUser(id)`.
5. The UUID is used as the owner key for collection and file-source queries and writes.

Knowledge Base must not:

- Create local account, role, permission, password, login, session or token tables.
- Issue, decode, cache, refresh or revoke user tokens.
- Trust a user ID supplied in request body, query parameters or headers other than the verified Pero response.
- Persist profile claims such as email, phone or display name.

A timeout or 5xx from Pero is a 503 dependency failure; an invalid token is 401; a valid token without the required scope is 403. Tokens and Authorization headers must never be logged.

## 3. Data ownership

- `Collection.user_id`: UUID subject from Pero.
- `FileSource.user_id`: UUID subject from Pero; null means a shared/public file source.
- No foreign key is created to Pero's database. Both services retain independent databases and deployability.
- Collection reads may include the caller's own collections, explicitly public collections and legacy ownerless collections.
- Collection updates/deletes are owner-only.
- Document management is scoped to collections owned by the current subject.
- File-source reads may include the caller's own and shared file sources; mutations are owner-only.
- The client cannot assign or overwrite `user_id`.

The old `tenant_id` and `role_id` collection columns are removed. Tenant and role semantics belong to an upstream identity/authorization service and are not duplicated here.

## 4. Application layers

| Layer | Responsibility |
|---|---|
| `api/` | HTTP endpoints, request validation, API versioning and response mapping |
| `service/` | Knowledge-base domain services and workflow orchestration |
| `ext/` | Pluggable parser, chunker, embedding, index, file-source, workflow and infrastructure adapters |
| `core/` | Application lifecycle, middleware, exception mapping, logging and shared response types |
| `config/` | Typed configuration and extension registry |

Dependencies point inward: API routes call services; services depend on extension interfaces/factories; extension providers do not own HTTP identity or user lifecycle.

## 5. Extension design

Keep the existing extension registry and provider factories. New implementations should register through the existing factory/registry patterns rather than adding provider-specific conditionals to API routes.

Supported extension points include:

- Document parser engines and processors.
- Text chunking strategies.
- Embedding providers.
- Indexing providers (dense, sparse and hybrid).
- File-source providers.
- Workflow tasks and schedulers.
- Redis, HTTPX, Tortoise ORM and Celery infrastructure.

## 6. Knowledge-base chat (TODO)

The reserved endpoint is `POST /v1/chat/completions` and currently returns HTTP 501. Before implementation, define:

- Collection selection and owner/public visibility enforcement.
- Retrieval strategy (dense, sparse or hybrid), top-k and reranking.
- Conversation/session persistence and retention.
- Prompt construction, context-window budgeting and citation references.
- Streaming and non-streaming response contracts.
- LLM routing, timeout, retry and usage telemetry.

The chat service must reuse the same verified `CurrentUser.id` boundary and must not introduce a second user or authorization subsystem.

## 7. Configuration and deployment

Configure the knowledge-base database and infrastructure under `extensions`, and configure the external identity endpoint under `user_center`:

```yaml
user_center:
  base_url: "https://pero.example.com"
  userinfo_path: "/api/oauth2/userinfo"
  timeout_seconds: 5
```

Use HTTPS in production. The HTTPX client is managed by the existing extension lifecycle. The ORM is configured with only the `knowledge_base` connection.
