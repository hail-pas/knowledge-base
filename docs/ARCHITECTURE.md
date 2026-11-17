# Knowledge Base V2 Architecture

## 1. Service responsibility

Knowledge Base owns the complete knowledge-base lifecycle and its own authorization policy:

- Collection and file-source management.
- Document ingestion from uploads and external URIs.
- Parsing, normalization, chunking, metadata extraction and deduplication.
- Embedding generation and sparse/dense/hybrid indexing.
- Workflow execution, retries, status tracking and document maintenance.
- LLM, embedding, index backend and storage provider configuration.
- Local Role / Permission / Resource RBAC.
- Future retrieval-augmented chat (RAG Chat, TODO).

Pero is the only user center. Knowledge Base must not duplicate user account lifecycle or user profile data.

## 2. Authentication and identity boundary

1. A client sends `Authorization: Bearer <access_token>`.
2. Knowledge Base calls Pero: `GET {user_center.base_url}/api/oauth2/userinfo`.
3. Pero validates the token and returns OIDC claims; the `openid` scope is required.
4. Knowledge Base reads only `sub`, validates it as a UUID, and creates `CurrentUser(id)`.
5. The UUID is used for collection/file-source ownership and local role assignment.

Knowledge Base must not:

- Create or maintain User/Account models or user CRUD APIs.
- Store username, phone, email, password, user status, login history or profile claims.
- Issue, decode, cache, refresh or revoke user access tokens.
- Trust a user ID supplied by request body, query parameters or client-controlled headers.
- Create foreign keys to Pero's database.

A timeout or 5xx from Pero is a 503 dependency failure; an invalid token is 401; a valid token without the required scope is 403. Tokens and Authorization headers must never be logged.

## 3. Local RBAC boundary

User identity and authorization policy are intentionally separated:

- **Pero** answers: who is the caller? It owns accounts and token lifecycle.
- **Knowledge Base** answers: what can this caller do? It owns Role, Permission, Resource, menu definitions, permission assignments and authorization decisions.

### 3.1 Data model

| Model | Responsibility |
|---|---|
| `Role` | Local role definition, including `is_super_admin`; a role can own many resources |
| `Permission` | API permission code, label, type and deprecation state |
| `Resource` | Hierarchical menu/button/API resource; links to roles and permissions |
| `UserRole` | External identity mapping only: `user_id(UUID)` + `role_id`; no user profile fields |

The data relationship is:

```text
Pero OIDC sub (UUID)
        |
        v
UserRole(user_id, role_id)
        |
        v
Role
        |
        v
Resource (menu / button / API)
        |
        v
Permission (API permission code)
```

No User or Account table is created. The `user_role` table does not have a foreign key to Pero; it stores only the external UUID and a local Role foreign key.

### 3.2 Authorization decision

For each protected route:

1. Validate Bearer Token through Pero and resolve the UUID subject.
2. Load the local role assignments from `UserRole`.
3. If any assigned role is marked `is_super_admin`, grant the super-admin wildcard.
4. Otherwise, resolve enabled/assignable resources assigned to those roles.
5. Resolve non-deprecated permission codes attached to those resources.
6. Match the required code: `{app_code}:{HTTP_METHOD}:{root_path}{route_path}`, `{app_code}:*` or `*`.
7. Deny with 403 if no matching permission exists.

All Knowledge Base business routers and RBAC management routers use `api_permission_check`. Health checks remain public.

### 3.3 Role assignment

- `PUT /v1/user-role/{user_id}/roles` replaces the role assignment for a Pero UUID.
- `GET /v1/user-role/{user_id}/roles` reads its local role assignments.
- These endpoints do not list, create, edit or delete Pero users.
- The first local super-admin is bootstrapped from a known Pero UUID by `deploy/bin/authorization/bootstrap_super_admin.py`.

### 3.4 API permission and menu management

- `Permission` rows are synchronized from protected API routes by `deploy/bin/permission/menu_perm.py`.
- `Resource` manages menu/resource trees and binds API permission codes.
- `Role` binds one or more resources.
- `UserRole` maps an external Pero UUID to one or more local roles.
- Deprecated permission codes remain visible for audit but are not granted during authorization.

## 4. Knowledge-base data ownership

- `Collection.user_id`: UUID subject from Pero.
- `FileSource.user_id`: UUID subject from Pero; null means a shared/public file source.
- Collection reads may include the caller's own collections, explicitly public collections and legacy ownerless collections.
- Collection updates/deletes are owner-only.
- Document management is scoped to collections owned by the current subject.
- File-source reads may include the caller's own and shared file sources; mutations are owner-only.
- A shared file source (`user_id IS NULL`) must be an S3/MinIO/OSS source with an `allowed_prefix_template` containing `{user_id}` or `{collection_id}`. Upload, URI import, document access and cleanup all enforce the resolved prefix; missing prefixes fail closed.
- The client cannot assign or overwrite `user_id`.
- The former `tenant_id` and collection-level `role_id` fields are removed. Authorization is centralized in local RBAC rather than duplicated as collection columns.

## 5. Application layers

| Layer | Responsibility |
|---|---|
| `api/` | HTTP routes, request validation, API versioning and response mapping |
| `service/` | Knowledge-base domain services, RBAC schemas and workflow orchestration |
| `ext/` | Pluggable parser, chunker, embedding, index, file-source, workflow and infrastructure adapters; ORM models |
| `core/` | Application lifecycle, middleware, exception mapping, logging and shared response types |
| `config/` | Typed configuration and extension registry |

Dependencies point inward: API routes call services; services depend on extension interfaces/factories; extension providers do not own HTTP identity or token lifecycle.

## 6. Extension design

Keep the existing extension registry and provider factories. New implementations should register through the existing factory/registry patterns rather than adding provider-specific conditionals to API routes.

Supported extension points include:

- Document parser engines and processors.
- Text chunking strategies.
- Embedding providers.
- Indexing providers (dense, sparse and hybrid).
- File-source providers.
- Workflow tasks and schedulers.
- Redis, HTTPX, Tortoise ORM and Celery infrastructure.

## 7. Knowledge-base chat (TODO)

The reserved endpoint is `POST /v1/chat/completions` and currently returns HTTP 501. Before implementation, define:

- Collection selection and owner/public visibility enforcement.
- Retrieval strategy (dense, sparse or hybrid), top-k and reranking.
- Conversation/session persistence and retention.
- Prompt construction, context-window budgeting and citation references.
- Streaming and non-streaming response contracts.
- LLM routing, timeout, retry and usage telemetry.

Chat must reuse the same Pero UUID and local RBAC boundary; it must not introduce a second user or authorization subsystem.

## 8. Configuration and deployment

Configure the Knowledge Base database and infrastructure under `extensions`, and configure the external identity endpoint under `user_center`. For a new PostgreSQL deployment, create both databases in the same PostgreSQL instance:

```yaml
extensions:
  rdb_authorization:
    url: "postgresql://username:password@127.0.0.1:5432/knowledge_base_authorization"
  rdb_knowledge_base:
    url: "postgresql://username:password@127.0.0.1:5432/knowledge_base"

user_center:
  base_url: "https://pero.example.com"
  userinfo_path: "/api/oauth2/userinfo"
  timeout_seconds: 5
```

Use HTTPS in production. The HTTPX client is managed by the existing extension lifecycle. The ORM uses two PostgreSQL database connections on the same PostgreSQL instance: `knowledge_base` for knowledge-base domain tables and `authorization` for Role/Permission/Resource/UserRole. Both use the asyncpg driver. It never connects to Pero's database.

Set `KB_SECRET_ENCRYPTION_KEY` to a stable 32-byte hex key (generate with `openssl rand -hex 32`) in every API and Worker instance. Credential fields and external collection authorization are encrypted with AES-256-GCM before persistence; do not commit the key or rotate it without a dedicated re-encryption procedure. A shared file source must declare a user- or collection-scoped `allowed_prefix_template`; shared-source URI access fails closed when the prefix is absent or mismatched.
