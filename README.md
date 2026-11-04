# Knowledge Base Service

Knowledge Base 是独立的知识库服务，负责知识库全生命周期管理、文档处理与索引，并预留基于知识库的检索增强聊天（RAG Chat）。

## 服务边界

- **Pero 是唯一用户中心**：负责用户注册、登录、密码、会话、Token 签发、用户资料与用户生命周期。
- **Knowledge Base 不维护用户账户**：不保存用户名、手机号、邮箱、密码、用户状态等账户信息，不提供用户 CRUD 或登录接口。
- **Knowledge Base 保留本地 RBAC**：继续管理 Role、Permission、Resource（菜单/按钮/API 资源），负责角色分配、权限校验和菜单管理。
- **身份关联只使用 UUID**：Knowledge Base 调用 Pero OIDC UserInfo 校验 Bearer Token，只读取 `sub` UUID；通过本地 `user_role` 关联表将该 UUID 关联到 Knowledge Base 的角色。
- **认证与授权分离**：Pero 确认“用户是谁”；Knowledge Base 本地 RBAC 决定“用户能做什么”。
- 访问令牌需要包含 `openid` scope；Knowledge Base 不自行解析、签发、刷新或撤销 Token。

## 核心能力

1. **知识库集合管理**：集合创建、编辑、查询、删除、公开集合只读访问。
2. **文档全流程管理**：上传或 URI 导入、解析、清洗、切块、Embedding、索引、重试、重切块、页面与 Chunk 管理、FAQ 管理。
3. **模型与基础设施配置**：LLM、Embedding、索引后端、文件源配置。
4. **本地权限管理**：角色管理、API 权限编码同步、菜单/资源树管理、资源与权限绑定、Pero UUID 与本地角色分配。
5. **可扩展处理引擎**：通过 extensions 注册文档解析器、文本切块策略、Embedding Provider、索引后端、文件存储 Provider 和工作流。
6. **知识库聊天**：规划中（TODO），后续基于集合检索、权限过滤、上下文组装和 LLM 生成实现。

## 认证与权限链路

```text
客户端 Bearer Token
        |
        v
Knowledge Base -> Pero OIDC UserInfo
        |
        | 校验 Token，读取 sub UUID
        v
CurrentUser(id=Pero UUID)
        |
        v
本地 user_role 关联表
        |
        v
Role -> Resource -> Permission
        |
        v
API 权限校验 / 菜单资源返回
```

Knowledge Base 不创建 User/Account 模型，也不在本地保存用户资料。唯一用户关联键是 Pero 返回的 UUID subject。

## 本地 RBAC 模型

| 模型 | 职责 |
|---|---|
| `Role` | 本地角色定义，可绑定多个资源；支持系统超级管理员角色标识 |
| `Permission` | API 权限编码，由受保护的 API 路由同步生成 |
| `Resource` | 菜单、按钮、API 资源树，可绑定 Permission 和 Role |
| `UserRole` | 仅保存 `user_id(UUID)` 与 `role_id` 的关联，不保存用户资料 |

权限判定链路为 `Pero sub UUID -> UserRole -> Role -> Resource -> Permission`。超级管理员能力由本地 Role 标记，不依赖 Pero 用户表字段。

## 关键 API

- `/knowledge/v1/role`：角色列表、详情、创建、更新、删除。
- `/knowledge/v1/resource`：资源/菜单创建、更新、删除和管理资源树查询；`GET /resource/my-trees` 返回当前用户通过角色获得的菜单资源。
- `/knowledge/v1/permission`：权限编码查询、废弃状态管理。
- `/knowledge/v1/user-role/{user_id}/roles`：按 Pero UUID 查询或替换角色分配。
- `/knowledge/v1/collection`、`/document`、`/config`：知识库业务接口，统一经过本地 RBAC 权限校验。
- `/knowledge/v1/chat/completions`：预留接口，当前返回 HTTP 501。

API 权限编码由 `deploy/bin/permission/menu_perm.py` 根据受保护路由同步；菜单结构可在 `deploy/bin/permission/data.py` 中配置。

## 目录结构

```text
api/                         HTTP 路由与 API 版本
  entrypoint/                ASGI 应用入口
  knowledge_base/            知识库与本地 RBAC API
service/                     领域服务与业务编排
  collection/                集合生命周期
  document/                  文档生命周期
  role/                      角色 schema
  resource/                  菜单/资源 schema 与树处理
  permission/                权限 schema
  user_role/                 外部 UUID 与本地角色关联 schema
  workflow/                  文档处理工作流
ext/                         可插拔基础能力
  document_parser/           文档解析
  text_chunker/              文本切块
  embedding/                 向量化 Provider
  indexing/                  向量/关键词索引 Provider
  file_source/               文件存储 Provider
  workflow/                  通用工作流引擎
  ext_tortoise/              ORM 与迁移（知识库业务 + 本地 RBAC）
  ext_redis/                 Redis 连接扩展
  ext_celery/                异步任务扩展
  ext_httpx/                 HTTP 客户端扩展
core/                        应用生命周期、响应、中间件、异常处理
config/                      类型化配置与扩展注册
docs/                        架构和集成说明
tests/                       单元与集成测试
```

## 运行环境

- Python 3.13+
- uv
- PostgreSQL 16+
- Redis
- Celery Broker / Result Backend（异步任务需要）
- Pero 用户中心（OIDC UserInfo）

## 配置

复制 `etc/template.yaml` 为当前环境配置（例如 `etc/local.yaml`），配置知识库数据库、Redis、Celery、HTTPX 和 Pero：

```yaml
extensions:
  tortoise:
    authorization:
      url: "postgresql://username:password@127.0.0.1:5432/knowledge_base_authorization"
    knowledge_base:
      url: "postgresql://username:password@127.0.0.1:5432/knowledge_base"

user_center:
  base_url: "https://pero.example.com"
  userinfo_path: "/api/oauth2/userinfo"
  timeout_seconds: 5
```

生产环境必须通过 HTTPS 访问 Pero；不要在 Knowledge Base 配置或日志中保存用户密码或访问令牌。

## 启动与检查

```bash
make setup
make server
make check
```

首次运行测试前，请在 PostgreSQL 中创建 `knowledge_base_test` 数据库，并将 `tests/.env.template` 复制为 `tests/.env`，配置 `TEST_DATABASE_URL`。

```bash
source tests/.env && uv run pytest
```

默认 ASGI 入口：`api.entrypoint.factory:service_api`。

- 服务文档：`/knowledge/docs/`
- 知识库 API：`/knowledge/v1/`
- 健康检查：`/knowledge/health`

## 首个管理员初始化

首次部署时，使用 Pero 中已存在的用户 UUID（OIDC `sub`）初始化 Knowledge Base 本地超级管理员角色：

```bash
uv run python deploy/bin/authorization/bootstrap_super_admin.py --user-id <PERO_USER_UUID>
```

该脚本只创建本地 Role 和 UserRole 关联，不创建、修改或查询 Pero 用户账户。后续角色分配由本地 `/user-role` 管理接口维护。

首次部署时，需要在同一个 PostgreSQL 实例中创建 `knowledge_base` 和 `knowledge_base_authorization` 两个数据库。迁移脚本会在 PostgreSQL 中创建对应业务表和本地 RBAC 表。若原系统数据仍在 MySQL 中，不能直接将 MySQL 数据库地址配置为 PostgreSQL；需要先将 Role、Permission、Resource 及关联表导入 PostgreSQL，再根据 Pero 的真实 UUID 重建 UserRole 关联。该过程不会迁移或读取旧 Account 用户表。

## 数据归属约定

- `Collection.user_id` 与 `FileSource.user_id` 存储 Pero 返回的 UUID subject。
- 用户 ID 只能从已验证的 Token 获取，禁止从请求 Body、Query 或客户端可控字段覆盖。
- `UserRole.user_id` 同样只能使用经过 Pero 校验得到的 UUID；不建立指向 Pero 数据库的外键。
- 未归属用户的文件源可作为公共文件源读取；公共集合可读取，但只有集合所有者可以修改或删除。
- Pero、Knowledge Base 业务库和 Knowledge Base 本地 authorization 库保持独立的数据边界。
