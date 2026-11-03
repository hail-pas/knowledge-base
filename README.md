# Knowledge Base Service

Knowledge Base 是独立的知识库服务，专注于知识库全生命周期管理、文档处理与索引，以及后续基于知识库的检索增强聊天（RAG Chat）。

## 服务边界

- 用户身份、注册、登录、密码、会话、Token 签发和用户资料统一由外部 **Pero** 用户中心负责。
- Knowledge Base 不保存用户账户、密码、角色或权限数据。
- Knowledge Base 只从 Pero 的 OIDC UserInfo 接口取得 `sub`（UUID），将其作为知识库资源的 `user_id` 进行数据归属和隔离。
- 当前认证链路：客户端携带 Bearer Token → Knowledge Base 调用 Pero `GET /api/oauth2/userinfo` → 校验 Token 并读取 `sub` → 按 `user_id` 访问数据。
- 访问令牌需要包含 `openid` scope；Knowledge Base 不自行解析、签发或缓存 Token。

## 核心能力

1. **知识库集合管理**：集合创建、编辑、查询、删除、公开集合只读访问。
2. **文档全流程管理**：上传或 URI 导入、解析、清洗、切块、Embedding、索引、重试、重切块、页面与 Chunk 管理、FAQ 管理。
3. **模型与基础设施配置**：LLM、Embedding、索引后端、文件源配置。
4. **可扩展处理引擎**：通过 extensions 注册文档解析器、文本切块策略、Embedding Provider、索引后端、文件存储 Provider 和工作流。
5. **知识库聊天**：规划中（TODO），后续基于集合检索、权限过滤、上下文组装和 LLM 生成实现。

## 目录结构

```text
api/                         HTTP 路由与 API 版本
  entrypoint/                ASGI 应用入口
  knowledge_base/            知识库 API
service/                     知识库领域服务与业务编排
  collection/                集合生命周期
  document/                  文档生命周期
  workflow/                  文档处理工作流
  llm_model/                 模型配置与调用
ext/                         可插拔基础能力
  document_parser/           文档解析
  text_chunker/              文本切块
  embedding/                 向量化 Provider
  indexing/                  向量/关键词索引 Provider
  file_source/               文件存储 Provider
  workflow/                  通用工作流引擎
  ext_tortoise/              ORM 与迁移
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
- MySQL
- Redis
- Celery Broker / Result Backend（异步任务需要）
- Pero 用户中心（OIDC UserInfo）

## 配置

复制 `etc/template.yaml` 为当前环境配置（例如 `etc/local.yaml`），配置知识库数据库、Redis、Celery、HTTPX 和 Pero：

```yaml
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
source tests/.env && uv run pytest
```

默认 ASGI 入口：`api.entrypoint.factory:service_api`。

- 服务文档：`/knowledge/docs/`
- 知识库 API：`/knowledge/v1/`
- 健康检查：`/knowledge/health`

## 数据归属约定

- `Collection.user_id` 与 `FileSource.user_id` 存储 Pero 返回的 UUID subject。
- 用户 ID 只能从已验证的 Token 获取，禁止从请求 Body、Query 或客户端可控字段覆盖。
- 未归属用户的文件源可作为公共文件源读取；公共集合可读取，但只有集合所有者可以修改或删除。
- 不在 Knowledge Base 中建立指向 Pero 数据库的外键，两个服务保持独立部署和独立数据边界。
