# Knowledge Base Service

主要APP：

- ` /user `: 用户中心与认证
- ` /knowledge `: 知识库、文档、RAG 相关能力

项目基于 Python `3.13`、FastAPI、Tortoise ORM、Redis，以及一组本地 workspace 扩展模块（文档解析、文件源、向量索引、LLM、工作流等）。

## 目录结构

```text
api/        HTTP API 入口与路由
service/    业务编排与领域服务
ext/        可复用扩展模块（embedding / llm / workflow / indexing / file_source ...）
core/       应用基建、响应模型、中间件、上下文
config/     配置定义
deploy/     初始化脚本、权限脚本、迁移辅助
tests/      测试
```

## 开发环境

要求：

- Python `3.13`
- `uv`
- 相关组件

初始化：

```bash
make setup
```

如果需要跑测试，使用测试环境变量：

```bash
source tests/.env && uv run pytest 
```

## 启动服务

开发模式：

```bash
make server
```

默认入口：

```text
api.entrypoint.factory:service_api
```

启动后可访问：

- `/docs`
- `/user/docs/`
- `/knowledge/docs/`

## 常用命令

```bash
make format                     # isort + black
make check                      # ruff + mypy
make shell                      # 进入项目交互 shell
```


## 数据库迁移

项目使用 `aerich` 管理 Tortoise ORM 迁移：

```bash
make db-migrate
make db-upgrade
make db-downgrade
```

迁移目录：

```text
ext/ext_tortoise/migrate/
```

## 开发约定

- API 层放在 `api/`，尽量保持薄路由
- 业务逻辑优先落在 `service/`
- 通用能力或可插拔组件沉淀到 `ext/`
- 提交前至少执行一次 `make check`
- 涉及集成能力的测试通常需要 `source tests/.env`
- 提交之前至少需要执行 `make format` `make check` 并修复
