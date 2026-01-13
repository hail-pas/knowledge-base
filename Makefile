# 定义目录和变量
DIRS = ./config ./api ./constant ./core ./deploy ./enhance ./ext ./util
export MYPYPATH=./
PYTHON_VERSION = 3.13
VENV_PATH = .venv
APP = api.entrypoint.factory:service_api
HOST = 0.0.0.0
PORT = 8000
RELOAD = true

# 默认目标
.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "知识库项目 - Makefile 帮助"
	@echo ""
	@echo "用法: make <target>"
	@echo ""
	@echo "可用目标:"
	@awk 'BEGIN {FS = ":.*##"; printf "%-20s %s\n", "目标", "描述"} /^[a-zA-Z_-]+:.*?##/ { printf "  %-18s %s\n", $$1, $$2 } /^##@/ { printf "\n%s\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ 开发环境设置
.PHONY: dev setup
dev: setup ## 设置开发环境（别名）
setup: ## 设置开发环境
	uv venv --python $(PYTHON_VERSION) || true
	uv sync --group dev all-extras
	@echo "开发环境已设置完成！"

.PHONY: deps
deps: ## 安装项目依赖
	uv sync --all-extras

.PHONY: up
up: ## 更新依赖到最新版本
	uv lock --upgrade
	uv sync --all-extras

##@ 代码质量
.PHONY: style format
style: format ## 格式化代码（别名）
format: ## 自动格式化代码（black + isort）
	uv run isort --length-sort $(DIRS)
	uv run black $(DIRS)
	@echo "代码格式化完成！"

.PHONY: lint check
lint: check ## 检查代码质量（别名）
check: ## 运行代码检查（ruff + mypy）
	uv run ruff check --fix .
	uv run mypy --explicit-package-bases --implicit-reexport .
	@echo "代码检查完成！"

.PHONY: type-check
type-check: ## 仅运行类型检查
	uv run mypy --explicit-package-bases --implicit-reexport .

.PHONY: lint-check
lint-check: ## 仅运行 lint 检查（不自动修复）
	uv run ruff check .

##@ 测试
.PHONY: test
test: ## 运行测试（使用 ARGS 传递参数，如：make test ARGS=tests/test_api.py）
	source tests/.env && uv run pytest -s -p no:warnings  $(ARGS)

.PHONY: test-cov
test-cov: ## 运行测试并生成覆盖率报告
	uv run pytest --cov=./ --cov-report=html --cov-report=term

.PHONY: test-watch
test-watch: ## 监听文件变化并运行测试
	uv run pytest-watch

##@ 数据库
.PHONY: db-init
db-init: ## 初始化数据库迁移
	uv run aerich init -t ext.ext_tortoise.migrate.env.TORTOISE_ORM_CONFIG

.PHONY: db-migrate
db-migrate: ## 创建新的数据库迁移
	uv run aerich --app $(APP) migrate

.PHONY: db-upgrade
db-upgrade: ## 应用数据库迁移
	uv run aerich --app $(APP) upgrade

.PHONY: db-downgrade
db-downgrade: ## 回滚数据库迁移
	uv run aerich --app $(APP) downgrade

##@ 运行
.PHONY: run server
run: server ## 运行服务器（别名）
server: ## 启动开发服务器
	uv run uvicorn $(APP) --reload --host $(HOST) --port $(PORT)

.PHONY: shell
shell: ## 进入 Python shell
	uv run shell.py

##@ 工具
.PHONY: clean
clean: ## 清理缓存和临时文件
	@echo "清理缓存文件..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name "*.pyd" -delete 2>/dev/null || true
	@find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	@rm -rf .coverage htmlcov 2>/dev/null || true
	@echo "清理完成！"

.PHONY: pre-commit
pre-commit: ## 手动执行 pre-commit 检查
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) test

.PHONY: add-dep
add-dep: ## 添加新的生产依赖（使用：make add-dep PACKAGE=package-name）
	uv add $(PACKAGE)

.PHONY: add-dep-dev
add-dep-dev: ## 添加新的开发依赖（使用：make add-dep-dev PACKAGE=package-name）
	uv add --group dev $(PACKAGE)

.PHONY: freeze
freeze: ## 导出依赖列表
	uv pip freeze > requirements.txt
	@echo "依赖已导出到 requirements.txt"

.PHONY: info
info: ## 显示项目信息
	@echo "项目信息:"
	@echo "  项目名称: fastapi-starter"
	@echo "  Python 版本: $(PYTHON_VERSION)"
	@echo "  虚拟环境: $(VENV_PATH)"
	@uv --version
	@uv python list

.PHONY: lock
lock: ## 更新 lock 文件
	uv lock
