# Indexing 模块：后端选择与生命周期

## 设计原则

- 保持 `BaseIndexModel` 的 CRUD、Dense/Sparse/Hybrid Search、过滤、分页、批量写入接口不变。
- ES 与 Milvus 的实现仍在 `providers/`；工厂负责连接和动态模型，业务端使用 `CollectionIndexModelHelper`。
- 默认调用仍由启动注册的默认后端处理；特殊知识库可按请求显式指定后端，不再修改全局 `Meta.provider`。
- 稠密与稀疏索引是**两个独立角色**。当前默认实现是 Milvus（Dense/FAQ）与 Elasticsearch（Sparse）；切换连接配置不代表把两种索引的数据自动搬迁到另一服务。

## 使用示例

```python
from ext.indexing.models import CollectionIndexModelHelper
from ext.ext_tortoise.models.knowledge_base import IndexingBackendConfig

# 默认后端（保持向后兼容）
helper = CollectionIndexModelHelper(collection)

# 针对一个知识库选择特定后端配置，不影响其他请求
dense_config = await IndexingBackendConfig.get(id=milvus_config_id, is_enabled=True)
sparse_config = await IndexingBackendConfig.get(id=es_config_id, is_enabled=True)
helper = await CollectionIndexModelHelper.for_backends(
    collection, dense_config=dense_config, sparse_config=sparse_config
)
dense = await helper.get_dense_model()
sparse = helper.sparse_model
faq = await helper.get_faq_model()
```

`IndexingProviderFactory.create(config)` 按配置 ID 复用连接，参数改变时刷新连接。配置禁用或删除后可调用 `await IndexingProviderFactory.invalidate(config_id)`。应用退出统一调用 `clear_cache()` 断开连接。

`IndexModelFactory.create_for_embedding(base_model, embedding_config, provider=...)` 按基础模型类、向量维度、Provider 实例进行隔离；`bind_provider(base_model, provider)` 用于无需向量维度的模型。动态 Schema 在进程中同一个模型实例仅初始化一次。调用 `IndexModelFactory.clear_cache()` 可以清除这些记录。

## 切换边界

- **不是热数据迁移**：换到空的后端后，需安排重建/回填索引并核验数量及检索质量，再把生产流量切到新的配置。
- **并发注意**：同一配置 ID 的连接参数在线更新会重建连接；应在维护窗口修改连接地址/凭据，以免旧请求持有失效连接。若需零中断切换，应创建新配置 ID 并在知识库层灰度切流。
- **能力差异**：调用方依然需要确认目标 Provider 支持所用查询类型（如原生 Hybrid、分区和全文分析器），不保证任意后端可互换所有索引角色。
- **已有问题不在本次改动范围**：索引命名、索引数据迁移和 ES/Milvus 底层查询行为保留原实现，避免引入破坏性迁移。

## 验证

```bash
uv run pytest tests/ext/indexing/test_provider_routing.py -q
uv run pytest tests/ext/indexing/test_indexing_factory.py -q
```

连接真实 Elasticsearch、Milvus 的集成测试仍需在具备服务的环境执行。
