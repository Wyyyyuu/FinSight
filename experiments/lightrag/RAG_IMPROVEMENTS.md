# RAG 第一轮改进与验证（2026-09-14）

已实现排序修复、实际向量输出校验、模型空间隔离、可选中文词法召回，以及可选本地 ONNX 重排。没有修改现有模型配置、主环境依赖或生产数据库，没有启动索引重建，也没有把 LightRAG 接入正式请求链。

## 实现

- `backend/rag/hybrid_service.py`：内存和 PostgreSQL 都移除 scope 固定加分。原 persistent 加分 0.15，而默认 RRF 两路最高总分仅约 0.0328；保留 scope/TTL 的生命周期语义。
- `backend/rag/embedder.py`：校验实际编码数量、维度、有限值；修复 BGE 缺失时报告 96 维、实际输出 1024 维的不一致。BGE 未返回 dense 时不得把 hash 冒充 BGE。模型运行失败的原因随实际状态更新。
- 每个新写入文档在保留 metadata 中记录 `_embedding_profile`（模型、维度、编码版本）。查询依据实际编码结果匹配；不兼容或无标记文档不参与 dense，文本召回仍可使用。学习式 sparse 的 token ID 也不跨模型匹配。只有 profile 相同才允许覆盖已有向量；不同或未知 profile 均拒绝覆盖，须使用独立 collection 重建。
- 不再仅凭 PostgreSQL 列维度自动改选另一个 embedding 模型。部署时若旧库维度不一致，会要求显式对齐配置；本次没有修改现有库或配置。
- `backend/rag/lexical.py`：NFKC 归一化、英文/数字 token、连续汉字双字组合，独立于 dense 的原有分词过程。默认仍为 legacy，通过 `RAG_LEXICAL_MODE=cjk` 在服务启动前选择。内存使用稀疏权重匹配；PostgreSQL 新写入内容保留原文词项并附加双字词项，标记 `_lexical_version`，开启时使用 OR 查询。无标记旧行继续原 `plainto_tsquery`，不会自动重写旧索引。
- `backend/rag/reranker.py`：可选 `RAG_RERANKER=bge-reranker-onnx`，使用 fastembed 的 `BAAI/bge-reranker-base`。只加载本地缓存，不自动下载。设置 `RAG_RERANKER_CACHE_DIR` 指向缓存；现有执行链仍需 `RAG_ENABLE_RERANKER=1`。默认未启用。缺少模型/依赖、输出缺项或非有限分数时保留原候选。
- 可选 `RAG_REQUIRE_SEMANTIC_EMBEDDING=1` 在编码时拒绝 hash 降级；默认未开启。当前主 `.venv` 的实测仍是 `hash:1024:v1`，原因是缺少 FlagEmbedding。本实验使用独立环境中的真实中文向量，不能把实验收益视为生产已获得的收益。

编码版本需要在模型权重版本、分词、池化或截断配置改变时更新。内置 BGE 标记包含 max_length；自定义 embedding adapter 应提供对应 revision。metadata 标记用于内部索引身份，不是内容签名。

## 实测结果

统一 16 篇虚构金融文档、12 问；10 道可回答题计算召回与排序，另外 2 道缺失信息题不计算这些指标。每个方案每问 3 次，共 144 次检索。所有方案使用同一真实 `BAAI/bge-small-zh-v1.5`，最终 top-4；重排先取 12 个候选。单进程 CPU，模型预热后轮换方案顺序。计时包含 embedding、召回和可选重排。

| 方案 | Recall@4 | nDCG@4 | 集齐全部证据的题目 | 热查询 P50 | 热查询 P95 |
|---|---:|---:|---:|---:|---:|
| 修改前基线 | 82.50% | 0.8680 | 6/10 | 14.03 ms | 18.78 ms |
| 修正生命周期加分 | 82.50% | 0.8680 | 6/10 | 13.51 ms | 25.17 ms |
| 再加中文词法召回 | 90.83% | 0.8324 | 8/10 | 14.06 ms | 22.79 ms |
| 中文召回 + BGE 重排 | 90.83% | 0.9062 | 7/10 | 2148.12 ms | 2370.88 ms |

同生命周期语料中，移除固定加分不应改变排序，实测符合预期。另列的对抗探针验证：旧版把无关 persistent 文档排到相关 ephemeral 文档前面，修复后相关文档第一；中文短语探针需要中文词法模式才能命中正确首条。这两项探针未混入主评测指标。

中文召回增加覆盖，但排序分数下降。重排改善排序，却增加约 2.13 秒 CPU 延迟；其集齐全部证据的题数还少于单独中文召回，因此不设为默认。下一步适合用更多真实业务问题验证中文模式，以及在报告链路按预算试用重排。本次没有进行回答生成、人工答案评审或 LLM judge，不能据此声称最终回答正确率提升。API 调用为 0。

具体退步发生在 `risk` 题：中文召回前四条包含 S01 的独家供应商证据；重排后 S01 被现金流相关资料挤出，供给与回款两方面的证据不再齐全。重排的总体排序分提高，不等于复合问题覆盖一定提高。

主结果文件：`runs/20260914T150708Z-rag-improvements/report.json`，含逐题检索、耗时、基线源码快照及 SHA256；便于保留的摘要在 `rag_improvements_assessment.json`。基线源码快照只冻结 `hybrid_service.py`，共用编码器没有改变真实模型或 dense 分词。

## 验证与复现

后端相关回归 84 项通过，覆盖 RAG、embedding、reranker、chunker、router、observability；另有原 LightRAG 实验 10 项通过。最终复核补上了未知 profile 旧行的写入保护，避免普通 upsert 悄悄重嵌入旧行。PostgreSQL 校验包含 SQL 捕获、参数绑定、upsert 冲突和旧索引查询分支，尚未对真实 PostgreSQL 执行计划或规模化延迟进行测试。内存 benchmark 的耗时不能外推到数据库。

```powershell
.venv/Scripts/python.exe -m pytest backend/tests/test_rag_retrieval_safety.py backend/tests/test_rag_v2_service.py backend/tests/test_embedder.py backend/tests/test_rag_router.py backend/tests/test_reranker.py backend/tests/test_chunker.py backend/tests/test_rag_observability_store.py backend/tests/test_rag_observability_execute_plan.py backend/tests/test_rag_observability_system_router.py backend/tests/test_rag_observability_auth.py -q -p no:cacheprovider
experiments/lightrag/.venv/Scripts/python.exe -m pytest experiments/lightrag/test_experiment.py -q -p no:cacheprovider
experiments/lightrag/.venv/Scripts/python.exe experiments/lightrag/rag_improvements.py
```

实验依赖独立 `.venv` 的 fastembed 0.8.0、现有 `.annual-models` 中文 embedding 缓存、`experiments/lightrag/models` 重排缓存，以及 `runs/rag-improvements-baseline/hybrid_service.py` 的修改前快照。`runs` 和模型不提交；若迁移机器，可从完整报告的 `baseline_source` 恢复快照。当前脚本不联网下载，也不读取模型 API 密钥。
