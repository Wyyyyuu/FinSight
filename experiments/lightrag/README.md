# FinSight LightRAG 独立实验

2026-09-16：已增加独立 Neo4j 图存储恢复、查询和一致性验证入口，见
[NEO4J.md](NEO4J.md)。原实验入口仍保留 NetworkX 默认值，不覆盖旧索引或主应用数据库。

2026-09-15 多文档追加实验已完成：Docker BGE-M3、真实财报片段、同页聚合建图及保留基础候选的融合修复。结论与评价口径见 [多文档实验结论](MULTIDOC_FINDINGS.md)；局部开发集收益不作为替换主流程的依据。

状态（2026-09-14）：用户已授权并完成首次真实实验：10项实验单元检查、35项原有RAG回归、16篇合成资料建图、12题×4组配对评测，以及来源约束修复后的12份回答回放。**本轮没有足够证据支持替换原RAG**；完整结果与指标限制见 [RESULTS.md](RESULTS.md)。实验目录之外的业务代码、应用依赖、配置与生产数据均未修改。

## 隔离方式

- 唯一入口为本目录 `run.py`，默认 `--mode plan` 只显示计划。
- 未在 FastAPI、Agent、Planner、`get_rag_service()` 或前端注册任何新后端。
- 原有检索器直接构造独立 `HybridRAGService(backend="memory")` 实例，不读取生产数据库，不调用应用单例，不安装 observability hook。
- LightRAG 使用文件存储：JSON KV、NanoVectorDB、NetworkX、JSON DocStatus。每次运行创建全新 `runs/<时间>-<随机ID>/index`，不复用生产 collection，也不覆盖上次实验。
- 依赖安装在 `experiments/lightrag/.venv`；原项目 `.venv` 与根目录 requirements 不变。模型默认只读取已有 `.annual-models` 缓存，缺失时报错，不能悄悄退回 hash。
- `runs/`、虚拟环境和模型下载目录已在本目录 `.gitignore` 排除。取消实验只需停止独立进程；没有需要切回的生产开关。

## 对照设计

| 组 | 检索器 | 向量 | 用途 |
|---|---|---|---|
| `legacy_hash` | 现有 HybridRAG 内存实现 | 显式 1024 维 hash | 重现缺失 bge-m3 时的降级诊断 |
| `legacy_semantic` | 同一 HybridRAG 实现 | bge-small-zh-v1.5 | 主基线 |
| `lightrag_naive` | LightRAG 文本向量检索 | 同一 bge-small-zh-v1.5 | 区分引擎变化与图谱变化 |
| `lightrag_mix` | LightRAG 图谱与文本混合检索 | 同一 bge-small-zh-v1.5 | 实验组 |

不以 `mix - hash` 作为图谱的收益结论。主比较是 `mix - legacy_semantic`，辅助比较 `mix - naive`。

双方使用同一语料、原文单元、Top-K、上下文 token 上限、回答 prompt 和生成模型；重排均关闭。legacy 保留自身词法匹配和 RRF，LightRAG 使用自身召回规则。两个 LightRAG 模式共享同一次已构建的图索引，因此索引成本单列为 `lightrag_shared`，不能当作仅构建纯向量索引的成本。

题集包含16篇虚构证据、12个问题，覆盖数值事实、多文档、供应链多跳、否定关系、时间冲突、比较与正确拒答。所有组检索整个语料，不给每个问题预先筛选正确资料。标准答案只供评分，禁止送入检索或建图。原文 `S01` 等 ID 通过 LightRAG `file_paths` 回传并保留到答案引用。

该题集用于验证受控场景，不能代表真实财报准确性。后续应增加有原始出处的真实金融资料，并独立标注；原项目旧题集存在标准答案与上下文不完全一致的案例，不能直接把旧报告当基线。

## 运行方式

在项目根目录使用 PowerShell：

```powershell
python -m venv experiments/lightrag/.venv
experiments/lightrag/.venv/Scripts/python.exe -m pip install -r experiments/lightrag/requirements.txt

# 只查看设计，不加载模型，不调用 API，不生成评测报告
experiments/lightrag/.venv/Scripts/python.exe experiments/lightrag/run.py

# 只跑两组本地检索基线，无 LLM 调用
experiments/lightrag/.venv/Scripts/python.exe experiments/lightrag/run.py --mode baseline

# 真实建图、四组检索、生成和评估，需要先配置模型凭据
experiments/lightrag/.venv/Scripts/python.exe experiments/lightrag/run.py --mode live
```

当前独立虚拟环境已安装依赖，无需重复创建。若换机器，可按上述方式重建。仅在确实需要下载模型时显式增加：

```powershell
experiments/lightrag/.venv/Scripts/python.exe experiments/lightrag/run.py --mode baseline --model-cache experiments/lightrag/models --allow-model-download
```

**填写入口：本目录 `.env.local`**。地址和模型名已预填 DeepSeek，只需在 `LIGHTRAG_EVAL_API_KEY=""` 的双引号内填写密钥并保存。新环境可以复制 `.env.local.example` 为 `.env.local`。本地配置已被 Git 忽略，保存不会启动实验，也不会改动项目现有设置。

模型配置优先读取 `LIGHTRAG_EVAL_API_KEY`、`LIGHTRAG_EVAL_BASE_URL`、`LIGHTRAG_EVAL_MODEL` 环境变量，其次读取本目录 `.env.local`（仅解析，不修改进程环境）。三项必须完整；若已提供部分实验配置，不会偷偷借用原项目密钥。完全没有实验配置时才读取项目 `user_config.json` 已启用且有密钥的第一个 endpoint，最后兼容单 endpoint 与 `OPENAI_COMPATIBLE_*` 环境变量。仅支持 OpenAI 兼容 chat completions 接口。实验固定所选 endpoint，不轮换、不修改配置。不要把密钥写进命令历史、仓库或报告。

可在 `.env.local` 或环境变量中完整填写 `LIGHTRAG_EVAL_JUDGE_MODEL`、`LIGHTRAG_EVAL_JUDGE_BASE_URL`、`LIGHTRAG_EVAL_JUDGE_API_KEY` 指定评估模型；未配置则复用生成模型，并在报告标记 `same_model_judge=true`。独立评估模型三项必须一起指定。相同模型自评存在偏差，后续应人工复核差异案例。

默认每次请求60秒、建图1200秒、整个实验至多180次实际 LLM 调用，客户端不重试。LightRAG 的抽取与关键词调用也计入预算。DeepSeek官方端点统一显式关闭思考模式，避免默认思考消耗短输出预算；该设置对抽取、四组生成及Judge一致，并记入报告。需要 JSON 输出支持；模型截断、空响应、来源映射异常和评分解析失败都必须记录，不能当作质量分数。

图谱描述只有在其全部来源原文实际进入上下文后才保留；来源不能通过图谱绕过Top-K和原文引用边界。首次完整运行发现此问题后已修复，并用保存的召回结果单独回放混合组回答：`replay_grounding.py <report.json>`；若出现连接错误，可用 `--retry-errors` 仅重试失败样本，另存记录。这种回放不重新建图/检索、不重评Judge，不等同于第二轮完整四组性能实验。

## 报告内容与解释

每完成一组问题即原子保存 `runs/.../report.json`，并生成方便阅读的 `report.md` 对比表，同时保存题集快照、hash、依赖版本、原始召回、实际生成上下文、答案与逐次 LLM 调用统计。

- 检索：Recall@K、Precision@K（分母固定K）、nDCG@K、所有必要证据是否齐备。无法回答的问题不计算召回分数，避免虚高。
- 引用：编号有效性仅检查 ID 是否来自实际提供原文，不等同于事实支持。未知来源数单独记录。
- 生成：模型 Judge 检查原文忠实度、答案要点覆盖、引用支持、拒答是否正确。Judge 看不到检索器名称或图谱，只看问题、参考答案、原始证据与回答；数值、年份、单位和推测性表达包含在评判要求中。这些是本实验评分，不冒称 RAGAS 或现有 V2 指标。
- 性能：向量模型冷启动、各组索引时间、首次查询与后续热查询 p50/p95、生成耗时、索引字节数。默认每问3次检索，生成一次；首次查询发生在索引已经构建之后，不叫系统冷启动。
- 成本：逐阶段实际 LLM 调用次数、服务返回的输入/输出 tokens；未返回 usage 则保留 null。没有模型价格配置时货币成本为 null，不编造金额。
- 比较：同题 `mix - legacy_semantic` 差值、分类型汇总、每项已测样本数和错误数。失败或缺失值不补零、不计成通过；有失败时报告为 partial 且进程退出码为2。

判断是否值得接入时，优先看多跳证据齐备率和回答覆盖是否提高，同时确认数字/日期/引用错误与拒答没有恶化，再对照建图成本和查询延迟。12题的小样本 p95 与一次回答不能支持统计显著性或生产容量结论。`legacy_semantic` 的 memory 后端、词法实现和 bge-small 向量也不等同于部署中的 PostgreSQL+bge-m3。

## 验证记录

以下检查已经通过；后续修改涉及相应行为时可复跑：

```powershell
experiments/lightrag/.venv/Scripts/python.exe -m pytest experiments/lightrag/test_experiment.py -q
.venv/Scripts/python.exe -m pytest backend/tests/test_rag_v2_service.py backend/tests/test_rag_router.py backend/tests/test_reranker.py backend/tests/test_chunker.py -q
```

`lightrag-hku==1.5.7` 与配置的 DeepSeek API 已实际跑通。首次运行曾因默认思考模式下输出截断而中止，该失败记录保留，不纳入收益比较。完整实验虽无运行失败，但同模型Judge出现明确误判；原始自评分只作诊断，不能据此判断答案质量提高，详见结果报告。

未来若接入产品，另行实现 KB 异步晋升、collection/权限隔离、TTL、引用元数据和 Inspector 适配；本次没有启用这些生产能力。

参考：[LightRAG 官方仓库](https://github.com/HKUDS/LightRAG)。

## 后续：RAG 第一轮改进

用户随后授权实现排序修复、向量模型隔离、中文召回和重排对照。这一轮已修改后端代码，保留现有模型配置和数据库；其范围与此前仅限 LightRAG 实验不同。实现、兼容性变化、94 项回归结果及真实模型对照见 [RAG_IMPROVEMENTS.md](RAG_IMPROVEMENTS.md)。中文模式与 ONNX 重排默认未开启。

后续又完成了237道公开真实财报问题的无重排实验，包含开发／测试分离、BM25、表头和语言配置，以及真实模型数值诊断，详见 [真实金融检索评测](../financial_retrieval/README.md)。不同语料与评测范围的分数不能直接与这里的16篇虚构语料比较。
