# LightRAG 多文档方案恢复：Neo4j 图存储

本入口恢复之前的真实财报多文档实验：BGE-M3 + BM25/RRF 基础召回，
LightRAG mix 提供图辅助候选；逐子问题保护前两条基础候选，映射回原始证据，
最终按子问题均衡分配 8 条 / 3000 token 上下文。图谱摘要不直接充当财报原文。

本机部署与实测结果见 [NEO4J_RESULTS.md](NEO4J_RESULTS.md)。
恢复后的 Precision/Recall 配对评测见 [PRECISION_RECALL_RESULTS.md](PRECISION_RECALL_RESULTS.md)。

只把实验图存储从 NetworkXStorage 换为 Neo4JStorage。主应用的 RAG、
PostgreSQL/pgvector 和现有重排开关保持原状；聊天页面目前不会自动调用此实验。
LightRAG 的向量仍在 NanoVectorDB，文本/KV/状态仍在对应 JSON 文件。
因此不能只备份 Neo4j 数据卷：还要一并保留该实验工作目录。

## 启动及入口（项目根目录，PowerShell）

已有 experiments/lightrag/.venv，补装固定版本驱动：

```powershell
experiments/lightrag/.venv/Scripts/python.exe -m pip install neo4j==5.28.2
./scripts/start-lightrag-neo4j.ps1
```

脚本依次生成本地凭据、启动 Neo4j、恢复旧图、做缓存查询对照；任一步失败立即停止。
首次运行需要 Docker Desktop 正常启动并能拉取镜像。它不会重新调用 LLM 建图。
Neo4j Browser：<http://127.0.0.1:7474>；Bolt：`bolt://127.0.0.1:7687`。
用户名 `neo4j`，密码查看忽略提交的 `.env.neo4j.local`；不在日志中输出密码。

单独运行：

```powershell
experiments/lightrag/.venv/Scripts/python.exe experiments/lightrag/neo4j_experiment.py init
docker compose --env-file experiments/lightrag/.env.neo4j.local -f docker-compose.lightrag.yml up -d --wait
experiments/lightrag/.venv/Scripts/python.exe experiments/lightrag/neo4j_experiment.py migrate
experiments/lightrag/.venv/Scripts/python.exe experiments/lightrag/neo4j_experiment.py verify
```

新查询需运行项目 BGE-M3 服务（8010），LLM 沿用 `.env.local` 中 DeepSeek 配置。
`query` 只返回原文证据和图检索诊断，最多允许 12 次关键词提取调用，不生成答案：

```powershell
experiments/lightrag/.venv/Scripts/python.exe experiments/lightrag/neo4j_experiment.py query --collection "finqa:HIG/2013" --query "What collateral could be required after a ratings downgrade?"
```

可重复传 `--collection`。多子问题可在 `--query` 中使用每行
`- [finqa:HIG/2013] 子问题` 的格式；只能引用索引内报告，未知报告会报错。
未指定子问题时，对每份选中报告分别检索同一个问题。可选报告列表见迁移目录的 `plan.json`。
同一工作目录请串行运行命令，勿与建图/迁移同时查询。

可以先完全离线验证已有问题，无须新的模型请求；缓存缺失直接报错：

```powershell
experiments/lightrag/.venv/Scripts/python.exe experiments/lightrag/neo4j_experiment.py query --offline --collection "finqa:HIG/2011" --query "what is the change in value of fixed maturities afs from 2010 to 2011 , ( in billions ) ?"
```

## 迁移与验证边界

- 默认来源由 `multidoc_latest.json` 指向完整旧实验；支持 `--source <旧实验目录>`。
- 默认目标 `runs/neo4j-multidoc`，支持 `--target <新目录>`。
- 原目录只读；同时复制 GraphML、对应 KV/向量 JSON、BGE 查询缓存及文档映射。
- 每个目标生成独立 workspace 标签；拒绝 `NEO4J_WORKSPACE` 跨索引覆盖。
- Community 数据库明确指定 `neo4j`；workspace 标签仅用于逻辑隔离，不等于租户权限隔离。
- 节点先于边写入；对照全部节点/边属性并核对每个 source_id 能定位原文 chunk。
- 缺少 GraphML 或任一必需 JSON 索引文件即拒绝迁移，防止空向量库被当成成功。
- 源文件 SHA256 固定在 manifest；中断状态保持 preparing，完成校验才标记 ready。
- `verify` 覆盖跨公司、跨年度、三报告案例，比较 NetworkX 与 Neo4j 的原文来源集合及排序；
  禁止新 LLM/嵌入调用，缓存未命中就失败，不悄悄联网补算。
- 两个对照后端都按实体名称规范邻边顺序，消除 LightRAG 1.5.7 对同分关系的存储遍历次序依赖；
  不改原有 degree/weight 分数。验证还对比图上下文和三个案例的最终原文证据。
- 本次是存储迁移与检索一致性测试，不是新的回答质量提升实验；历史开发集结果仍有原有局限。

Docker 使用 Neo4j Community 5.26.12，2 CPU / 2GB 上限，heap 最大 768MB，
page cache 256MB，仅绑定本机端口，独立命名数据卷保存数据库。未启用 APOC，核心图检索不需要它。

原方案回退：继续运行原实验入口即可，它仍默认 NetworkXStorage；原索引未覆盖。
暂停图数据库用 `docker compose --env-file experiments/lightrag/.env.neo4j.local -f docker-compose.lightrag.yml stop`。
不要执行 `down -v`，它会删除 Neo4j 数据卷。若数据卷已初始化，修改 env 密码不会改变数据库现有密码。

实现依据：[LightRAG](https://github.com/HKUDS/LightRAG)、
[Neo4j Docker 文档](https://neo4j.com/docs/operations-manual/current/docker/introduction/)。
