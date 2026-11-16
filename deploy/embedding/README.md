# Local Docker embedding service

This deploys **FlagEmbedding and BGE-M3**, while the existing Windows backend and
frontend continue to run locally. It does not deploy or modify the production
Compose stack, PostgreSQL instances, annual-report indexes, or prior experiments.

```powershell
docker compose -f docker-compose.embedding.yml up -d --build
docker compose -f docker-compose.embedding.yml logs -f embedding
Invoke-RestMethod http://127.0.0.1:8010/health
```

Wait for `ready: true`. The health endpoint is served only after a real dense and
sparse encoding succeeds. A container being created is not proof the model loaded.

Backend configuration (restart the backend after setting):

```dotenv
RAG_EMBEDDING=bge-m3-http
RAG_EMBEDDING_URL=http://127.0.0.1:8010
RAG_REQUIRE_SEMANTIC_EMBEDDING=true
RAG_ENABLE_RERANKER=true
RAG_V2_TOP_K=24
RAG_V2_RERANK_TOP_N=8
```

The current local reranker uses the separately cached BGE-reranker-base ONNX model.
The Docker embedding service itself has no LLM credentials and requires none.

Model: `BAAI/bge-m3`, pinned revision
`5617a9f61b028005a4858fdac845db406aefb181`, 1024 dimensions, maximum 512 model tokens
per input, CPU inference. Model files persist in
`finsight-embedding_embedding_models`; the port binds only to `127.0.0.1`.
The input limit is deliberately recorded in vector profiles to prevent mixing
vectors from different preprocessing settings.

`RAG_EMBEDDING=bge-m3` means loading FlagEmbedding inside the backend process;
`bge-m3-http` means calling this container. Installing a library in a container
does not make it importable by Windows Python.

The HTTP adapter rejects wrong model identity, dimensions, invalid vectors and
mid-batch revision changes. It never silently returns hash vectors on failure.
Old hash vectors must not be reinterpreted as BGE-M3 just because both have 1024
dimensions. Existing collection profile checks reject in-place model replacement;
reindex persistent corpora into separate collections before switching references.

```powershell
# Stop only the embedding service, retaining model files.
docker compose -f docker-compose.embedding.yml stop
# Restore it using the same cache.
docker compose -f docker-compose.embedding.yml start
```

Model capabilities and dense/sparse/ColBERT examples:
[official BGE-M3 model card](https://huggingface.co/BAAI/bge-m3).
This deployment produces dense and learned sparse vectors; it does not store
ColBERT token vectors. Retrieval selection and measured results are recorded in
`experiments/docker_embedding/`.

The current 16 GB Windows machine uses CPU only, 2 container CPUs, a 4 GB memory
limit and embedding batch4. CUDA reranking is an unvalidated optional experiment;
it is paused following a Windows video-memory bugcheck. Do not use its performance
as evidence for this deployment.

After the experiment writes `selection.json`, `scripts/start-local.ps1 -SkipBuild`
checks that the live embedding revision matches the evaluated model and applies
the chosen memory-store retrieval mode. It retains CPU ONNX reranking (batch4,
24 candidates -> 8 passages). Main memory collections are rebuilt from new
ingestion; existing annual-report indexes keep their separate FastEmbed pipeline.

To run experiments with bounded memory, prepare embeddings first, stop only the
embedding service, and replay cached vectors with the CPU reranker:

```powershell
.venv\Scripts\python.exe experiments/docker_embedding/evaluate.py dev --prepare-only
.venv\Scripts\python.exe experiments/docker_embedding/evaluate.py test --prepare-only
.venv\Scripts\python.exe experiments/docker_embedding/evaluate.py regression --prepare-only
docker compose -f docker-compose.embedding.yml stop
.venv\Scripts\python.exe experiments/docker_embedding/evaluate.py dev --rerank --offline
.venv\Scripts\python.exe experiments/docker_embedding/evaluate.py test --rerank --offline
.venv\Scripts\python.exe experiments/docker_embedding/evaluate.py regression --rerank --offline
docker compose -f docker-compose.embedding.yml start
```

Offline replay fails if any vector is missing. Cached score times are excluded
from online latency; `runtime_check.py` measures the live service separately.
