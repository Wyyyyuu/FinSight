# Local CUDA reranker (paused, not validated)

This optional CUDA experiment was interrupted by a Windows reboot. The active
deployment uses CPU ONNX reranking. Do not assume this container has passed
runtime or parity checks; see the embedding experiment log.

Runs the **existing BGE-reranker-base FP32 ONNX weights** on an NVIDIA GPU.
The model is mounted read-only from the previously prepared local cache; no model
download or API key is required. GPU libraries are installed in this Docker image,
not in the Windows backend environment.

```powershell
docker compose -f docker-compose.reranker.yml up -d --build
Invoke-RestMethod http://127.0.0.1:8011/health
```

Requires Docker Desktop Linux containers with NVIDIA GPU passthrough. Startup
asserts the actual CUDA session provider, runs real inference, and verifies an
ORT trace contains CUDA matrix multiplication kernels. Failed CUDA loading fails
startup instead of reporting a healthy CPU fallback.

```dotenv
RAG_ENABLE_RERANKER=true
RAG_RERANKER=bge-reranker-http
RAG_RERANKER_URL=http://127.0.0.1:8011
```

FP32 precision is retained, TF32 is disabled, and batches contain at most 8 pairs.
Retrieving 24 candidates therefore uses 3 inference batches and still returns the
best 8 across all 24. Model-input batch size is not the retrieval Top K.

The API binds to loopback only. `/health` records the model revision, weight hash,
runtime/provider identity and actual startup CUDA kernel count. CPU and GPU score
caches are kept separate in the experiment.

CPU alternative: `RAG_RERANKER=bge-reranker-onnx` with
`RAG_RERANKER_CACHE_DIR=experiments/lightrag/models` in the Windows backend.
This uses the same weights but does not require this GPU container.

References: [ORT CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html),
[FastEmbed GPU support](https://qdrant.github.io/fastembed/examples/FastEmbed_GPU/).
