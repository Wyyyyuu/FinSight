# Docker BGE-M3 experiment

Final status (2026-09-15): COMPLETE for the requested CPU deployment and evaluation.
FlagEmbedding/BGE-M3 is running in Docker on port8010; FinSight API8002 and
frontend5173 are healthy. Main RAG uses bge-m3-http, memory-store BM25, RRF60,
and the CPU ONNX BGE-reranker-base (24 -> 8). Existing annual-report pipeline is
separate. REPORT.md / assessment.json contain final results and limitations.
The optional CUDA experiment remains paused and unvalidated; it is not active.

User scope: deploy FlagEmbedding in Docker, test real embeddings and choose a
single/hybrid retrieval strategy based on evidence. Preserve prior experiments.

Implementation:
- Dedicated CPU FlagEmbedding sidecar, port 8010 bound to loopback, separate model volume.
- Fixed BGE-M3 revision and ML dependency versions; no LLM secrets in build context.
- Main embedder HTTP adapter validates vector identity/dimensions/weights and fails closed.
- Reuse FinQA/CN cases read-only; new vector/rerank caches and reports only.
- Development selection uses final top8 after real top24 cross-encoder reranking.
- Test/regr modes validate dataset, source code, embedding and reranker identities.
- Multi-report global24-to8 and balanced3000 without reranking are separate measurements.

Initial status (historical): dependency image built; downloading pinned weights; 28 targeted
local tests passed. Runtime has not yet been switched. No retrieval quality result
is claimed until the real-model evaluation completes.

Verification update: pinned model SHA256 verified; container healthy with real
dense+sparse startup inference. 54 core RAG/remote tests plus 212 annual-report
regression tests passed (266 total). Dev100 evaluation is running; runtime remains
on the previous configuration until selection/validation completes.

Performance finding: CPU ONNX reranking 24 candidates is about ten seconds
per query on this machine, materially slower than embedding. A separate CUDA
container for the identical FP32 ONNX model is being built; it must pass actual
CUDA kernel profiling and CPU/GPU score/top8 parity before use. The CPU dev run
remains preliminary. Final GPU selection will rerun dev with a distinct model/runtime
cache identity; no CPU scores will be silently reused as GPU scores.

Reboot recovery (2026-09-15): Windows reported bugcheck 0x10E (video memory
management); cause is not established. GPU experiment is paused and unvalidated.
The machine has 16 GB RAM. CPU container is now bounded to 2 CPUs / 4 GB, batch4.
Both host SQLite caches pass PRAGMA quick_check (8,047 vectors, 4,217 old scores).
Docker initially failed on stale AF_UNIX socket reparse points; the runtime-only
directories were renamed to backups. Desktop subsequently recorded a factory-reset
UI action and came up with no images or volumes. Rebuild CPU image and model cache.
No factory reset was issued by agent tooling. Project/host caches remain intact.
Cached offline evaluation uses the exact health identity observed before reboot;
the restored service must match it before activation. Added encoding-only and
strict cached-replay phases to avoid overlapping model memory footprints.
CPU reranker now length-groups batch4 and restores input score order. Two real
EN/ZH checks produced identical scores/top8; measured 8.20s -> 5.28s and
2.48s -> 1.87s (small samples, not a full latency benchmark).

Recovery validation: the original CPU score profile was reconstructed and all
4,217 old pair keys matched. 3,860 overlapping recomputed pairs had exactly zero
score difference; 373 remaining pairs were copied with provenance recorded in
cache_recovery.json. This reuses CPU pair scores only, never benchmark timings.
CPU image and all model files were rebuilt/restored, with the same verified weight
SHA256. Dev100 completed: selected m3_bm25_rrf; all three semantic arms reached
93.6667% final Recall@8, with the BM25 arm slightly ahead in nDCG@8 (0.80586).
Test200 replay is running with that selection frozen. New checks passed: 60 RAG
unit/regression cases, plus a PowerShell parser check and Python compilation.
The report aggregator now verifies generation and latency artifact provenance
against the exact current retrieval selection/report before combining results.

Restored CPU service startup passed: /health returned ready=true with the exact
pre-reboot bge-m3:1024 revision and maxlen512 CPU profile. With Windows reranking
also resident, host free RAM dropped to about 1 GB; stop the embedding container
and gracefully stop Docker Desktop while replaying cached vectors. No containers
were running when the engine was stopped. Resume Desktop/embedding before the
remaining 1,950 regression input encodings and final runtime verification.

Test200 + multi40 completed (runs/20260915T045612Z-test): English Recall@24
77.25% -> 96.6667%; after actual rerank8 Recall 72.3155% -> 90.0238%, complete
evidence 64% -> 83.5%. Global multi rerank8 recall 22.2946% -> 55.9405%, complete
10% -> 22.5%. Global reranking reduces the new arm's multi recall versus its own
unreranked top8 (61.1905%); complex queries remain a weakness. Separate balanced
3000-token experiment reaches 97.5357% recall and is NOT the deployed global path.
Paired DeepSeek audit: 48 successful calls / 53,374 tokens / no request errors.
Single numeric9/16 ->9/16, numeric+evidence6/16 ->6/16; multi numeric2/14 ->6/14,
numeric+evidence2/14 ->5/14; both arms correctly abstained on 2/2 missing reports.
Do not infer general answer accuracy from retrieval improvements. Six of seven
failed selected single answers had all annotated gold snippets supplied, yet
abstained; minimal gold labels do not prove complete calculation context.
Restored vectors passed 16-input parity (max dense delta1.70e-7, sparse7.75e-7).
Regression embedding preparation resumed with 1,950 missing inputs, CPU only.

Final acceptance:
- All 155 regression questions completed: English130 final Recall@8 67.0513% ->
  86.7308%; Chinese25 97.6667% ->97.6667%. No test-driven parameter retuning.
- Live uncached service check completed: 12 warm samples per arm (10 EN, 2 ZH).
  Combined retrieval+rerank p50 3.045s ->3.929s, p95 4.220s ->5.725s.
  These exclude ingestion/cold loading/LLM generation and are not a load test.
- Docker's stale-socket issue recurred after a clean engine stop. Backing up BOTH
  Docker/run and docker-secrets-engine runtime directories together restored it;
  image/model volume remained intact. No factory reset was used for this recovery.
- start-local.ps1 -SkipBuild succeeded, and saved the active non-secret RAG config
  to logs/local-deployment/rag-runtime.json.
- /health is healthy and reports RAG memory / bge-m3 /1024d; chat, annual-reports,
  API docs return HTTP200. Docker embedding container is healthy (about837 MiB
  Docker-reported usage while idle, limit4GiB; this excludes WSL overhead/cache).
- 60 focused RAG checks passed after CPU batching changes. Report provenance
  assertions, real-model parity, PowerShell parsing and Python compilation passed.
- REPORT.md distinguishes retrieval gains, unchanged single-answer correctness,
  multi-answer gains, CPU latency cost, and the separate balanced-context experiment.
