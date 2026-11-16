# Docker BGE-M3 evaluation

Selected on dev100: `m3_bm25_rrf`.

Same candidate K=24, final K=8, RRF constant=60; all comparisons use the same corpus and source field.

English test eligible passages per question: median 86.5, range 14–217; 5/200 questions have at most 24 passages. Report scope is already known.

| Split | Method | Recall@24 | Recall@8 before rerank | Recall@8 after rerank | Complete evidence@8 | nDCG@8 |
|---|---|---:|---:|---:|---:|---:|
| English test (n=200) | hash_legacy | 77.25% | 58.73% | 72.32% | 64.00% | 65.25% |
| English test (n=200) | m3_bm25_rrf | 96.67% | 87.10% | 90.02% | 83.50% | 79.13% |
| English regression (n=130) | hash_legacy | 70.13% | 52.82% | 67.05% | 57.69% | 59.17% |
| English regression (n=130) | m3_bm25_rrf | 94.17% | 86.15% | 86.73% | 78.46% | 74.70% |
| Chinese regression (n=25) | hash_legacy | 100.00% | 51.67% | 97.67% | 92.00% | 94.44% |
| Chinese regression (n=25) | m3_bm25_rrf | 100.00% | 96.67% | 97.67% | 92.00% | 94.44% |

## English test ablation before reranking

| Method | Recall@8 | Recall@24 | nDCG@8 |
|---|---:|---:|---:|
| hash_legacy | 58.73% | 77.25% | 45.15% |
| hash_bm25 | 71.23% | 93.21% | 56.20% |
| m3_dense | 84.39% | 95.75% | 70.81% |
| m3_sparse_rrf | 86.65% | 96.75% | 75.97% |
| m3_bm25_rrf | 87.10% | 96.67% | 77.27% |

The upgrade combines a semantic encoder and BM25. The hash+BM25 arm isolates lexical improvements; its final rerank8 was not evaluated. These gains must not all be attributed to the encoder alone.

## Multi-report tasks

| Method | Tasks | Global recall@8 before rerank | Global rerank8 recall | Global rerank8 complete evidence | Balanced 3000-token recall (no rerank) | Balanced complete evidence |
|---|---:|---:|---:|---:|---:|---:|
| hash_legacy | 40 | 13.84% | 22.29% | 10.00% | 72.74% | 42.50% |
| m3_bm25_rrf | 40 | 61.19% | 55.94% | 22.50% | 97.54% | 90.00% |

## Generated answers (paired DeepSeek sample)

| Group | Method | Numeric correct | Numeric + required citation + quote checks | Complete cases |
|---|---|---:|---:|---:|
| single | previous | 9/16 | 6/16 | 6/16 |
| single | selected | 9/16 | 6/16 | 6/16 |
| multi | previous | 2/14 | 2/14 | 1/6 |
| multi | selected | 6/14 | 5/14 | 2/6 |

## Runtime (uncached real service, small sample)

Total means retrieval plus reranking only. Ingestion, cold model loading, and answer generation are excluded; ingestion and repeat0 measurements remain in the raw runtime artifact.

| Language | Method | Warm samples | Retrieval p50 / p95 | Rerank p50 / p95 | Total p50 / p95 |
|---|---|---:|---:|---:|---:|
| en | hash_legacy | 10 | 0.006s / 0.007s | 3.060s / 4.239s | 3.066s / 4.245s |
| en | m3_bm25_rrf | 10 | 0.284s / 0.330s | 4.038s / 5.416s | 4.312s / 5.735s |
| zh | hash_legacy | 2 | 0.002s / 0.003s | 1.189s / 1.216s | 1.192s / 1.217s |
| zh | m3_bm25_rrf | 2 | 0.252s / 0.265s | 1.198s / 1.245s | 1.450s / 1.510s |

## Limits

- Previously exposed FinQA data: this is a controlled replay, not a fresh unseen holdout.
- Report scope is supplied; corpus contains QA-bearing pages, not full-PDF search across the whole business KB.
- Multi-report tasks combine author questions; missing-report evidence is scored on answerable facets only.
- Balanced 3000-token contexts can contain more than 8 passages and are a separate retrieval-only experiment.
- Single/multi-evidence labels count gold snippets; they do not certify factual vs true multi-hop reasoning.
- Generation sample is small. Non-numeric/empty references remain unscored; quote matching does not establish full narrative factuality.
- Gold snippet recall does not guarantee all arithmetic operands or table headers are present. 6/7 failed selected single-question answers received all labeled gold snippets.
- Balanced context packing is a separate experiment; the deployed main path remains global retrieval24 -> rerank8.
- GPU path remains paused and unvalidated after the reboot. All final measurements use CPU.
