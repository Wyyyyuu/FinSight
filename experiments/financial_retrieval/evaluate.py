"""No-reranker ablations: tune on dev, freeze selection, evaluate held-out test.

Uses exact vectorized dot products and the existing lexical/RRF conventions.
Embedding caches accelerate development only; final latency encodes each query.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
from statistics import mean
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np
from backend.rag.bm25 import BM25Index, reciprocal_rank_fusion
from backend.rag.embedder import _tokenize, EmbeddingResult, SparseVector
from backend.rag.lexical import lexical_tokens
from experiments.lightrag.adapters import SharedEmbedding
from experiments.lightrag.metrics import retrieval_metrics, percentile


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def sparse_scores(query, texts, tokenizer):
    q = Counter(tokenizer(query)); scale = max(q.values(), default=1)
    weights = {t: v / scale for t, v in q.items()}
    norm = sum(v * v for v in weights.values()) ** .5 or 1
    result = []
    for counts in texts:
        maximum = max(counts.values(), default=1)
        # Match the service's accumulation order, including ties at float precision.
        score = 0.0
        for token, weight in weights.items():
            if token in counts:
                score += weight * (counts[token] / maximum)
        result.append(score / norm)
    return result


class Engine:
    def __init__(self, documents, vectors, config):
        self.config = config
        self.routes = ({lang: Engine(documents, vectors, cfg) for lang, cfg in config["by_language"].items()}
                       if "by_language" in config else None)
        if self.routes:
            return
        self.groups = defaultdict(list)
        for d in documents:
            self.groups[d["collection"]].append(d)
        self.index = {}
        field = config["field"]
        for collection, docs in self.groups.items():
            texts = [d[field] for d in docs]
            tokenizer = _tokenize if config["lexical"] == "legacy" else lexical_tokens
            lexical = (BM25Index(texts) if config["lexical"] == "bm25"
                       else [Counter(tokenizer(t)) for t in texts])
            matrix = np.asarray([vectors[sha(d[config.get("dense_field", field)])] for d in docs], dtype=np.float64)
            self.index[collection] = (matrix, lexical, tokenizer)

    def search(self, question, vector, k=4):
        if self.routes:
            return self.routes[self.query_language(question)].search(question, vector, k)
        docs = self.groups[question["collection"]]
        matrix, lexical, tokenizer = self.index[question["collection"]]
        dense = matrix @ np.asarray(vector, dtype=np.float64)
        sparse = (lexical.scores(question["question"]) if isinstance(lexical, BM25Index)
                  else sparse_scores(question["question"], lexical, tokenizer))
        ranks = reciprocal_rank_fusion(dense, sparse, rrf_k=self.config["rrf_k"],
                                       sparse_weight=self.config["sparse_weight"])
        return [docs[i] for i in ranks[:k]]

    @staticmethod
    def query_language(question):
        # Routing reads the query only, never its gold labels or dataset identity.
        return "zh" if re.search(r"[\u3400-\u9fff]", question["question"]) else "en"

    def context_field(self, question):
        return (self.routes[self.query_language(question)].context_field(question) if self.routes
                else self.config["field"])


BASELINE = dict(field="plain", lexical="legacy", rrf_k=60, sparse_weight=1.0)
CJK = dict(field="plain", lexical="cjk", rrf_k=60, sparse_weight=1.0)


def aggregate(rows):
    metrics = ["recall_at_k", "ndcg_at_k", "precision_at_k", "all_evidence", "mrr", "duplicate_rate"]
    return {**{k: mean(r[k] for r in rows) if rows else None for k in metrics},
            "questions": len(rows), "latency_p50_ms": percentile([v for r in rows for v in r.get("latency_ms", [])], .5),
            "latency_p95_ms": percentile([v for r in rows for v in r.get("latency_ms", [])], .95),
            "context_tokens_mean": mean(r["context_tokens"] for r in rows) if rows else None}


def row_metrics(q, hits, encoding, field):
    ids = [d["id"] for d in hits]; required = q["required"]
    reciprocal = next((1 / (i + 1) for i, v in enumerate(ids) if v in required), 0)
    fingerprints = [sha(d[field]) for d in hits]
    tokens = sum(len(encoding.encode(d[field])) for d in hits)
    assert tokens <= 1500, "Context budget exceeded; do not silently compare larger contexts"
    return dict(case_id=q["id"], language=q["language"], group=q["group"], dataset=q["dataset"],
                category=q["category"], retrieved=ids, required=required,
                mrr=reciprocal, duplicate_rate=1 - len(set(fingerprints)) / max(1, len(hits)),
                context_tokens=tokens,
                **retrieval_metrics(ids, required, 4))


def evaluate(engine, questions, vectors, encoding, encoder=None):
    rows = []
    for q in questions:
        latency = []
        for repeat in range(3 if encoder else 1):
            start = time.perf_counter()
            vector = encoder.encode([q["question"]]).dense[0] if encoder else vectors[sha(q["question"])]
            hits = engine.search(q, vector)
            latency.append((time.perf_counter() - start) * 1000)
        row = row_metrics(q, hits, encoding, engine.context_field(q))
        if encoder:
            row["latency_ms"] = latency
        rows.append(row)
    return rows


def prepare_vectors(encoder, documents, questions):
    cache = HERE / "cache"; cache.mkdir(exist_ok=True)
    path = cache / "bge-small-zh-46fbe35-vectors.npz"
    vectors = {}
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            vectors = dict(zip(stored["keys"].tolist(), stored["vectors"]))
    texts = list(dict.fromkeys([d[field] for d in documents for field in ("plain", "structured")]
                              + [q["question"] for q in questions]))
    pending = [t for t in texts if sha(t) not in vectors]
    for i in range(0, len(pending), 64):
        batch = pending[i:i + 64]
        result = encoder.encode(batch)
        for text, vector in zip(batch, result.dense):
            vectors[sha(text)] = np.asarray(vector, dtype=np.float32)
        if i % 512 == 0 or i + 64 >= len(pending):
            np.savez_compressed(path, keys=np.asarray(list(vectors)), vectors=np.asarray(list(vectors.values())))
            print("embedded", min(i + 64, len(pending)), "/", len(pending), flush=True)
    return vectors


def parity_check(data, vectors):
    """Check optimization preserves actual current service ranks, not a straw baseline."""
    from backend.rag.hybrid_service import HybridRAGService, RAGDocument
    class Cached:
        model_name = "BAAI/bge-small-zh-v1.5"
        dim = 512
        fallback_reason = None
        def encode(self, texts):
            rows = [Counter(_tokenize(t)) for t in texts]
            return EmbeddingResult([vectors[sha(t)].tolist() for t in texts],
                                   [SparseVector({k: v / max(c.values(), default=1) for k,v in c.items()}) for c in rows],
                                   self.model_name, self.dim)
    checked = 0
    for config in [BASELINE, CJK]:
        engine = Engine(data["documents"], vectors, config)
        os.environ["RAG_LEXICAL_MODE"] = config["lexical"]
        sample = [q for lang in ("en", "zh") for q in [x for x in data["questions"] if x["split"] == "dev" and x["language"] == lang][:8]]
        for q in sample:
            service = HybridRAGService(backend="memory", vector_dim=512, rrf_k=60, embedder=Cached())
            service.ingest_documents([RAGDocument(collection=d["collection"], scope="persistent", source_id=d["id"],
                                      content=d["plain"]) for d in engine.groups[q["collection"]]])
            reference = service.hybrid_search(q["question"], collection=q["collection"], top_k=4)
            candidate = engine.search(q, vectors[sha(q["question"])])
            assert [d["id"] for d in candidate] == [d["source_id"] for d in reference], q["id"]
            checked += 1
    return checked


def cluster_ci(before, after, metric):
    old = {r["case_id"]: r for r in before}
    groups = defaultdict(list)
    for r in after:
        groups[(r["dataset"], r["group"])].append(r[metric] - old[r["case_id"]][metric])
    values = list(groups.values()); rng = np.random.default_rng(20260914)
    samples = []
    for _ in range(2000):
        draw = [v for index in rng.integers(0, len(values), len(values)) for v in values[index]]
        samples.append(mean(draw))
    return dict(delta=mean(v for group in values for v in group),
                lower=float(np.quantile(samples, .025)), upper=float(np.quantile(samples, .975)),
                clusters=len(values), resamples=2000)


def code_identity():
    paths = [Path(__file__), ROOT / "backend/rag/bm25.py", ROOT / "backend/rag/lexical.py"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def grouped_summary(rows):
    by_dataset = {dataset: aggregate([r for r in rows if r["dataset"] == dataset]) for dataset in sorted({r["dataset"] for r in rows})}
    return dict(overall=aggregate(rows), by_dataset=by_dataset,
                macro_dataset={m: mean(s[m] for s in by_dataset.values()) for m in ("recall_at_k", "all_evidence", "ndcg_at_k", "mrr")},
                by_language={lang: aggregate([r for r in rows if r["language"] == lang]) for lang in ("en", "zh")},
                by_category={cat: aggregate([r for r in rows if r["category"] == cat]) for cat in sorted({r["category"] for r in rows})})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=["dev", "test"])
    args = parser.parse_args()
    payload = (HERE / "dataset.json").read_text(encoding="utf-8")
    data = json.loads(payload); dataset_hash = sha(payload)
    questions = [q for q in data["questions"] if q["split"] == args.mode]
    selection_path = HERE / "selection.json"
    if args.mode == "test":
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        assert selection["dataset_sha256"] == dataset_hash and selection["code_sha256"] == code_identity()
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    encoder = SharedEmbedding(ROOT / ".annual-models")
    vectors = prepare_vectors(encoder, data["documents"], questions)
    if args.mode == "dev":
        parity = parity_check(data, vectors)
        configs = [BASELINE, CJK] + [dict(field=f, lexical="bm25", rrf_k=k, sparse_weight=w)
                                      for f in ("plain", "structured") for k in (20, 60) for w in (.5, 1.0, 2.0)]
        # Test retaining raw-row semantics while headers enrich only lexical matching.
        configs += [dict(field="structured", dense_field="plain", lexical="bm25", rrf_k=k, sparse_weight=w)
                    for k in (20, 60) for w in (.5, 1.0, 2.0)]
        candidates = []
        for c in configs:
            rows = evaluate(Engine(data["documents"], vectors, c), questions, vectors, encoding)
            summaries = {language: aggregate([r for r in rows if r["language"] == language]) for language in ("en", "zh")}
            # Equal language weighting prevents the larger English slice hiding Chinese regressions.
            score = mean(s[m] for s in summaries.values() for m in ("recall_at_k", "all_evidence", "ndcg_at_k"))
            candidates.append(dict(config=c, score=score, overall=aggregate(rows), by_language=summaries))
            print(json.dumps(candidates[-1]), flush=True)
        base = candidates[0]["overall"]
        eligible = [c for c in candidates if all(c["overall"][m] >= base[m] - 1e-12
                    for m in ("recall_at_k", "all_evidence", "ndcg_at_k"))
                    and all(c["by_language"][lang][m] >= candidates[0]["by_language"][lang][m] - 1e-12
                            for lang in ("en", "zh") for m in ("recall_at_k", "all_evidence", "ndcg_at_k"))]
        winner = max(eligible, key=lambda c: c["score"])
        routed = {}
        for language in ("en", "zh"):
            safe = [c for c in candidates if all(c["by_language"][language][m] >= candidates[0]["by_language"][language][m] - 1e-12
                    for m in ("recall_at_k", "all_evidence", "ndcg_at_k"))]
            best = max(safe, key=lambda c: mean(c["by_language"][language][m] for m in ("recall_at_k", "all_evidence", "ndcg_at_k")))
            routed[language] = best["config"]
        save(selection_path, dict(status="frozen_before_test", created_at=datetime.now(timezone.utc).isoformat(),
                                 dataset_sha256=dataset_hash, code_sha256=code_identity(), config={"by_language": routed},
                                 uniform_config=winner["config"],
                                 selection_rule="per-language no recall/complete/nDCG regression vs baseline, then mean of these metrics; query-only language routing; also retain best safe uniform config", candidates=candidates,
                                 parity_checks=parity, dev_questions=len(questions)))
        print("FROZEN", routed, "uniform", winner["config"], flush=True)
    else:
        out = HERE / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out.mkdir(parents=True, exist_ok=False)
        configs = {"baseline": BASELINE, "cjk": CJK, "uniform_bm25": selection["uniform_config"], "selected": selection["config"]}
        results = {}
        # Round-robin order per query avoids consistent per-arm timing position.
        engines = {name: Engine(data["documents"], vectors, cfg) for name, cfg in configs.items()}
        results = {name: [] for name in configs}
        for i, q in enumerate(questions):
            names = list(configs); names = names[i % len(names):] + names[:i % len(names)]
            for name in names:
                results[name].extend(evaluate(engines[name], [q], vectors, encoding, encoder=encoder))
            if i % 20 == 0:
                print("test", i + 1, "/", len(questions), flush=True)
        report = dict(status="complete", dataset_sha256=dataset_hash, selection=selection, k=4, context_budget_tokens=1500,
                      model="BAAI/bge-small-zh-v1.5", no_reranker=True, rows=results,
                      summary={name: grouped_summary(rows) for name, rows in results.items()},
                      delta_ci95={metric: cluster_ci(results["baseline"], results["selected"], metric)
                                  for metric in ("recall_at_k", "all_evidence", "ndcg_at_k")},
                      delta_ci95_by_dataset={dataset: {metric: cluster_ci(
                          [r for r in results["baseline"] if r["dataset"] == dataset],
                          [r for r in results["selected"] if r["dataset"] == dataset], metric)
                          for metric in ("recall_at_k", "all_evidence", "ndcg_at_k")}
                          for dataset in sorted({q["dataset"] for q in questions})})
        save(out / "report.json", report)
        save(HERE / "assessment.json", {k: v for k, v in report.items() if k != "rows"})
        print(json.dumps(report["summary"], indent=2), flush=True)
        print("report", out / "report.json", flush=True)


if __name__ == "__main__":
    main()
