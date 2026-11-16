"""Deterministic evidence metrics; never treat missing measurements as zero."""
from __future__ import annotations

import math
import re
from statistics import mean


def retrieval_metrics(retrieved: list[str], required: list[str], k: int) -> dict:
    ranked = list(dict.fromkeys(retrieved))[:k]
    gold = set(required)
    if not gold:
        return dict(recall_at_k=None, precision_at_k=None, ndcg_at_k=None,
                    all_evidence=None)
    relevant = [int(item in gold) for item in ranked]
    dcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(relevant))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold))))
    return dict(recall_at_k=len(set(ranked) & gold) / len(gold),
                precision_at_k=sum(relevant) / k,
                ndcg_at_k=dcg / ideal,
                all_evidence=float(gold <= set(ranked)))


def citation_metrics(answer: str, supplied: list[str]) -> dict:
    # This checks reference identity only, not whether the source entails a claim.
    refs = re.findall(r"\[(S\d+)\]", answer)
    return dict(citation_count=len(refs),
                citation_id_validity=(sum(r in supplied for r in refs) / len(refs)
                                      if refs else None))


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize(rows: list[dict]) -> dict:
    result = {}
    metric_names = ("recall_at_k", "precision_at_k", "ndcg_at_k", "all_evidence",
                    "citation_id_validity", "faithfulness", "answer_coverage",
                    "citation_support", "abstention_correct")
    for arm in sorted({row["arm"] for row in rows}):
        subset = [row for row in rows if row["arm"] == arm]
        means = {}
        for name in metric_names:
            values = [row[name] for row in subset if row.get(name) is not None]
            means[name] = {"mean": mean(values) if values else None,
                           "measured": len(values), "total": len(subset)}
        for label, values in {
            "first_query_ms": [r["retrieval_ms"][0] for r in subset if r.get("retrieval_ms")],
            "warm_query_ms": [v for r in subset for v in r.get("retrieval_ms", [])[1:]],
            "answer_ms": [r["answer_ms"] for r in subset if "answer_ms" in r],
        }.items():
            means[label] = {"p50": percentile(values, .5), "p95": percentile(values, .95),
                            "measured": len(values)}
        means["errors"] = sum(bool(r.get("error") or r.get("judge_error")) for r in subset)
        result[arm] = means
    return result
