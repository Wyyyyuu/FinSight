"""Small paired numeric-answer audit using the existing configured experiment LLM.

No LLM judge. Numeric targets come from the source dataset; citation checks are
identity checks only, not a claim that every citation entails the answer.
"""
from __future__ import annotations
import asyncio
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np
from experiments.financial_retrieval.evaluate import BASELINE, Engine, sha, save
from experiments.lightrag.adapters import ChatClient, CallBudget, endpoint_config


def target_number(q):
    value = q.get("answer")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").replace("%", "").replace("$", "")
        if re.fullmatch(r"-?\d+(?:\.\d+)?", cleaned):
            return float(cleaned)
    return None


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retry-errors", type=Path)
    parser.add_argument("--context", choices=["retrieval", "structured"], default="retrieval")
    args = parser.parse_args()
    previous = json.loads(args.retry_errors.read_text(encoding="utf-8")) if args.retry_errors else None
    if previous:
        assert previous.get("context_mode", "retrieval") == args.context
    data = json.loads((HERE / "dataset.json").read_text(encoding="utf-8"))
    selection = json.loads((HERE / "selection.json").read_text(encoding="utf-8"))
    assessment = json.loads((HERE / "assessment.json").read_text(encoding="utf-8"))
    assert assessment["status"] == "complete"
    assert selection["dataset_sha256"] == assessment["dataset_sha256"]
    candidates = [q for q in data["questions"] if q["split"] == "test" and target_number(q) is not None]
    chosen = []
    for language, limit in [("en", 12), ("zh", 8)]:
        chosen += sorted([q for q in candidates if q["language"] == language],
                         key=lambda q: sha("numeric-audit-v1" + q["id"]))[:limit]
    with np.load(HERE / "cache/bge-small-zh-46fbe35-vectors.npz", allow_pickle=False) as stored:
        vectors = dict(zip(stored["keys"].tolist(), stored["vectors"]))
    engines = {name: Engine(data["documents"], vectors, cfg)
               for name, cfg in {"baseline": BASELINE, "selected": selection["config"]}.items()}
    folder = HERE / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-answers")
    folder.mkdir(parents=True, exist_ok=False)
    client = ChatClient(endpoint_config(), CallBudget(48), timeout=60)
    client.phase = "numeric_answer_test"
    semaphore = asyncio.Semaphore(2)
    rows = [r for r in previous["rows"] if "error" not in r] if previous else []
    retry_keys = {(r["case_id"], r["arm"]) for r in previous["rows"] if "error" in r} if previous else None
    previous_calls = previous["calls"] if previous else []
    async def answer(q, arm):
        engine = engines[arm]
        hits = engine.search(q, vectors[sha(q["question"])])
        supplied = [d["id"] for d in hits]
        field = "structured" if args.context == "structured" else engine.context_field(q)
        context = "\n\n".join(f"[{d['id']}]\n{d[field]}" for d in hits)
        prompt = ("依据给定财报片段回答数值问题，不使用外部知识。证据不足时 answer 为 null。"
                  "百分比用百分数数值（如百分之十二写12）；其他金额保留原文单位。"
                  "只返回JSON：{\"answer\":数值或null,\"unit\":\"单位\",\"citations\":[\"原样证据ID\"],"
                  "\"calculation\":\"简短计算依据\"}。\n问题：" + q["question"] + "\n证据：\n" + context)
        record = dict(case_id=q["id"], arm=arm, language=q["language"], question=q["question"],
                      reference=target_number(q), supplied=supplied, required=q["required"],
                      complete_evidence=set(q["required"]) <= set(supplied))
        async with semaphore:
            started = time.perf_counter()
            try:
                text = await client.complete(prompt, max_tokens=1100)
                cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
                result = json.loads(cleaned)
                value = result.get("answer")
                numeric = float(value) if value is not None and not isinstance(value, bool) else None
                tolerance = max(.01, abs(record["reference"]) * .001)
                citations = result.get("citations") or []
                record.update(output=result, numeric_correct=(numeric is not None and abs(numeric-record["reference"]) <= tolerance),
                              tolerance=tolerance, citation_ids_valid=bool(citations) and all(c in supplied for c in citations))
            except Exception as exc:
                record.update(error=type(exc).__name__, numeric_correct=False, citation_ids_valid=None)
            record["elapsed_ms"] = (time.perf_counter()-started)*1000
            rows.append(record)
            save(folder / "partial.json", dict(rows=rows, calls=previous_calls + client.calls))
            print("answered", len(rows), "/", len(chosen)*2, arm, q["id"], flush=True)
    try:
        await asyncio.gather(*(answer(q, arm) for q in chosen for arm in engines
                               if retry_keys is None or (q["id"], arm) in retry_keys))
    finally:
        await client.close()
    summary = {}
    for arm in engines:
        subset = [r for r in rows if r["arm"] == arm]
        summary[arm] = dict(questions=len(subset), numeric_correct=sum(r["numeric_correct"] for r in subset),
                           citation_ids_valid=sum(r.get("citation_ids_valid") is True for r in subset),
                           errors=sum("error" in r for r in subset))
    calls = previous_calls + client.calls
    report = dict(status="partial" if any("error" in r for r in rows) else "complete", model=client.config["model"],
                  dataset_sha256=selection["dataset_sha256"], selection_config=selection["config"],
                  context_mode=args.context,
                  retry_of=str(args.retry_errors) if previous else None,
                  sample_rule="fixed SHA256 ordering, 12 English + 8 Chinese numeric test questions, paired",
                  numeric_tolerance="max(0.01, abs(reference)*0.001)", summary=summary, rows=rows,
                  calls=calls, prompt_tokens=sum(r.get("prompt_tokens") or 0 for r in calls),
                  completion_tokens=sum(r.get("completion_tokens") or 0 for r in calls),
                  limitations=["20-question numeric subset; no production capacity or broad answer-quality claim",
                               "Citation identity is not citation entailment; no model self-grading",
                               "Reference answers and evidence are never provided beyond retrieved context"])
    save(folder / "report.json", report)
    filename = "answer_assessment.json" if args.context == "retrieval" else "answer_structured_assessment.json"
    save(HERE / filename, {k:v for k,v in report.items() if k not in {"rows", "calls"}})
    print(json.dumps(summary), flush=True)
    print(folder / "report.json", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
