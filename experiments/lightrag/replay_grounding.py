"""Replay mix answers from saved retrieval, without indexing or remeasuring retrieval.

This is a targeted provenance repair check, not a new four-arm performance run.
It deliberately does not repeat the unreliable same-model quality judge.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.lightrag.adapters import CallBudget, ChatClient, endpoint_config
from experiments.lightrag.metrics import citation_metrics
from experiments.lightrag.run import ANSWER_SYSTEM, pack_context


async def replay(report_path: Path, retry_errors: bool = False):
    import tiktoken
    report = json.loads(report_path.read_text(encoding="utf-8"))
    dataset = json.loads((report_path.parent / "dataset.json").read_text(encoding="utf-8"))
    docs = {d["id"]: d["text"] for d in dataset["documents"]}
    cases = {c["id"]: c for c in dataset["cases"]}
    tokenizer = tiktoken.get_encoding("cl100k_base")
    client = ChatClient(endpoint_config(), CallBudget(12), 60)
    if client.config["model"] != report["answer_model"]:
        await client.close()
        raise ValueError("回放模型必须与原运行一致")
    result = {"scope": "mix_answer_grounding_replay", "source_run": report_path.parent.name,
              "status": "running", "retrieval_repeated": False, "judge_repeated": False,
              "model": client.config["model"], "results": []}
    output = report_path.parent / "grounding-replay.json"
    previous_calls = []
    retry_ids = None
    if retry_errors:
        result = json.loads(output.read_text(encoding="utf-8"))
        retry_ids = {r["case_id"] for r in result["results"] if r.get("error")}
        previous_calls = result.get("calls", [])
        result["status"] = "running"
        result["retry_source"] = output.name
        output = output.with_name("grounding-replay-retry.json")
    if output.exists():
        await client.close()
        raise FileExistsError("回放记录已存在，不能覆盖")
    try:
        for old in report["results"]:
            if old["arm"] != "lightrag_mix":
                continue
            if retry_ids is not None and old["case_id"] not in retry_ids:
                continue
            case = cases[old["case_id"]]
            data = (old.get("raw_retrieval") or {}).get("data") or {}
            retrieval = {"sources": old["retrieved_sources"],
                         "graph": {k: data.get(k, []) for k in ("entities", "relationships")}}
            context, supplied = pack_context(retrieval, docs, tokenizer, report["config"]["context_tokens"])
            client.phase = "grounding_replay:" + old["case_id"]
            row = {"case_id": old["case_id"], "supplied_sources": supplied,
                   "context_tokens": len(tokenizer.encode(context)), "context": context,
                   "before_citation_id_validity": old.get("citation_id_validity")}
            started = time.perf_counter()
            try:
                answer = await client.complete(f"问题：{case['question']}\n资料：\n{context}",
                                               system_prompt=ANSWER_SYSTEM, max_tokens=1600)
                row.update(answer=answer, answer_ms=(time.perf_counter() - started) * 1000,
                           **citation_metrics(answer, supplied))
            except Exception as exc:
                row["error"] = type(exc).__name__
            if retry_ids is not None:
                row["retried_after_error"] = next(r["error"] for r in result["results"] if r["case_id"] == row["case_id"])
                result["results"] = [row if r["case_id"] == row["case_id"] else r for r in result["results"]]
            else:
                result["results"].append(row)
            result["calls"] = previous_calls + client.calls
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(old["case_id"], flush=True)
        result["status"] = "partial" if any(r.get("error") for r in result["results"]) else "complete"
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = type(exc).__name__
        raise
    finally:
        await client.close()
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)
    if result["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--retry-errors", action="store_true", help="Only retry failed answers; retain original replay artifact")
    args = parser.parse_args()
    asyncio.run(replay(args.report, args.retry_errors))
