"""Prepare a reproducible evidence-retrieval adaptation of public finance QA.

Ground truth is used only to label questions, never to render/rank corpus chunks.
All available pages in a chosen report enter its collection before questions are
selected. This tests within-report evidence retrieval, not report routing.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from decimal import Decimal, InvalidOperation

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
SEED = "finsight-real-finance-v1-20260914"


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def order(values, key=lambda x: x):
    return sorted(values, key=lambda x: digest(SEED + str(key(x))))


def render_row(table, row_index, structured):
    if not structured:
        return " | ".join(str(v) for v in table[row_index])
    from backend.rag.table_context import render_table_row
    return render_table_row(table, row_index)


def render_row_legacy(table, row_index, structured):
    """Historical v1 representation, retained only to reproduce frozen baselines."""
    row = [str(v) for v in table[row_index]]
    header = [str(v) for v in table[0]]
    if not structured or row_index == 0:
        return " | ".join(row)
    # Carry multi-row column headings (e.g. period title followed by years).
    for extra in table[1:]:
        if str(extra[0]).strip():
            break
        header = [" ".join(dict.fromkeys(v for v in [header[i] if i < len(header) else "", str(cell)] if v))
                  for i, cell in enumerate(extra)]
    label = row[0] if row else ""
    return label + "; " + "; ".join(
        f"{header[i] if i < len(header) and header[i] else 'column ' + str(i)}: {v}"
        for i, v in enumerate(row[1:], 1))


def finqa():
    records = []
    for split in ("train", "dev", "test"):
        for row in json.loads((RAW / f"finqa_dataset_{split}.json").read_text(encoding="utf-8")):
            records.append({**row, "original_split": split})
    groups = defaultdict(list)
    for r in records:
        if r.get("qa", {}).get("gold_inds"):
            groups[r["filename"].rsplit("/", 1)[0]].append(r)
    eligible = order([g for g, rows in groups.items() if len(rows) >= 5])
    selected_groups = {g: "dev" if i < 14 else "test" for i, g in enumerate(eligible[:40])}
    documents, questions = [], []
    for group, split in selected_groups.items():
        collection = "finqa:" + group
        pages = {r["filename"]: r for r in groups[group]}
        aliases = {}
        canonical = {}
        for page, r in sorted(pages.items()):
            blocks = [(f"text_{i}", text, text, "text")
                      for i, text in enumerate(r["pre_text"] + r["post_text"])]
            blocks += [(f"table_{i}", render_row(r["table"], i, False),
                        render_row_legacy(r["table"], i, True), "table") for i in range(len(r["table"]))]
            for key, plain, structured, kind in blocks:
                plain, structured = plain.strip(), structured.strip()
                if not re.search(r"[a-zA-Z0-9\u4e00-\u9fff]", plain):
                    continue
                identity = digest(collection + "\n" + structured)
                sid = "F" + identity[:18]
                aliases[(page, key)] = sid
                if sid in canonical:
                    canonical[sid]["locations"].append({"page": page, "evidence_key": key})
                else:
                    item = dict(id=sid, collection=collection, language="en", kind=kind,
                                plain=plain, structured=structured,
                                locations=[{"page": page, "evidence_key": key}])
                    canonical[sid] = item
                    documents.append(item)
        chosen = order(groups[group], key=lambda r: r["id"])[:5]
        for r in chosen:
            qa = r["qa"]
            required = sorted({aliases[(r["filename"], key)] for key in qa["gold_inds"]})
            questions.append(dict(id="finqa:" + r["id"], dataset="FinQA", language="en", split=split,
                                  collection=collection, group=group, original_split=r["original_split"],
                                  question=qa["question"].strip(), answer=qa.get("answer"),
                                  numeric_answer=qa.get("exe_ans"), program=qa.get("program"),
                                  required=required, category="multi_evidence" if len(required) > 1 else "single_evidence",
                                  source_page=r["filename"], source_evidence_keys=list(qa["gold_inds"])))
    return documents, questions


def numeric_cell(value, *, absolute=True):
    raw = str(value).replace(",", "").replace("$", "").replace("%", "").strip()
    raw = raw.replace("(", "-").replace(")", "")
    raw = re.sub(r"^(\d{4})\s*年$", r"\1", raw)
    try:
        number = Decimal(raw)
        return abs(number) if absolute else number
    except InvalidOperation:
        return None


def multitat():
    original = json.loads((RAW / "tatqa_dataset_raw_tatqa_dataset_dev.json").read_text(encoding="utf-8"))
    lookup = {q["uid"]: (r, q) for r in original for q in r["questions"]}
    translated = json.loads((RAW / "multitat_dataset_dev.zh.json").read_text(encoding="utf-8"))
    documents, questions, excluded = [], [], []
    by_context = {}
    pending = []
    for item in translated:
        if not item["source"]["dataset"].startswith("TAT-QA"):
            continue
        qid = item["source"]["qid"]
        if qid not in lookup:
            excluded.append(dict(qid=qid, reason="original_qid_missing")); continue
        context, q = lookup[qid]
        gid = context["table"]["uid"]
        table = item["table"]["content"]
        paragraphs = item["text"]["paragraph"]
        original_table = context["table"]["table"]
        original_paragraphs = sorted(context["paragraphs"], key=lambda p: p["order"])
        if ([len(r) for r in table] != [len(r) for r in original_table]
                or len(paragraphs) != len(original_paragraphs)):
            excluded.append(dict(qid=qid, reason="translation_structure_mismatch")); continue
        # Require numerical cells to survive translation unchanged.
        if any(numeric_cell(a, absolute=False) != numeric_cell(b, absolute=False) for row_a, row_b in zip(table, original_table)
               for a, b in zip(row_a, row_b) if numeric_cell(a) is not None or numeric_cell(b) is not None):
            excluded.append(dict(qid=qid, reason="translation_numeric_mismatch")); continue
        required_rows = set()
        if q["answer_from"] != "text":
            if q["answer_type"] != "arithmetic" or q.get("req_comparison"):
                excluded.append(dict(qid=qid, reason="row_evidence_requires_manual_span_count_annotation")); continue
            operands = {numeric_cell(n) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", q["derivation"])}
            cells = defaultdict(set)
            for i, row in enumerate(original_table[1:], 1):
                for cell in row[1:]:
                    n = numeric_cell(cell)
                    if n is not None:
                        cells[n].add(i)
            unresolved = False
            for n in operands:
                matched = cells.get(n, set())
                if len(matched) == 1:
                    required_rows.update(matched)
                elif not matched and n in {Decimal(100), Decimal(1000), Decimal(1000000)}:
                    pass  # percentage/unit conversion constants, not evidence values
                else:
                    unresolved = True
            if unresolved or not required_rows:
                excluded.append(dict(qid=qid, reason="ambiguous_or_unmapped_derivation_operand")); continue
        paragraph_orders = {str(p["order"]): i for i, p in enumerate(original_paragraphs)}
        required_paragraphs = [paragraph_orders[str(v)] for v in q.get("rel_paragraphs", [])]
        if not required_rows and not required_paragraphs:
            excluded.append(dict(qid=qid, reason="no_explicit_evidence")); continue
        # Multiple translations of one context must not become conflicting copies.
        context_key = json.dumps([table, paragraphs], ensure_ascii=False)
        if gid in by_context and by_context[gid]["raw"] != context_key:
            excluded.append(dict(qid=qid, reason="conflicting_translation_of_same_context")); continue
        by_context[gid] = dict(table=table, paragraphs=paragraphs, raw=context_key)
        pending.append((item, q, gid, required_rows, required_paragraphs))
    group_order = order(list(by_context))
    dev_groups = set(group_order[:max(1, len(group_order) // 3)])
    for gid, ctx in by_context.items():
        for i, row in enumerate(ctx["table"]):
            documents.append(dict(id=f"M{gid}:table_{i}", collection="multitat:" + gid, language="zh",
                                  kind="table", plain=render_row(ctx["table"], i, False),
                                  structured=render_row_legacy(ctx["table"], i, True),
                                  locations=[dict(context_uid=gid, table_row=i)]))
        for i, paragraph in enumerate(ctx["paragraphs"]):
            documents.append(dict(id=f"M{gid}:text_{i}", collection="multitat:" + gid, language="zh",
                                  kind="text", plain=paragraph, structured=paragraph,
                                  locations=[dict(context_uid=gid, paragraph_index=i)]))
    for item, q, gid, rows, paragraphs in pending:
        questions.append(dict(id="multitat:" + q["uid"], dataset="MultiTAT-TATQA", language="zh",
                              split="dev" if gid in dev_groups else "test", collection="multitat:" + gid,
                              group=gid, question=item["question"], answer=item["answer"],
                              numeric_answer=item["answer"] if isinstance(item["answer"], (int, float)) else None,
                              required=sorted([f"M{gid}:table_{i}" for i in rows] + [f"M{gid}:text_{i}" for i in paragraphs]),
                              category=q["answer_type"], original_question=q["question"], scale=q.get("scale"),
                              derivation=q.get("derivation"), annotation="original_paragraph_orders+unambiguous_derivation_rows"))
    return documents, questions, excluded


def main():
    documents, questions = finqa()
    md, mq, excluded = multitat()
    documents += md
    questions += mq
    for d in documents:
        d["plain"], d["structured"] = d["plain"].strip(), d["structured"].strip()
    for q in questions:
        q["question"] = q["question"].strip()
    manifest = dict(version=SEED, status="prepared", documents=documents, questions=questions, excluded=excluded,
                    sources=[dict(name="FinQA", url="https://github.com/czyssrs/FinQA",
                                  revision=(RAW / "finqa_commit.txt").read_text().strip(), license="MIT"),
                             dict(name="MultiTAT", url="https://github.com/zhxlia/MULTITAT",
                                  revision=(RAW / "multitat_commit.txt").read_text().strip(),
                                  license="No explicit repository license found; local evaluation only; do not redistribute"),
                             dict(name="TAT-QA", url="https://github.com/NExTplusplus/TAT-QA",
                                  revision=(RAW / "tatqa_commit.txt").read_text().strip(), license="CC-BY-4.0")])
    validate(manifest)
    (HERE / "dataset.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"documents": len(documents), "questions": len(questions),
                      "splits": dict(Counter(q["split"] for q in questions)),
                      "language": dict(Counter(q["language"] for q in questions)),
                      "excluded": dict(Counter(r["reason"] for r in excluded)),
                      "gold_counts": dict(Counter(len(q["required"]) for q in questions))}, ensure_ascii=False))


def validate(data):
    docs = {d["id"]: d for d in data["documents"]}
    assert len(docs) == len(data["documents"])
    group_splits = defaultdict(set)
    for q in data["questions"]:
        group_splits[(q["dataset"], q["group"])].add(q["split"])
        assert q["required"]
        assert all(s in docs and docs[s]["collection"] == q["collection"] for s in q["required"])
    assert all(len(s) == 1 for s in group_splits.values()), "Group leakage across splits"
    ids = [q["id"] for q in data["questions"]]
    assert len(ids) == len(set(ids))


if __name__ == "__main__":
    main()
