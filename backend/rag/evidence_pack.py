"""Token-bounded evidence packing, without neural or LLM reranking.

Callers provide ranked source records and an exact serialized token counter.
Parent expansion stays within collection + table identity. Round-robin packing
prevents one requested facet from consuming the entire summary budget.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence


def pack_evidence(ranked: Sequence[dict], *, all_documents: Sequence[dict],
                  token_count: Callable[[dict], int], budget: int,
                  expand_tables: bool = False) -> list[dict]:
    if budget < 1:
        raise ValueError("Evidence budget must be positive")
    parents: dict[tuple, list[dict]] = {}
    if expand_tables:
        for d in all_documents:
            for parent in d.get("table_ids", [d.get("table_id")]):
                if parent:
                    parents.setdefault((d["collection"], parent), []).append(d)
        for (_, parent), rows in parents.items():
            rows.sort(key=lambda d: (d.get("parent_rows", {}).get(parent, d.get("row_index", 0)), d["id"]))
    selected, seen, used = [], set(), 0

    def add(d):
        nonlocal used
        identity = (d["collection"], d["id"])
        if identity in seen:
            return
        size = token_count(d)
        if size < 0:
            raise ValueError("Negative token count")
        if used + size <= budget:
            selected.append(d); seen.add(identity); used += size

    for d in ranked:
        add(d)
        if expand_tables and (d["collection"], d["id"]) in seen and d.get("table_id"):
            rows = parents.get((d["collection"], d["table_id"]), [])
            # A complete small table gives the reader denominators and units.
            # Large tables use bounded neighboring rows, never guessed values.
            if sum(token_count(r) for r in rows) <= budget // 2:
                for row in rows:
                    add(row)
            else:
                nearby = sorted(rows, key=lambda r: (abs(r.get("row_index", 0)-d.get("row_index", 0)), r["id"]))
                for row in nearby[:3]:
                    add(row)
    return selected


def balanced_evidence(rankings: Sequence[Sequence[dict]], *, all_documents: Sequence[dict],
                      token_count: Callable[[dict], int], budget: int,
                      expand_tables: bool = True) -> list[dict]:
    if not rankings:
        return []
    if budget < 1:
        raise ValueError("Evidence budget must be positive")
    quota = budget // len(rankings)
    per_facet = [pack_evidence(rows, all_documents=all_documents, token_count=token_count,
                              budget=quota, expand_tables=expand_tables) if quota else []
                 for rows in rankings]
    selected, seen = [], set()
    for position in range(max(map(len, per_facet), default=0)):
        for rows in per_facet:
            if position < len(rows):
                d = rows[position]
                identity = (d["collection"], d["id"])
                if identity not in seen:
                    selected.append(d); seen.add(identity)
    # Spend unused quota on a round-robin tail; identities remain source-bound.
    for position in range(max(map(len, rankings), default=0)):
        for rows in rankings:
            if position < len(rows):
                d = rows[position]
                identity = (d["collection"], d["id"])
                if identity not in seen:
                    selected.append(d); seen.add(identity)
    return pack_evidence(selected, all_documents=all_documents, token_count=token_count,
                         budget=budget, expand_tables=False)
