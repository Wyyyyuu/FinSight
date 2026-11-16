"""Conservative financial-table rendering and bounded source context.

Only source cells and nearby source paragraphs are used; no question or gold label
is accepted here. Amounts must never become column headings. Unknown headings
remain numbered columns instead of being guessed.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

_NUMBER = re.compile(r"^[\s$€£¥￥()−+\-]*\d[\d,.]*(?:\s*%|\s*(?:million|billion|千|万|亿))?[\s)]*$", re.IGNORECASE)
_YEAR = re.compile(r"^(?:fy\s*)?(?:19|20)\d{2}(?:\s*年)?$", re.IGNORECASE)


def header_rows(table: Sequence[Sequence[str]]) -> list[int]:
    """Recognize only a leading run of headings, including split year headings."""
    result = []
    for i, cells in enumerate(table):
        values = [str(v).strip() for v in cells]
        if len(values) < 2:
            break
        payload = [v for v in values[1:] if v and v not in {"-", "—", "–"}]
        if not payload:
            break
        # Currency-marked four-digit amounts are NOT years.
        if any(_NUMBER.fullmatch(v) and not _YEAR.fullmatch(v) for v in payload):
            break
        if all(_YEAR.fullmatch(v) for v in payload) and values[0] and not re.search(
            r"year|period|fiscal|ended|项目|年度|年份", values[0], re.IGNORECASE
        ):
            break
        if i and values[0]:
            break
        result.append(i)
    return result


def render_table_row(table: Sequence[Sequence[str]], row_index: int) -> str:
    row = [str(v).strip() for v in table[row_index]]
    headings = header_rows(table)
    if row_index in headings:
        return " | ".join(row)
    labels = []
    for column in range(1, len(row)):
        pieces = list(dict.fromkeys(str(table[i][column]).strip() for i in headings
                                    if column < len(table[i]) and str(table[i][column]).strip()))
        labels.append(" ".join(pieces) or f"column {column}")
    return "; ".join([row[0] if row else ""] + [f"{label}: {value}" for label, value in zip(labels, row[1:])])


def table_topic(pre_text: Sequence[str], *, max_words: int = 65) -> str:
    """Keep the nearest substantive source paragraph, without inventing a title."""
    for raw in reversed(pre_text):
        value = " ".join(str(raw).split())
        if len(value) < 25 or not re.search(r"[a-zA-Z\u3400-\u9fff]", value):
            continue
        words = value.split()
        return " ".join(words[-max_words:])
    return ""


def contextual_table_row(table: Sequence[Sequence[str]], row_index: int,
                         pre_text: Sequence[str]) -> str:
    topic = table_topic(pre_text)
    row = render_table_row(table, row_index)
    return f"Table context: {topic}\n{row}" if topic else row
