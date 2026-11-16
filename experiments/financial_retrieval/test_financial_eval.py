"""Dataset and benchmark integrity, without network or LLM calls."""
from decimal import Decimal
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.financial_retrieval.build_dataset import numeric_cell, render_row, validate
from experiments.financial_retrieval.evaluate import Engine, BASELINE, grouped_summary


def test_year_translation_and_negative_values():
    assert numeric_cell("2019 年", absolute=False) == numeric_cell("2019", absolute=False)
    assert numeric_cell("($1,200)", absolute=False) == Decimal(-1200)
    assert numeric_cell("-12", absolute=False) != numeric_cell("12", absolute=False)


def test_all_table_rows_use_same_serializer_without_gold():
    table = [["", "Year ended", "Year ended"], ["", "2019", "2018"], ["revenue", "30", "20"], ["cost", "10", "9"]]
    assert "2019: 30" in render_row(table, 2, True)
    assert "2019: 10" in render_row(table, 3, True)
    assert render_row(table, 2, False) == "revenue | 30 | 20"


def test_language_routing_cannot_read_labels():
    assert Engine.query_language({"question": "收入多少", "language": "en", "required": []}) == "zh"
    assert Engine.query_language({"question": "revenue", "language": "zh", "required": ["secret"]}) == "en"


def test_search_does_not_depend_on_gold():
    from experiments.financial_retrieval.evaluate import sha
    docs = [dict(id="a", collection="same", plain="revenue growth"), dict(id="b", collection="same", plain="weather")]
    vectors = {sha("revenue growth"): np.array([1.,0.]), sha("weather"): np.array([0.,1.])}
    engine = Engine(docs, vectors, BASELINE)
    query = {"collection": "same", "question": "revenue", "required": ["b"]}
    before = engine.search(query, [1., 0.])
    query["required"] = ["a"]
    assert engine.search(query, [1.,0.]) == before


def test_prepared_data_integrity():
    path = Path(__file__).parent / "dataset.json"
    if not path.exists():
        pytest.skip("Run fetch_sources.py and build_dataset.py for local licensed data first")
    data = json.loads(path.read_text(encoding="utf-8"))
    validate(data)
    assert len(data["questions"]) >= 200
    assert {q["language"] for q in data["questions"]} == {"en", "zh"}
    # The data include distracting evidence, not just the gold answer passages.
    gold = {s for q in data["questions"] for s in q["required"]}
    assert len(data["documents"]) > 5 * len(gold)
