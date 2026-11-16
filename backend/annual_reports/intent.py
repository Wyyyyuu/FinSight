"""Optional local classifiers and guards for executable annual-report intents."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

LABELS = ("lookup", "yoy", "open")
MAX_TOKENS = 96
LocalIntentSource = Literal["tinybert", "minirbt", "mmbert", "qwen"]
LOCAL_INTENT_MODELS: dict[str, LocalIntentSource] = {
    "huawei-noah/TinyBERT_4L_zh": "tinybert",
    "hfl/minirbt-h256": "minirbt",
    "jhu-clsp/mmBERT-small": "mmbert",
    "Qwen/Qwen3-0.6B-Base": "qwen",
}
LOCAL_INTENT_NAMES = {"tinybert": "TinyBERT", "minirbt": "MiniRBT", "mmbert": "mmBERT", "qwen": "Qwen"}

_PREVIOUS = r"同比|上(?:一(?:个)?|个)?(?:财)?年(?:度|同期)?|去年(?:同期)?|前一(?:个)?年度"
_QUESTION_PREFIX = r"(?:(?:请|帮我|麻烦|我|能否|能)?(?:想知道|想了解|想问|查一下|查下|说一下|说下|看看|告诉我|给我|告知|请问))?"
_ASK = r"(?:的|一共|总共|合计|大约|大概|约|到底|究竟|实际|具体|是|为|有|达|到|多少钱|多少(?:亿(?:元)?|万(?:元)?)?|什么|啥|数字|数据|数额|总数|数|金额|水平|规模|多大|量|了|啊|呀|呢|吗)"
_CHANGE = r"(?:和|与|跟|比较|一比|比|相较于|相比|较|对照|对比|看|下来|多出|高出|多|少|高|低|差额|差|净增|增|减|涨|跌|变化|变动|增长|下降|增加|减少|升|降|幅度|幅|百分点|百分比|个点|百分之几|几个点|几个百分点|几亿)"
_OPERATION_WORD = re.compile(rf"{_PREVIOUS}|{_ASK}|{_CHANGE}")
_LEADING_COMPARISON = re.compile(rf"(?:与|和|跟|比|较|对照)(?:{_PREVIOUS})(?:相比|比较|对照|比)?[，,]")


@dataclass(frozen=True)
class MetricIntent:
    operation: str
    company: str
    metric: str
    years: tuple[int, ...]


@lru_cache(maxsize=2)
def _classifier_metadata(directory: Path) -> dict:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError("Invalid intent classifier metadata")
    return metadata


def local_intent_source(directory: Path) -> LocalIntentSource:
    try:
        base_model = _classifier_metadata(directory).get("base_model")
    except (OSError, TypeError, ValueError):
        return "tinybert"
    return LOCAL_INTENT_MODELS.get(base_model, "tinybert") if isinstance(base_model, str) else "tinybert"


@lru_cache(maxsize=2)
def _load_classifier(directory: Path):
    import onnxruntime as ort
    from tokenizers import Tokenizer

    metadata = _classifier_metadata(directory)
    if metadata.get("labels") != list(LABELS):
        raise ValueError("Unexpected intent classifier labels")
    threshold = float(metadata["threshold"])
    if not 0.5 <= threshold <= 1.0:
        raise ValueError("Invalid intent classifier threshold")
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(directory / "model.onnx"), options, providers=["CPUExecutionProvider"])
    return tokenizer, session, threshold


def predict_intent(question: str, directory: Path) -> str | None:
    """Run only local artifacts. A missing or invalid model leaves routing to the caller."""
    try:
        import numpy as np

        tokenizer, session, threshold = _load_classifier(directory)
        encoded = tokenizer.encode(question)
        if not 0 < len(encoded.ids) <= MAX_TOKENS:
            return None
        features = {name: np.asarray([value], dtype=np.int64) for name, value in (
            ("input_ids", encoded.ids), ("attention_mask", encoded.attention_mask),
        )}
        logits = session.run(["logits"], features)[0][0]
        if logits.shape != (len(LABELS),) or not np.isfinite(logits).all():
            return None
        scores = np.exp(logits - logits.max())
        scores /= scores.sum()
        index = int(scores.argmax())
        confidence = float(scores[index])
        return LABELS[index] if confidence >= threshold else None
    except Exception:  # noqa: BLE001 - optional local artifacts and runtime must fail closed.
        return None


def local_metric_intent(question: str, companies: set[str],
                        metrics: dict[str, tuple[str, ...]]) -> tuple[MetricIntent, LocalIntentSource] | None:
    model_dir = os.getenv("ANNUAL_REPORT_INTENT_MODEL_DIR", "").strip()
    if not model_dir:
        return None
    directory = Path(model_dir)
    intent = guarded_metric_intent(question, companies, metrics, lambda text: predict_intent(text, directory))
    return (intent, local_intent_source(directory)) if intent is not None else None


def normalize_intent_question(question: str) -> str:
    return re.sub(r"\s+", "", question).rstrip("？?。！!")


def guarded_metric_intent(question: str, companies: set[str], metrics: dict[str, tuple[str, ...]],
                          propose_operation: Callable[[str], str | None]) -> MetricIntent | None:
    """Validate both model paths against the original company, year and metric wording."""
    if len(companies) != 1 or len(question) > 80:
        return None
    company, = companies
    short = re.sub(r"(?:股份有限公司|有限责任公司|有限公司)$", "", company)
    company_pattern = "|".join(map(re.escape, sorted({company, short, short.removesuffix("集团")} - {""}, key=len, reverse=True)))
    aliases = {alias: metric for metric, names in metrics.items() for alias in names}
    metric_pattern = "|".join(map(re.escape, sorted(aliases, key=len, reverse=True)))
    compact = normalize_intent_question(question)
    comparison = _LEADING_COMPARISON.match(compact)
    body = compact[comparison.end():] if comparison else compact
    subject = rf"{_QUESTION_PREFIX}(?:(?:{company_pattern})(?:的)?)?"
    year_pattern = r"(?P<year>(?:19|20)\d{2})年(?:度)?"
    metric_pattern = rf"(?P<metric>{metric_pattern})"
    match = re.fullmatch(rf"{subject}{year_pattern}(?:的)?{metric_pattern}(?P<ask>.+)", body)
    if not match:
        match = re.fullmatch(rf"{subject}{metric_pattern}(?:在|于){year_pattern}(?P<ask>.+)", body)
    if not match:
        return None
    ask = match["ask"]
    # Only accept amount/change vocabulary after the exact metric. Qualifiers,
    # causal questions, unrelated tasks, and other entities need semantic planning.
    if "".join(word.group() for word in _OPERATION_WORD.finditer(ask)) != ask:
        return None
    operation = propose_operation(compact)
    year = int(match["year"])
    has_prior_period = comparison is not None or bool(re.search(_PREVIOUS, ask))
    if operation == "yoy" and year > 1900 and has_prior_period:
        return MetricIntent("yoy", company, aliases[match["metric"]], (year - 1, year))
    if operation == "lookup" and not has_prior_period and re.fullmatch(rf"{_ASK}+", ask):
        return MetricIntent("lookup", company, aliases[match["metric"]], (year,))
    return None
