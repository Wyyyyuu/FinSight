"""Chinese lexical retrieval and optional, real semantic embeddings.

There is deliberately no synthetic vector fallback.  Callers can distinguish a
lexical-only answer from a successful semantic/hybrid search in every hit.
"""

from __future__ import annotations

import math
import os
import re
import threading
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any


def tokenize(text: str) -> list[str]:
    """Tokenize without a downloadable dictionary: CJK unigrams and bigrams.

    Bigrams preserve finance phrases such as 现金/营收 while unigrams keep
    previously unseen Chinese company names searchable. ASCII words and numbers
    remain whole tokens. This is BM25 tokenization, not semantic embedding.
    """
    result: list[str] = []
    for token in re.findall(
        r"[\u3400-\u9fff]+|[a-z0-9]+(?:[._%-][a-z0-9]+)*", text.lower()
    ):
        if re.match(r"[\u3400-\u9fff]", token):
            result.extend(token)
            result.extend(token[i : i + 2] for i in range(len(token) - 1))
        else:
            result.append(token)
    return result


def bm25_rank(query: str, chunks: Sequence[dict[str, Any]]) -> list[tuple[str, float]]:
    query_terms = set(tokenize(query))
    if not query_terms or not chunks:
        return []
    terms = [
        Counter(tokenize(f"{chunk.get('section', '')}\n{chunk['text']}"))
        for chunk in chunks
    ]
    lengths = [sum(item.values()) for item in terms]
    average = max(sum(lengths) / len(lengths), 1.0)
    document_frequency = Counter(
        term for item in terms for term in item if term in query_terms
    )
    ranked: list[tuple[str, float]] = []
    for chunk, frequencies, length in zip(chunks, terms, lengths):
        score = 0.0
        for term in query_terms:
            tf = frequencies.get(term, 0)
            if not tf:
                continue
            df = document_frequency[term]
            idf = math.log(1 + (len(chunks) - df + 0.5) / (df + 0.5))
            score += idf * (tf * 2.5) / (tf + 1.5 * (0.25 + 0.75 * length / average))
        if score > 0:
            ranked.append((chunk["id"], score))
    return sorted(ranked, key=lambda item: (-item[1], item[0]))


def reciprocal_rank_fusion(
    *rankings: Sequence[tuple[str, float]], k: int = 60
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (chunk_id, _score) in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class SemanticUnavailable(RuntimeError):
    """A real semantic model or valid embeddings are unavailable."""


class SemanticEmbedder:
    """Lazy optional fastembed model, or a caller-provided embedder for tests.

    Runtime never downloads weights by default. Enable fastembed explicitly with
    ANNUAL_REPORT_EMBEDDING_PROVIDER=fastembed, prepare its cache separately, or
    explicitly permit the first download with ANNUAL_REPORT_ALLOW_MODEL_DOWNLOAD=1.
    A failed initialization is memoized for this store's lifetime.
    """

    def __init__(self, embedder: Any = None) -> None:
        self._model = embedder
        self._injected = embedder is not None
        self._attempted = self._injected
        self._error: str | None = None
        self._lock = threading.Lock()
        self.model_name = os.getenv(
            "ANNUAL_REPORT_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"
        )
        self.provider = (
            "injected"
            if self._injected
            else os.getenv("ANNUAL_REPORT_EMBEDDING_PROVIDER", "none").lower()
        )
        if self._injected:
            self.cache_key = str(
                getattr(
                    embedder,
                    "cache_key",
                    getattr(embedder, "model_name", type(embedder).__qualname__),
                )
            )
        else:
            self.cache_key = f"fastembed:{self.model_name}:window400-overlap40-v1"

    @property
    def status(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model_name if not self._injected else self.cache_key,
            "loaded": self._model is not None,
            "error": self._error,
        }

    def _load(self) -> Any:
        with self._lock:
            if self._model is not None:
                return self._model
            if self._attempted:
                raise SemanticUnavailable(
                    self._error or "语义模型不可用，已降级为 BM25 检索。"
                )
            self._attempted = True
            if self.provider != "fastembed":
                self._error = "未启用语义模型，已降级为 BM25 检索。"
                raise SemanticUnavailable(self._error)
            try:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(
                    model_name=self.model_name,
                    cache_dir=os.getenv("ANNUAL_REPORT_MODEL_CACHE")
                    or os.getenv("ANNUAL_REPORT_EMBEDDING_CACHE_DIR")
                    or None,
                    local_files_only=os.getenv(
                        "ANNUAL_REPORT_ALLOW_MODEL_DOWNLOAD", "0"
                    )
                    != "1",
                    threads=max(
                        1,
                        min(8, int(os.getenv("ANNUAL_REPORT_EMBEDDING_THREADS", "2"))),
                    ),
                )
            except Exception as exc:
                self._error = (
                    f"语义模型未就绪（{type(exc).__name__}），已降级为 BM25 检索。"
                )
                raise SemanticUnavailable(self._error) from exc
            return self._model

    def encode(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        model = self._load()
        try:
            # BGE-small-zh accepts 512 tokens. Preserve long Chinese evidence by
            # embedding every bounded window, rather than silently truncating
            # the tail of a 900-character source chunk. PDF layout spaces carry
            # no semantic information and are collapsed only for the model.
            windows: list[str] = []
            groups: list[list[int]] = []
            for text in texts:
                clean = " ".join(text.split())
                group: list[int] = []
                for start in range(0, max(1, len(clean)), 360):
                    group.append(len(windows))
                    windows.append(clean[start : start + 400])
                    if start + 400 >= len(clean):
                        break
                groups.append(group)
            if query and hasattr(model, "query_embed"):
                raw: Iterable[Any] = model.query_embed(windows)
            elif hasattr(model, "embed"):
                raw = model.embed(windows)
            elif hasattr(model, "encode"):
                raw = model.encode(windows)
            elif callable(model):
                raw = model(windows)
            else:
                raise TypeError("embedder must implement embed or encode")
            vectors = [[float(value) for value in vector] for vector in raw]
            if len(vectors) != len(windows) or not vectors:
                raise ValueError("embedding count mismatch")
            dimension = len(vectors[0])
            for vector in vectors:
                if (
                    not dimension
                    or len(vector) != dimension
                    or not all(math.isfinite(value) for value in vector)
                ):
                    raise ValueError("invalid embedding vector")
                norm = math.sqrt(sum(value * value for value in vector))
                if norm <= 0 or not math.isfinite(norm):
                    raise ValueError("invalid embedding norm")
                vector[:] = [value / norm for value in vector]
            pooled: list[list[float]] = []
            for group in groups:
                combined = [
                    sum(vectors[index][axis] * len(windows[index]) for index in group)
                    for axis in range(dimension)
                ]
                norm = math.sqrt(sum(value * value for value in combined))
                if norm <= 0:
                    raise ValueError("invalid pooled embedding norm")
                pooled.append([value / norm for value in combined])
            self._error = None
            return pooled
        except SemanticUnavailable:
            raise
        except Exception as exc:
            self._error = f"语义计算失败（{type(exc).__name__}），已降级为 BM25 检索。"
            raise SemanticUnavailable(self._error) from exc


def semantic_rank(
    query_vector: list[float], vectors: dict[str, list[float]]
) -> list[tuple[str, float]]:
    ranked = []
    for chunk_id, vector in vectors.items():
        if len(vector) != len(query_vector) or not all(
            math.isfinite(value) for value in vector
        ):
            raise SemanticUnavailable("语义向量维度不匹配，已降级为 BM25 检索。")
        score = sum(left * right for left, right in zip(query_vector, vector))
        if not math.isfinite(score):
            raise SemanticUnavailable("语义向量无效，已降级为 BM25 检索。")
        ranked.append((chunk_id, score))
    return sorted(ranked, key=lambda item: (-item[1], item[0]))
