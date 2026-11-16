# -*- coding: utf-8 -*-
"""Cross-Encoder reranking service.

Uses ``BAAI/bge-reranker-v2-m3`` for production reranking.
Falls back gracefully when the model is unavailable (CI / lightweight envs).

Environment variables
---------------------
RAG_RERANKER
    ``bge-reranker`` (default), ``bge-reranker-onnx``, ``bge-reranker-http``, or ``none``.
    ONNX uses fastembed and a separately prepared BAAI/bge-reranker-base cache;
    it never downloads a model automatically.
    HTTP calls the local CUDA service at RAG_RERANKER_URL (default port 8011).
RAG_RERANKER_MAX_LENGTH
    Maximum token length for reranker input. Default ``512``.
"""
from __future__ import annotations

import logging
import os
import threading
import requests
from math import isfinite
from typing import Any, Sequence

logger = logging.getLogger(__name__)

_DEFAULT_TOP_N = 8


def _is_production_env() -> bool:
    values = [
        os.getenv("APP_ENV", ""),
        os.getenv("ENV", ""),
        os.getenv("NODE_ENV", ""),
        os.getenv("FASTAPI_ENV", ""),
    ]
    return any(str(v).strip().lower() in {"prod", "production"} for v in values)


def _log_reranker_degraded(*, reason: str, detail: str | None = None, requested_backend: str | None = None) -> None:
    payload: dict[str, Any] = {
        "event": "rag_reranker_degraded",
        "reason": reason,
        "requested_backend": requested_backend or os.getenv("RAG_RERANKER", "bge-reranker"),
    }
    if detail:
        payload["detail"] = detail
    log_fn = logger.warning if _is_production_env() else logger.info
    log_fn("rag_reranker_degraded %s", payload)


class _CrossEncoderWrapper:
    """Lazy-loaded bge-reranker-v2-m3 wrapper (thread-safe singleton)."""

    def __init__(self) -> None:
        self._model: Any = None
        self._lock = threading.Lock()
        self._max_length = int(os.getenv("RAG_RERANKER_MAX_LENGTH", "512"))

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]

            logger.info("Loading bge-reranker-v2-m3 (max_length=%d) ...", self._max_length)
            self._model = CrossEncoder(
                "BAAI/bge-reranker-v2-m3",
                max_length=self._max_length,
            )
            logger.info("bge-reranker-v2-m3 loaded successfully.")
            return self._model

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        model = self._load()
        scores = model.predict(pairs)
        return [float(score) for score in scores]


_reranker: _CrossEncoderWrapper | None = None
_reranker_lock = threading.Lock()


def _get_reranker() -> _CrossEncoderWrapper:
    global _reranker
    if _reranker is not None:
        return _reranker
    with _reranker_lock:
        if _reranker is None:
            _reranker = _CrossEncoderWrapper()
    return _reranker


class RerankerService:
    """Cross-Encoder reranking service with graceful fallback."""

    def __init__(self, *, force_backend: str | None = None) -> None:
        backend = (force_backend or os.getenv("RAG_RERANKER", "bge-reranker")).strip().lower()
        self._requested_backend = backend
        self._enabled = backend != "none"
        self._available: bool | None = None
        self._onnx_model: Any = None
        self._onnx_lock = threading.Lock()
        if not self._enabled:
            _log_reranker_degraded(
                reason="configured_none_backend",
                requested_backend=backend,
            )

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self._check_available()

    def _check_available(self) -> bool:
        """Lazy check whether sentence_transformers CrossEncoder is importable."""
        if self._available is not None:
            return self._available
        if not self._enabled:
            self._available = False
            return False
        try:
            if self._requested_backend == "bge-reranker-http":
                self._available = True
                return True
            if self._requested_backend == "bge-reranker-onnx":
                from fastembed.rerank.cross_encoder import TextCrossEncoder  # noqa: F401
            else:
                from sentence_transformers import CrossEncoder  # noqa: F401

            self._available = True
        except ImportError as exc:
            _log_reranker_degraded(
                reason="reranker_dependency_unavailable",
                detail=str(exc),
                requested_backend=self._requested_backend,
            )
            self._available = False
        return self._available

    def _predict_onnx(self, query: str, documents: list[str]) -> list[float]:
        with self._onnx_lock:
            if self._onnx_model is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
                self._onnx_model = TextCrossEncoder(
                    model_name="BAAI/bge-reranker-base",
                    cache_dir=os.getenv("RAG_RERANKER_CACHE_DIR"),
                    local_files_only=True,
                    threads=2,
                )
        # Group similarly sized passages to reduce padding and peak CPU memory.
        # Scores must be restored to input order before attaching source metadata.
        order = sorted(range(len(documents)), key=lambda i: len(documents[i]))
        batch_size = max(1, min(32, int(os.getenv("RAG_RERANKER_BATCH_SIZE", "4"))))
        values = list(self._onnx_model.rerank(query, [documents[i] for i in order], batch_size=batch_size))
        if len(values) != len(order):
            raise ValueError("ONNX reranker returned an incomplete batch")
        scores = [0.0] * len(order)
        for index, value in zip(order, values):
            scores[index] = float(value)
        return scores

    def _predict_remote(self, query: str, documents: list[str]) -> list[float]:
        url = os.getenv("RAG_RERANKER_URL", "http://127.0.0.1:8011").rstrip("/")
        response = requests.post(url + "/rerank", json={"query":query,"documents":documents}, timeout=(5,60))
        response.raise_for_status()
        data = response.json()
        if data.get("model") != "bge-reranker-base" or data.get("provider") != "CUDAExecutionProvider":
            raise ValueError("Unexpected remote reranker model/provider")
        return [float(value) for value in data["scores"]]

    def rerank(
        self,
        query: str,
        documents: Sequence[dict[str, Any]],
        *,
        top_n: int = _DEFAULT_TOP_N,
        content_key: str = "content",
    ) -> list[dict[str, Any]]:
        """Rerank *documents* by relevance to *query*."""
        doc_list = list(documents)
        if not doc_list or not query.strip():
            return doc_list[:top_n]

        if not self._enabled or not self._check_available():
            return doc_list[:top_n]

        try:
            pairs = [(query, str(doc.get(content_key) or "")) for doc in doc_list]
            if self._requested_backend == "bge-reranker-http":
                scores = self._predict_remote(query, [pair[1] for pair in pairs])
            elif self._requested_backend == "bge-reranker-onnx":
                scores = self._predict_onnx(query, [pair[1] for pair in pairs])
            else:
                scores = _get_reranker().predict(pairs)
            if len(scores) != len(doc_list) or not all(isfinite(score) for score in scores):
                raise ValueError("Reranker returned missing or non-finite scores")
            ranked = sorted(zip(doc_list, scores), key=lambda item: item[1], reverse=True)
            return [{**doc, "rerank_score": score} for doc, score in ranked[:top_n]]
        except Exception as exc:
            _log_reranker_degraded(
                reason="rerank_inference_error",
                detail=str(exc),
                requested_backend=self._requested_backend,
            )
            return doc_list[:top_n]


_reranker_service: RerankerService | None = None
_reranker_svc_lock = threading.Lock()


def get_reranker_service() -> RerankerService:
    """Return the module-level RerankerService singleton."""
    global _reranker_service
    if _reranker_service is not None:
        return _reranker_service
    with _reranker_svc_lock:
        if _reranker_service is None:
            _reranker_service = RerankerService()
    return _reranker_service


def reset_reranker_service() -> None:
    """Reset singleton (for testing)."""
    global _reranker_service
    with _reranker_svc_lock:
        _reranker_service = None


__all__ = [
    "RerankerService",
    "get_reranker_service",
    "reset_reranker_service",
]
