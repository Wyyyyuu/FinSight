"""Dependency-free BM25 lexical retrieval; no neural reranking or model calls.

The index is immutable and belongs to one collection/snapshot. Rebuild on corpus
changes; never reuse document frequencies across permission/collection boundaries.
"""
from collections import Counter, defaultdict
from math import isfinite, log
from typing import Callable, Sequence

from backend.rag.lexical import lexical_tokens


class BM25Index:
    def __init__(self, texts: Sequence[str], *, tokenizer: Callable = lexical_tokens,
                 k1: float = 1.2, b: float = 0.75):
        if not isfinite(k1) or k1 <= 0 or not isfinite(b) or not 0 <= b <= 1:
            raise ValueError("BM25 requires finite k1 > 0 and 0 <= b <= 1")
        self.tokenizer, self.k1, self.b = tokenizer, k1, b
        self.size = len(texts)
        self.lengths = []
        self.postings = defaultdict(list)
        for index, value in enumerate(texts):
            terms = Counter(tokenizer(value))
            self.lengths.append(sum(terms.values()))
            for token, count in terms.items():
                self.postings[token].append((index, count))
        self.avg_length = sum(self.lengths) / self.size if self.size else 1.0
        self.avg_length = self.avg_length or 1.0

    def scores(self, query: str) -> list[float]:
        result = [0.0] * self.size
        for token in sorted(set(self.tokenizer(query))):
            posting = self.postings.get(token, ())
            if not posting:
                continue
            idf = log(1 + (self.size - len(posting) + 0.5) / (len(posting) + 0.5))
            for index, frequency in posting:
                norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / self.avg_length)
                result[index] += idf * frequency * (self.k1 + 1) / (frequency + norm)
        return result


def reciprocal_rank_fusion(dense: Sequence[float], sparse: Sequence[float], *,
                           rrf_k: int = 60, sparse_weight: float = 1.0) -> list[int]:
    """Stable fusion of vector and lexical candidate ranks; zero sparse is no hit."""
    if len(dense) != len(sparse):
        raise ValueError("Dense and sparse candidate counts differ")
    if rrf_k < 1 or not isfinite(sparse_weight) or sparse_weight < 0:
        raise ValueError("Invalid fusion parameters")
    if not all(isfinite(v) for v in (*dense, *sparse)):
        raise ValueError("Non-finite retrieval scores")
    score = [0.0] * len(dense)
    for rank, i in enumerate(sorted(range(len(dense)), key=lambda i: (-dense[i], i)), 1):
        score[i] += 1 / (rrf_k + rank)
    positive = [i for i in range(len(sparse)) if sparse[i] > 0]
    for rank, i in enumerate(sorted(positive, key=lambda i: (-sparse[i], i)), 1):
        score[i] += sparse_weight / (rrf_k + rank)
    return sorted(range(len(score)), key=lambda i: (-score[i], -dense[i], -sparse[i], i))
