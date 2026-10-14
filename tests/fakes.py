"""Deterministic stand-ins for the ONNX models, so engine and service tests run offline."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from prodsearch.analysis import analyze


def _bucket(token: str, dim: int) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=4).digest(), "little") % dim


class HashingEncoder:
    """Bag of analysed tokens hashed into `dim` buckets, L2-normalised."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.max_length = 0

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for t in analyze(text):
            v[_bucket(t, self.dim)] += 1.0
        norm = np.linalg.norm(v)
        if norm == 0:
            v[0] = 1.0
            return v
        return v / norm

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim))

    def encode_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class OverlapScorer:
    """Cross-encoder stand-in: number of query tokens present in the passage."""

    def __init__(self) -> None:
        self.calls = 0

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        self.calls += 1
        q = set(analyze(query))
        return np.array([len(q & set(analyze(p))) for p in passages], dtype=np.float32)
