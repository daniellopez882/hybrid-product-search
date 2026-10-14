"""Rank fusion of the lexical and dense candidate lists.

* RRF (Cormack, Clarke and Buettcher, 2009): sum of 1 / (k + rank) over the lists a document
  appears in. Uses ranks only, so it needs no score calibration and no tuning beyond k.
* Convex combination of z-scored scores: both retrievers score the union of their candidates
  (the lexical index scores dense-only hits exactly and vice versa), each score vector is
  standardised over that union, and alpha * z_lex + (1 - alpha) * z_dense ranks the union.
  alpha is chosen by cross-validation in the benchmark.

Both return (doc ids, fused scores) best first, ties to the lower doc id.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

RRF_K = 60


def _ranked(docs: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((docs, -scores))
    return docs[order].astype(np.int32), scores[order]


def rrf(rankings: Sequence[np.ndarray], k: int = RRF_K) -> tuple[np.ndarray, np.ndarray]:
    lists = [np.asarray(r, dtype=np.int64) for r in rankings if len(r)]
    if not lists:
        return np.zeros(0, np.int32), np.zeros(0, np.float64)
    docs = np.concatenate(lists)
    contrib = np.concatenate([1.0 / (k + np.arange(1, len(r) + 1)) for r in lists])
    union, inverse = np.unique(docs, return_inverse=True)
    fused = np.bincount(inverse, weights=contrib, minlength=union.shape[0])
    return _ranked(union, fused)


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    std = values.std()
    if std == 0.0:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def union(*lists: np.ndarray) -> np.ndarray:
    parts = [np.asarray(x, dtype=np.int64) for x in lists if len(x)]
    if not parts:
        return np.zeros(0, np.int32)
    return np.unique(np.concatenate(parts)).astype(np.int32)


def convex(
    docs: np.ndarray, lexical: np.ndarray, dense: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse full score vectors over a fixed candidate set `docs`."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    fused = alpha * zscore(lexical) + (1.0 - alpha) * zscore(dense)
    return _ranked(np.asarray(docs), fused)
