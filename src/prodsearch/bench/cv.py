"""K-fold model selection over a precomputed (candidate x query) metric matrix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def fold_ids(n_queries: int, n_folds: int = 5, seed: int = 13) -> np.ndarray:
    """Fold index per query: a seeded permutation dealt into n_folds near-equal parts."""
    folds = np.empty(n_queries, dtype=np.int64)
    folds[np.random.default_rng(seed).permutation(n_queries)] = np.arange(n_queries) % n_folds
    return folds


@dataclass(frozen=True)
class Selection:
    chosen: np.ndarray  # candidate index per fold
    per_query: np.ndarray  # held-out metric per query, from its fold's chosen candidate


def select(matrix: np.ndarray, folds: np.ndarray) -> Selection:
    """For each fold pick the candidate with the best mean over the other folds' queries.

    NaN entries (metric undefined for a query) are ignored; ties go to the lowest index, so
    candidate order doubles as a preference order.
    """
    n_folds = int(folds.max()) + 1
    chosen = np.empty(n_folds, dtype=np.int64)
    held_out = np.full(matrix.shape[1], np.nan)
    for f in range(n_folds):
        train = folds != f
        means = np.nanmean(matrix[:, train], axis=1)
        chosen[f] = int(np.nanargmax(means))
        held_out[folds == f] = matrix[chosen[f], folds == f]
    return Selection(chosen=chosen, per_query=held_out)
