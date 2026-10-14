"""Graded relevance metrics and bootstrap intervals.

Gains are the WANDS grades with linear gain (Exact 2, Partial 1, Irrelevant 0) and a log2
discount, as trec_eval computes nDCG. Unjudged products count as gain 0. Recall@100 and MRR@10
treat only Exact as relevant. A query with no positive grade has no defined nDCG and is skipped
(NaN); a query without an Exact product is skipped for Recall and MRR.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

EXACT = 2


def ndcg_at(ranking: Sequence[int], grades: Mapping[int, int], k: int = 10) -> float:
    ideal = sorted((g for g in grades.values() if g > 0), reverse=True)[:k]
    if not ideal:
        return float("nan")
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = sum(grades.get(int(d), 0) * discounts[i] for i, d in enumerate(ranking[:k]))
    idcg = float(np.dot(ideal, discounts[: len(ideal)]))
    return float(dcg / idcg)


def recall_at(ranking: Sequence[int], grades: Mapping[int, int], k: int = 100) -> float:
    relevant = {d for d, g in grades.items() if g >= EXACT}
    if not relevant:
        return float("nan")
    return len(relevant.intersection(int(d) for d in ranking[:k])) / len(relevant)


def mrr_at(ranking: Sequence[int], grades: Mapping[int, int], k: int = 10) -> float:
    if not any(g >= EXACT for g in grades.values()):
        return float("nan")
    for i, d in enumerate(ranking[:k]):
        if grades.get(int(d), 0) >= EXACT:
            return 1.0 / (i + 1)
    return 0.0


METRICS = {
    "ndcg@10": lambda r, g: ndcg_at(r, g, 10),
    "recall@100": lambda r, g: recall_at(r, g, 100),
    "mrr@10": lambda r, g: mrr_at(r, g, 10),
}


def per_query(rankings: Sequence[Sequence[int]], grades: Sequence[Mapping[int, int]]):
    """{metric: array over queries} with NaN where the metric is undefined."""
    return {
        name: np.array([fn(r, g) for r, g in zip(rankings, grades, strict=True)])
        for name, fn in METRICS.items()
    }


def bootstrap_mean(
    values: np.ndarray, n_boot: int = 10_000, seed: int = 0
) -> dict[str, float | list[float] | int]:
    v = values[~np.isnan(values)]
    if v.size == 0:
        return {"mean": float("nan"), "ci95": [float("nan"), float("nan")], "n": 0}
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(v.mean()), "ci95": [float(lo), float(hi)], "n": int(v.size)}


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0
) -> dict[str, float | list[float] | int]:
    """Mean of a - b over queries defined for both, its 95% interval and P(boot mean <= 0)."""
    keep = ~(np.isnan(a) | np.isnan(b))
    diff = a[keep] - b[keep]
    if diff.size == 0:
        return {"mean": float("nan"), "ci95": [float("nan"), float("nan")], "n": 0}
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, diff.size, size=(n_boot, diff.size))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "mean": float(diff.mean()),
        "ci95": [float(lo), float(hi)],
        "p_le_zero": float((means <= 0).mean()),
        "n": int(diff.size),
    }


def latency_summary(ms: Sequence[float]) -> dict[str, float | int]:
    a = np.asarray(ms, dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    p50, p95, p99 = np.percentile(a, [50, 95, 99])
    mean = float(a.mean())
    return {
        "n": int(a.size),
        "mean_ms": mean,
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "p99_ms": float(p99),
        "qps_sequential": 1000.0 / mean if mean > 0 else float("inf"),
    }
