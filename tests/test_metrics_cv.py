import math

import numpy as np
import pytest

from prodsearch.bench.cv import fold_ids, select
from prodsearch.bench.metrics import (
    bootstrap_mean,
    latency_summary,
    mrr_at,
    ndcg_at,
    paired_bootstrap,
    recall_at,
)

GRADES = {1: 2, 2: 1, 3: 0, 4: 2}


def test_ndcg_perfect_and_reversed() -> None:
    assert ndcg_at([1, 4, 2, 3], GRADES) == pytest.approx(1.0)
    dcg = 1 / math.log2(3) + 2 / math.log2(4) + 2 / math.log2(5)
    idcg = 2 + 2 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at([3, 2, 1, 4], GRADES) == pytest.approx(dcg / idcg)


def test_ndcg_counts_unjudged_as_zero_and_cuts_at_k() -> None:
    assert ndcg_at([99, 98], GRADES) == 0.0
    assert ndcg_at([99, 1], GRADES, k=1) == 0.0


def test_undefined_metrics_are_nan() -> None:
    assert math.isnan(ndcg_at([1], {1: 0}))
    assert math.isnan(recall_at([1], {1: 1}))
    assert math.isnan(mrr_at([1], {1: 1}))


def test_recall_and_mrr_use_exact_only() -> None:
    assert recall_at([2, 1], GRADES, k=100) == 0.5
    assert mrr_at([3, 2, 4], GRADES) == pytest.approx(1 / 3)
    assert mrr_at([2, 3], GRADES) == 0.0


def test_bootstrap_interval_brackets_the_mean() -> None:
    values = np.random.default_rng(0).random(200)
    out = bootstrap_mean(np.append(values, np.nan), n_boot=2000)
    assert out["n"] == 200
    assert out["ci95"][0] < out["mean"] < out["ci95"][1]


def test_paired_bootstrap_detects_a_shift() -> None:
    rng = np.random.default_rng(1)
    b = rng.random(300)
    out = paired_bootstrap(b + 0.1, b, n_boot=2000)
    assert out["mean"] == pytest.approx(0.1)
    assert out["ci95"][0] > 0
    assert out["p_le_zero"] == 0.0


def test_latency_summary() -> None:
    out = latency_summary([1.0, 2.0, 3.0, 4.0])
    assert out["p50_ms"] == pytest.approx(2.5)
    assert out["qps_sequential"] == pytest.approx(400.0)


def test_folds_partition_the_queries() -> None:
    folds = fold_ids(23, 5, seed=3)
    assert sorted(np.bincount(folds).tolist()) == [4, 4, 5, 5, 5]
    assert np.array_equal(folds, fold_ids(23, 5, seed=3))


def test_select_never_peeks_at_the_held_out_fold() -> None:
    folds = np.array([0, 0, 1, 1])
    # Candidate 1 is best on fold 0's queries only, candidate 0 on fold 1's.
    matrix = np.array([[0.1, 0.1, 0.9, 0.9], [0.8, 0.8, 0.2, 0.2]])
    sel = select(matrix, folds)
    assert sel.chosen.tolist() == [0, 1]
    assert sel.per_query.tolist() == [0.1, 0.1, 0.2, 0.2]


def test_select_ignores_nan_and_prefers_earlier_candidates_on_ties() -> None:
    folds = np.array([0, 1, 1])
    matrix = np.array([[0.5, np.nan, 0.5], [0.5, 0.5, 0.5]])
    assert select(matrix, folds).chosen.tolist() == [0, 0]
