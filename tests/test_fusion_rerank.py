import numpy as np
import pytest
from fakes import OverlapScorer

from prodsearch.fusion import convex, rrf, union, zscore
from prodsearch.rerank import RerankBudget, rerank


def test_rrf_rewards_agreement() -> None:
    docs, scores = rrf([np.array([1, 2, 3]), np.array([3, 1, 4])], k=60)
    assert docs.tolist() == [1, 3, 2, 4]
    assert scores[0] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[1] == pytest.approx(1 / 63 + 1 / 61)


def test_rrf_ties_go_to_lower_id() -> None:
    docs, _ = rrf([np.array([7, 2]), np.array([2, 7])])
    assert docs.tolist() == [2, 7]


def test_rrf_with_one_empty_list() -> None:
    docs, _ = rrf([np.array([], dtype=np.int32), np.array([5, 4])])
    assert docs.tolist() == [5, 4]
    assert rrf([])[0].size == 0


def test_zscore_handles_constant_input() -> None:
    assert zscore(np.array([3.0, 3.0])).tolist() == [0.0, 0.0]
    z = zscore(np.array([1.0, 2.0, 3.0]))
    assert z.mean() == pytest.approx(0.0)
    assert z.std() == pytest.approx(1.0)


def test_convex_endpoints_reproduce_each_retriever() -> None:
    docs = np.array([10, 11, 12, 13])
    lexical = np.array([4.0, 0.0, 2.0, 1.0])
    dense = np.array([0.1, 0.9, 0.5, 0.7])
    assert convex(docs, lexical, dense, 1.0)[0].tolist() == [10, 12, 13, 11]
    assert convex(docs, lexical, dense, 0.0)[0].tolist() == [11, 13, 12, 10]
    mixed = convex(docs, lexical, dense, 0.5)[0].tolist()
    assert sorted(mixed) == docs.tolist()


def test_convex_rejects_bad_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        convex(np.array([1]), np.array([1.0]), np.array([1.0]), 1.5)


def test_union_is_sorted_and_unique() -> None:
    assert union(np.array([5, 1]), np.array([1, 3])).tolist() == [1, 3, 5]


PASSAGES = {
    0: "red velvet sofa",
    1: "oak dining table",
    2: "velvet sofa bed",
    3: "sofa",
    4: "lamp",
    5: "red sofa",
}


def test_rerank_reorders_head_and_keeps_tail() -> None:
    docs = np.array([4, 3, 0, 1, 2, 5])
    out = rerank(OverlapScorer(), "red velvet sofa", docs, PASSAGES.__getitem__, RerankBudget(4))
    # Head [4, 3, 0, 1] scores [0, 1, 3, 0]; stable order keeps 4 before 1. Tail stays.
    assert out.docs.tolist() == [0, 3, 4, 1, 2, 5]
    assert out.reranked == 4
    assert not out.truncated
    assert out.scores.tolist() == [3, 1, 0, 0]


def test_rerank_stops_at_the_deadline() -> None:
    ticks = iter(range(0, 1000, 30))  # every clock read advances 30 ms
    budget = RerankBudget(depth=6, deadline_ms=50.0, batch_size=2)
    scorer = OverlapScorer()
    docs = np.array([4, 1, 0, 2, 3, 5])
    out = rerank(
        scorer, "red velvet sofa", docs, PASSAGES.__getitem__, budget, lambda: next(ticks) / 1000
    )
    assert out.truncated
    assert out.reranked == 4
    assert scorer.calls == 2
    assert out.docs.tolist() == [0, 2, 4, 1, 3, 5]


def test_rerank_first_batch_always_runs() -> None:
    budget = RerankBudget(depth=4, deadline_ms=0.0, batch_size=2)
    out = rerank(OverlapScorer(), "sofa", np.array([4, 3, 1]), PASSAGES.__getitem__, budget)
    assert out.reranked == 2
    assert out.docs.tolist() == [3, 4, 1]
