import numpy as np
import pytest

from prodsearch.dense.hnsw import HNSWIndex, exact_search


def clustered(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(40, dim))
    x = centers[rng.integers(0, 40, n)] + 0.6 * rng.normal(size=(n, dim))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


@pytest.fixture(scope="module")
def data():
    vectors = clustered(3000, 32, seed=1)
    queries = clustered(200, 32, seed=2)
    return vectors, queries, HNSWIndex.build(vectors, M=12, ef_construction=100, seed=3)


def recall(index, vectors, queries, k=10, ef=64, mask=None) -> float:
    total = 0.0
    for q in queries:
        got, _ = index.search(q, k, ef, mask)
        truth, _ = exact_search(vectors, q, k, mask)
        total += len(set(got.tolist()) & set(truth.tolist())) / len(truth)
    return total / len(queries)


def test_recall_at_10_against_brute_force(data) -> None:
    vectors, queries, index = data
    assert recall(index, vectors, queries, ef=64) >= 0.95


def test_recall_grows_with_ef(data) -> None:
    vectors, queries, index = data
    assert recall(index, vectors, queries, ef=10) <= recall(index, vectors, queries, ef=128)


def test_graph_invariants(data) -> None:
    vectors, _, index = data
    n = vectors.shape[0]
    assert (index.cnt0 >= 1).all()
    assert (index.cnt0 <= 2 * index.M).all()
    for node in range(0, n, 97):
        links = index.nb0[node, : index.cnt0[node]]
        assert node not in links
        assert len(set(links.tolist())) == links.size
        assert ((links >= 0) & (links < n)).all()
    assert index.levels[index.entry] == index.top == index.levels.max()


def test_results_sorted_by_similarity(data) -> None:
    vectors, queries, index = data
    ids, sims = index.search(queries[0], 20, 64)
    assert np.all(np.diff(sims) <= 1e-6)
    assert np.allclose(sims, vectors[ids] @ queries[0], atol=1e-5)


def test_filtered_search_only_returns_allowed_ids(data) -> None:
    vectors, queries, index = data
    mask = (np.random.default_rng(4).random(vectors.shape[0]) < 0.25).astype(np.uint8)
    for q in queries[:30]:
        ids, _ = index.search(q, 10, 64, mask)
        assert ids.size == 10
        assert mask[ids].all()
    assert recall(index, vectors, queries, ef=64, mask=mask) >= 0.9


def test_save_load_roundtrip(data, tmp_path) -> None:
    vectors, queries, index = data
    index.save(tmp_path / "g.npz")
    loaded = HNSWIndex.load(tmp_path / "g.npz", vectors)
    for q in queries[:20]:
        assert np.array_equal(index.search(q, 10, 32)[0], loaded.search(q, 10, 32)[0])


def test_exact_search_breaks_ties_by_id() -> None:
    vectors = np.array([[1, 0], [0, 1], [1, 0], [0.6, 0.8]], dtype=np.float32)
    ids, sims = exact_search(vectors, np.array([1, 0], dtype=np.float32), 3)
    assert ids.tolist() == [0, 2, 3]
    assert sims.tolist() == pytest.approx([1.0, 1.0, 0.6])


def test_tiny_and_empty_graphs() -> None:
    one = HNSWIndex.build(np.array([[1.0, 0.0]], dtype=np.float32))
    assert one.search(np.array([0.0, 1.0], dtype=np.float32), 5)[0].tolist() == [0]
    empty = HNSWIndex.build(np.zeros((0, 4), dtype=np.float32))
    assert empty.search(np.ones(4, dtype=np.float32), 5)[0].size == 0
