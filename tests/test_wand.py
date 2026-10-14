"""WAND and block-max WAND return exactly the exhaustive top-k, scores and tie order included."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from prodsearch.lexical.bm25f import BM25FConfig, FieldedPostings, LexicalIndex

VOCAB = [f"t{i}" for i in range(14)]


@st.composite
def corpora(draw):
    n_docs = draw(st.integers(1, 300))
    # Skewed term choice so some lists are long and others short, as in real text.
    weights = [1.0 / (i + 1) for i in range(len(VOCAB))]
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    p = np.array(weights) / sum(weights)

    def field(max_len: int) -> list[list[str]]:
        return [
            [str(t) for t in rng.choice(VOCAB, size=rng.integers(0, max_len + 1), p=p)]
            for _ in range(n_docs)
        ]

    tokens = {"name": field(4), "body": field(25)}
    config = BM25FConfig(
        weights={
            "name": draw(st.sampled_from([0.0, 1.0, 3.0])),
            "body": draw(st.sampled_from([0.5, 1.0])),
        },
        b={"name": 0.5, "body": draw(st.sampled_from([0.0, 0.75, 1.0]))},
        k1=draw(st.sampled_from([0.9, 1.2, 2.0])),
    )
    block = draw(st.sampled_from([1, 2, 4, 16, 128]))
    return LexicalIndex.build(FieldedPostings.build(tokens), config, block_size=block)


queries = st.lists(st.sampled_from([*VOCAB, "unseen"]), min_size=0, max_size=6)


def brute_force(index: LexicalIndex, tokens, k: int, mask=None):
    """Top-k straight from the uncompressed impacts, without any kernel code."""
    terms, weights = index.query_terms(tokens)
    docs_all = index.layout.decode()
    ptr = index.layout.blk_post[index.layout.term_blk]
    acc = np.zeros(index.n_docs, dtype=np.int64)
    for t, w in zip(terms, weights, strict=True):
        lo, hi = ptr[t], ptr[t + 1]
        np.add.at(acc, docs_all[lo:hi], index.impacts[lo:hi].astype(np.int64) * w)
    cand = np.flatnonzero(acc > 0)
    if mask is not None:
        cand = cand[mask[cand] > 0]
    order = np.lexsort((cand, -acc[cand]))[:k]
    return cand[order], acc[cand[order]]


@settings(max_examples=120, deadline=None)
@given(corpora(), queries, st.integers(1, 40))
def test_pruned_kernels_match_exhaustive(index: LexicalIndex, tokens, k: int) -> None:
    ref_docs, ref_scores = brute_force(index, tokens, k)
    for method in ("exhaustive", "wand", "bmw"):
        docs, scores, _ = index.search(tokens, k, method=method)
        assert np.array_equal(docs, ref_docs), method
        assert np.allclose(scores, ref_scores * index.scale), method


@settings(max_examples=60, deadline=None)
@given(corpora(), queries, st.integers(1, 20), st.integers(0, 2**32 - 1))
def test_filtered_kernels_match_exhaustive(index: LexicalIndex, tokens, k: int, seed: int) -> None:
    mask = (np.random.default_rng(seed).random(index.n_docs) < 0.3).astype(np.uint8)
    ref_docs, _ = brute_force(index, tokens, k, mask)
    for method in ("exhaustive", "wand", "bmw"):
        docs, _, _ = index.search(tokens, k, mask=mask, method=method)
        assert np.array_equal(docs, ref_docs), method
        assert all(mask[docs])


@settings(max_examples=60, deadline=None)
@given(corpora(), queries, st.integers(0, 2**32 - 1))
def test_score_docs_agrees_with_search(index: LexicalIndex, tokens, seed: int) -> None:
    docs, scores, _ = index.search(tokens, index.n_docs, method="exhaustive")
    assert np.allclose(index.score(tokens, docs), scores)
    others = np.random.default_rng(seed).integers(0, index.n_docs, size=10)
    full = np.zeros(index.n_docs)
    full[docs] = scores
    assert np.allclose(index.score(tokens, others), full[others])


def test_block_max_prunes_on_a_skewed_corpus() -> None:
    rng = np.random.default_rng(3)
    n = 20_000
    common = [["common"] * int(rng.integers(1, 4)) for _ in range(n)]
    for d in rng.choice(n, size=40, replace=False):
        common[d] = [*common[d], "rare"]
    index = LexicalIndex.build(
        FieldedPostings.build({"body": common}),
        BM25FConfig(weights={"body": 1.0}, b={"body": 0.75}),
        block_size=64,
    )
    _, _, exhaustive = index.search(["common", "rare"], 10, method="exhaustive")
    _, _, wand = index.search(["common", "rare"], 10, method="wand")
    _, _, bmw = index.search(["common", "rare"], 10, method="bmw")
    assert bmw["scored"] <= wand["scored"] < exhaustive["scored"]
    assert bmw["blocks_decoded"] < exhaustive["blocks_decoded"]
