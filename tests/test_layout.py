import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from prodsearch.lexical.layout import PostingLayout


@st.composite
def posting_lists(draw):
    n_docs = draw(st.integers(1, 5000))
    n_terms = draw(st.integers(1, 12))
    lists = []
    for _ in range(n_terms):
        size = draw(st.integers(1, min(n_docs, 400)))
        docs = draw(
            st.lists(st.integers(0, n_docs - 1), min_size=size, max_size=size, unique=True)
        )
        lists.append(sorted(docs))
    return n_docs, lists


def _flatten(lists):
    ptr = np.zeros(len(lists) + 1, dtype=np.int64)
    np.cumsum([len(x) for x in lists], out=ptr[1:])
    return np.concatenate([np.array(x, dtype=np.int32) for x in lists]), ptr


@settings(max_examples=150, deadline=None)
@given(posting_lists(), st.sampled_from([1, 3, 8, 128]))
def test_pack_unpack_roundtrip(data, block_size: int) -> None:
    n_docs, lists = data
    docs, ptr = _flatten(lists)
    layout = PostingLayout.build(docs, ptr, n_docs, block_size)
    assert np.array_equal(layout.decode(), docs)
    assert layout.n_postings == docs.shape[0]
    # Every block's skip entry is its last doc id.
    assert np.array_equal(layout.blk_last, docs[layout.blk_post[1:] - 1])


def test_consecutive_ids_cost_no_payload() -> None:
    docs, ptr = _flatten([list(range(1000))])
    layout = PostingLayout.build(docs, ptr, 1000, 128)
    assert layout.words.shape[0] == 0
    assert set(layout.blk_bits.tolist()) == {0}
    assert np.array_equal(layout.decode(), docs)


def test_wide_gaps_round_trip() -> None:
    docs, ptr = _flatten([[0, 2**30, 2**31 - 2], [5]])
    layout = PostingLayout.build(docs, ptr, 2**31 - 1, 2)
    assert np.array_equal(layout.decode(), docs)


def test_compression_beats_int32_on_dense_lists() -> None:
    rng = np.random.default_rng(0)
    lists = [np.sort(rng.choice(20_000, size=4000, replace=False)).tolist() for _ in range(5)]
    docs, ptr = _flatten(lists)
    layout = PostingLayout.build(docs, ptr, 20_000, 128)
    assert layout.words.nbytes < 0.5 * docs.nbytes
