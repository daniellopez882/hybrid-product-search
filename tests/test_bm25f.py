import math

import numpy as np
import pytest

from prodsearch.lexical.bm25f import (
    QUANT_LEVELS,
    BM25FConfig,
    FieldedPostings,
    LexicalIndex,
)

TOKENS = {
    "name": [["oak", "bed"], ["sofa"], ["oak", "table"], []],
    "body": [["solid", "oak", "oak"], ["grey", "velvet", "sofa", "sofa"], ["table"], ["oak"]],
}
CONFIG = BM25FConfig(weights={"name": 2.0, "body": 1.0}, b={"name": 0.5, "body": 0.75}, k1=1.2)


def reference_bm25f(term: str, doc: int, config: BM25FConfig = CONFIG) -> float:
    n = len(TOKENS["name"])
    pseudo = 0.0
    for field, docs in TOKENS.items():
        lengths = [len(d) for d in docs]
        avg = sum(lengths) / n
        b = config.b[field]
        tf = docs[doc].count(term)
        if config.weights[field] and avg:
            pseudo += config.weights[field] * tf / (1 - b + b * lengths[doc] / avg)
    df = sum(
        1 for d in range(n) if any(term in TOKENS[f][d] for f in TOKENS if config.weights[f] > 0)
    )
    if pseudo == 0:
        return 0.0
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    return idf * pseudo / (config.k1 + pseudo)


@pytest.fixture(scope="module")
def index() -> LexicalIndex:
    return LexicalIndex.build(FieldedPostings.build(TOKENS), CONFIG, block_size=2)


def test_scores_match_the_bm25f_formula(index: LexicalIndex) -> None:
    for term in ("oak", "sofa", "table", "bed", "velvet"):
        docs, scores, _ = index.search([term], 10, method="exhaustive")
        for d, s in zip(docs, scores, strict=True):
            expected = reference_bm25f(term, int(d))
            # One quantisation step of the largest impact is the worst-case error.
            assert s == pytest.approx(expected, abs=index.scale)


def test_multi_term_score_is_the_sum(index: LexicalIndex) -> None:
    docs, scores, _ = index.search(["oak", "table"], 10)
    by_doc = dict(zip(docs.tolist(), scores, strict=True))
    for d, s in by_doc.items():
        expected = reference_bm25f("oak", d) + reference_bm25f("table", d)
        assert s == pytest.approx(expected, abs=2 * index.scale)


def test_repeated_query_term_counts_twice(index: LexicalIndex) -> None:
    once = dict(zip(*index.search(["sofa"], 10)[:2], strict=True))
    twice = dict(zip(*index.search(["sofa", "sofa"], 10)[:2], strict=True))
    assert twice == pytest.approx({d: 2 * s for d, s in once.items()})


def test_zero_weight_field_does_not_match() -> None:
    config = BM25FConfig(weights={"name": 0.0, "body": 1.0}, b={"name": 0.5, "body": 0.75})
    index = LexicalIndex.build(FieldedPostings.build(TOKENS), config, block_size=2)
    docs, _, _ = index.search(["bed"], 10)  # "bed" only occurs in a name
    assert docs.size == 0


def test_impacts_are_quantised_to_16_bits(index: LexicalIndex) -> None:
    assert index.impacts.dtype == np.uint16
    assert int(index.impacts.max()) == QUANT_LEVELS
    assert int(index.impacts[index.impacts > 0].min()) >= 1


def test_unknown_terms_and_empty_queries(index: LexicalIndex) -> None:
    for tokens in ([], ["nothing"], ["nothing", "here"]):
        docs, scores, _ = index.search(tokens, 5)
        assert docs.size == 0
        assert scores.size == 0


def test_save_load_roundtrip(index: LexicalIndex, tmp_path) -> None:
    path = tmp_path / "lexical.npz"
    index.save(path)
    loaded = LexicalIndex.load(path)
    assert loaded.vocab == index.vocab
    assert loaded.config == index.config
    for q in (["oak"], ["sofa", "grey"], ["table", "oak", "bed"]):
        a = index.search(q, 3)
        b = loaded.search(q, 3)
        assert np.array_equal(a[0], b[0])
        assert np.allclose(a[1], b[1])
