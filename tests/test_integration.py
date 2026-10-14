"""Runs against the real WANDS files and ONNX models; skipped unless `prodsearch fetch` ran."""

import numpy as np
import pytest

from prodsearch import paths, sources

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (sources.wands_present() and all(sources.model_present(m) for m in sources.MODELS)),
        reason="WANDS or models not fetched",
    ),
]


@pytest.fixture(scope="module")
def wands():
    from prodsearch.catalog import load_wands

    return load_wands(paths.wands_dir())


def test_wands_shape(wands) -> None:
    assert len(wands.catalog) == 42_994
    assert len(wands.queries) == 480
    assert wands.judgements.rows == 233_448


def test_bm25f_on_full_catalogue_matches_exhaustive(wands) -> None:
    from prodsearch.analysis import analyze
    from prodsearch.store import build_lexical

    index = build_lexical(wands.catalog)
    for q in wands.queries[:50]:
        tokens = analyze(q.text)
        ref = index.search(tokens, 100, method="exhaustive")[0]
        assert np.array_equal(index.search(tokens, 100, method="bmw")[0], ref)


def test_onnx_models_produce_sane_outputs(wands) -> None:
    from prodsearch.dense.encoders import OnnxBiEncoder, OnnxCrossEncoder

    encoder = OnnxBiEncoder()
    docs = [wands.catalog.passage(d) for d in range(8)]
    vectors = encoder.encode_passages(docs)
    assert vectors.shape == (8, 384)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
    q = encoder.encode_query(wands.catalog.names[3])
    assert int(np.argmax(vectors @ q)) == 3
    scores = OnnxCrossEncoder().score(wands.catalog.names[0], [docs[0], docs[5]])
    assert scores[0] > scores[1]
