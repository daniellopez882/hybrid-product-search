from __future__ import annotations

import pytest
from fakes import HashingEncoder, OverlapScorer

from prodsearch.catalog import Catalog, load_catalog
from prodsearch.dense.hnsw import HNSWIndex
from prodsearch.engine import SearchEngine
from prodsearch.store import build_lexical, sample_catalog_path


@pytest.fixture(scope="session")
def sample_catalog() -> Catalog:
    return load_catalog(sample_catalog_path())


@pytest.fixture(scope="session")
def hybrid_engine(sample_catalog: Catalog) -> SearchEngine:
    encoder = HashingEncoder()
    vectors = encoder.encode_passages(
        [sample_catalog.passage(d) for d in range(len(sample_catalog))]
    ).astype("float32")
    graph = HNSWIndex.build(vectors, M=8, ef_construction=32)
    return SearchEngine(
        sample_catalog,
        build_lexical(sample_catalog),
        encoder=encoder,
        vectors=vectors,
        hnsw=graph,
        cross_encoder=OverlapScorer(),
    )
