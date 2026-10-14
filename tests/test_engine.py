from prodsearch.engine import CascadeConfig, SearchEngine, SearchRequest
from prodsearch.filters import Filter
from prodsearch.rerank import RerankBudget
from prodsearch.store import sample_engine


def test_lexical_only_sample_engine() -> None:
    engine = sample_engine()
    assert engine.capabilities() == ["lexical"]
    response = engine.search(SearchRequest("bar stools", k=3))
    assert response.stages == ["lexical"]
    assert {h.name for h in response.hits[:2]} == {
        "counter height bar stools set of 2",
        "industrial metal bar stool",
    }
    assert {"analyze", "filter", "lexical", "total"} <= set(response.timings_ms)
    assert response.lexical_stats["scored"] >= 2


def test_hybrid_pipeline_runs_every_stage(hybrid_engine: SearchEngine) -> None:
    response = hybrid_engine.search(SearchRequest("turquoise velvet pillow", k=5))
    assert response.stages == ["lexical", "dense", "fusion:rrf", "rerank"]
    assert {"embed", "ann", "fusion", "rerank", "total"} <= set(response.timings_ms)
    top = response.hits[0]
    assert top.name == "turquoise velvet throw pillow"
    assert top.lexical_rank == 1
    assert top.dense_rank is not None
    assert top.rerank_score is not None


def test_convex_fusion_and_modes(hybrid_engine: SearchEngine) -> None:
    for mode, stages in (
        ("lexical", ["lexical"]),
        ("dense", ["dense"]),
        ("hybrid", ["lexical", "dense", "fusion:convex"]),
    ):
        response = hybrid_engine.search(
            SearchRequest("propane fire pit", mode=mode, fusion="convex", alpha=0.6, rerank=False)
        )
        assert response.stages == stages
        assert response.hits[0].product_class == "Outdoor Fireplaces"


def test_filters_restrict_every_stage(hybrid_engine: SearchEngine) -> None:
    flt = Filter(classes=("Outdoor Fireplaces",))
    response = hybrid_engine.search(SearchRequest("table", k=10, filters=flt))
    assert response.filter_matches == 2
    assert response.hits
    assert {h.product_class for h in response.hits} == {"Outdoor Fireplaces"}


def test_empty_filter_short_circuits(hybrid_engine: SearchEngine) -> None:
    response = hybrid_engine.search(SearchRequest("sofa", filters=Filter(classes=("nope",))))
    assert response.hits == []
    assert response.filter_matches == 0
    assert response.stages == []


def test_rerank_depth_budget(hybrid_engine: SearchEngine) -> None:
    engine = SearchEngine(
        hybrid_engine.catalog,
        hybrid_engine.lexical,
        hybrid_engine.encoder,
        hybrid_engine.vectors,
        hybrid_engine.hnsw,
        hybrid_engine.cross_encoder,
        CascadeConfig(rerank=RerankBudget(depth=3)),
    )
    response = engine.search(SearchRequest("chair", k=10))
    assert response.reranked == 3
    assert sum(h.rerank_score is not None for h in response.hits) == 3


def test_query_without_known_terms(hybrid_engine: SearchEngine) -> None:
    response = hybrid_engine.search(SearchRequest("zzzz qqqq", mode="lexical"))
    assert response.hits == []
    assert hybrid_engine.search(SearchRequest("zzzz qqqq")).hits  # dense still answers
