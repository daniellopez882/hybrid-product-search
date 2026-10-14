"""HTTP contract of the Falcon ASGI app."""

import pytest
from falcon import testing

from prodsearch.engine import SearchEngine
from prodsearch.service import create_app
from prodsearch.store import sample_engine


@pytest.fixture(scope="module")
def client() -> testing.TestClient:
    return testing.TestClient(create_app(sample_engine()))


@pytest.fixture(scope="module")
def hybrid_client(hybrid_engine: SearchEngine) -> testing.TestClient:
    return testing.TestClient(create_app(hybrid_engine))


def test_healthz(client: testing.TestClient) -> None:
    result = client.simulate_get("/healthz")
    assert result.status_code == 200
    assert result.json == {"status": "ok", "products": 40, "capabilities": ["lexical"]}


def test_search_get_contract(client: testing.TestClient) -> None:
    result = client.simulate_get("/search", params={"q": "floor mirror", "k": "2"})
    assert result.status_code == 200
    body = result.json
    assert set(body) == {
        "query",
        "filters",
        "stages",
        "timings_ms",
        "reranked",
        "rerank_truncated",
        "filter_matches",
        "lexical_stats",
        "hits",
    }
    assert body["query"] == "floor mirror"
    assert len(body["hits"]) == 2
    assert body["hits"][0]["name"] == "full length floor mirror"
    assert set(body["hits"][0]) == {
        "product_id",
        "name",
        "product_class",
        "fused_score",
        "lexical_rank",
        "dense_rank",
        "rerank_score",
    }
    assert body["timings_ms"]["total"] >= body["timings_ms"]["lexical"]


def test_search_post_with_filters(hybrid_client: testing.TestClient) -> None:
    result = hybrid_client.simulate_post(
        "/search",
        json={
            "q": "rug",
            "k": 5,
            "fusion": "convex",
            "alpha": 0.7,
            "class": ["Area Rugs"],
            "attr": ["color=cream"],
        },
    )
    assert result.status_code == 200
    body = result.json
    assert body["filters"] == {"class": ["Area Rugs"], "attr": ["color=cream"]}
    assert body["filter_matches"] == 1
    assert [h["name"] for h in body["hits"]] == ["shag area rug 8 x 10"]
    assert body["stages"] == ["lexical", "dense", "fusion:convex", "rerank"]
    assert {"embed", "ann", "fusion", "rerank"} <= set(body["timings_ms"])


def test_repeated_query_parameters(hybrid_client: testing.TestClient) -> None:
    result = hybrid_client.simulate_get(
        "/search", query_string="q=stool&class=Bar%20Stools&class=Desks&rerank=false"
    )
    assert result.status_code == 200
    assert "rerank" not in result.json["stages"]
    assert {h["product_class"] for h in result.json["hits"]} <= {"Bar Stools", "Desks"}


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"q": "  "},
        {"q": "sofa", "k": "0"},
        {"q": "sofa", "k": "many"},
        {"q": "sofa", "k": "101"},
        {"q": "sofa", "mode": "fuzzy"},
        {"q": "sofa", "fusion": "max"},
        {"q": "sofa", "alpha": "2"},
        {"q": "sofa", "rerank": "maybe"},
        {"q": "sofa", "attr": "color"},
        {"q": "sofa", "min_rating": "9"},
    ],
)
def test_bad_requests(client: testing.TestClient, params: dict) -> None:
    result = client.simulate_get("/search", params=params)
    assert result.status_code == 400
    assert "title" in result.json


def test_post_rejects_non_object(client: testing.TestClient) -> None:
    result = client.simulate_post("/search", json=["sofa"])
    assert result.status_code == 400
