"""The query lifecycle: analyse, filter, retrieve (BM25F + HNSW), fuse, rerank, with timings."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from prodsearch.analysis import analyze
from prodsearch.catalog import Catalog
from prodsearch.dense.encoders import Encoder, PairScorer
from prodsearch.dense.hnsw import HNSWIndex, exact_search
from prodsearch.filters import AttributeIndex, Filter
from prodsearch.fusion import convex, rrf, union
from prodsearch.lexical.bm25f import LexicalIndex
from prodsearch.rerank import RerankBudget, rerank

Mode = Literal["hybrid", "lexical", "dense"]
Fusion = Literal["rrf", "convex"]


@dataclass(frozen=True)
class CascadeConfig:
    """Per-stage candidate depths and the rerank budget."""

    lexical_depth: int = 100
    dense_depth: int = 100
    ef_search: int = 128
    # Filters matching fewer documents than this are searched exactly instead of through HNSW,
    # where a very selective filter would make the beam wander through non-matching nodes.
    exact_below: int = 2000
    rerank: RerankBudget = field(default_factory=lambda: RerankBudget(50, 1000.0))


@dataclass(frozen=True)
class SearchRequest:
    query: str
    k: int = 10
    mode: Mode = "hybrid"
    fusion: Fusion = "rrf"
    alpha: float = 0.5
    rerank: bool = True
    filters: Filter = field(default_factory=Filter)
    lexical_method: str = "bmw"


@dataclass
class Hit:
    doc: int
    product_id: int
    name: str
    product_class: str
    fused_score: float
    lexical_rank: int | None
    dense_rank: int | None
    rerank_score: float | None

    def to_json(self) -> dict[str, object]:
        return {
            "product_id": self.product_id,
            "name": self.name,
            "product_class": self.product_class,
            "fused_score": round(self.fused_score, 6),
            "lexical_rank": self.lexical_rank,
            "dense_rank": self.dense_rank,
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 4),
        }


@dataclass
class SearchResponse:
    query: str
    hits: list[Hit]
    timings_ms: dict[str, float]
    stages: list[str]
    reranked: int = 0
    rerank_truncated: bool = False
    filter_matches: int | None = None
    lexical_stats: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "query": self.query,
            "stages": self.stages,
            "timings_ms": {k: round(v, 3) for k, v in self.timings_ms.items()},
            "reranked": self.reranked,
            "rerank_truncated": self.rerank_truncated,
            "filter_matches": self.filter_matches,
            "lexical_stats": self.lexical_stats,
            "hits": [h.to_json() for h in self.hits],
        }


class SearchEngine:
    def __init__(
        self,
        catalog: Catalog,
        lexical: LexicalIndex,
        encoder: Encoder | None = None,
        vectors: np.ndarray | None = None,
        hnsw: HNSWIndex | None = None,
        cross_encoder: PairScorer | None = None,
        cascade: CascadeConfig | None = None,
    ) -> None:
        if (encoder is None) != (vectors is None):
            raise ValueError("dense retrieval needs both an encoder and the product vectors")
        self.catalog = catalog
        self.lexical = lexical
        self.attributes = AttributeIndex(catalog)
        self.encoder = encoder
        self.vectors = vectors
        self.hnsw = hnsw
        self.cross_encoder = cross_encoder
        self.cascade = cascade or CascadeConfig()

    @property
    def has_dense(self) -> bool:
        return self.encoder is not None

    def capabilities(self) -> list[str]:
        caps = ["lexical"]
        if self.has_dense:
            caps.append("dense-hnsw" if self.hnsw is not None else "dense-exact")
        if self.cross_encoder is not None:
            caps.append("rerank")
        return caps

    def _dense(self, q: np.ndarray, mask: np.ndarray | None, n_match: int | None):
        cfg = self.cascade
        if self.hnsw is None or (n_match is not None and n_match < cfg.exact_below):
            return exact_search(
                self.vectors, q, cfg.dense_depth, None if mask is None else mask > 0
            )
        return self.hnsw.search(q, cfg.dense_depth, cfg.ef_search, mask)

    def search(self, req: SearchRequest) -> SearchResponse:
        cfg = self.cascade
        timings: dict[str, float] = {}
        stages: list[str] = []
        t_start = time.perf_counter()

        def lap(name: str, since: float) -> float:
            now = time.perf_counter()
            timings[name] = (now - since) * 1000.0
            return now

        tokens = analyze(req.query)
        t = lap("analyze", t_start)
        mask = self.attributes.mask(req.filters)
        n_match = None if mask is None else int(mask.sum())
        t = lap("filter", t)
        if n_match == 0:
            timings["total"] = (time.perf_counter() - t_start) * 1000.0
            return SearchResponse(req.query, [], timings, stages, filter_matches=0)

        use_lex = req.mode in ("hybrid", "lexical")
        use_dense = req.mode in ("hybrid", "dense") and self.has_dense
        lex_docs = np.zeros(0, np.int32)
        lex_scores = np.zeros(0, np.float64)
        lex_stats: dict[str, int] = {}
        if use_lex:
            lex_docs, lex_scores, lex_stats = self.lexical.search(
                tokens, cfg.lexical_depth, mask, req.lexical_method
            )
            stages.append("lexical")
            t = lap("lexical", t)
        dense_docs = np.zeros(0, np.int32)
        dense_scores = np.zeros(0, np.float32)
        q_vec = None
        if use_dense:
            q_vec = self.encoder.encode_query(req.query)
            t = lap("embed", t)
            dense_docs, dense_scores = self._dense(q_vec, mask, n_match)
            stages.append("dense")
            t = lap("ann", t)

        if use_lex and use_dense:
            if req.fusion == "rrf":
                fused_docs, fused_scores = rrf([lex_docs, dense_docs])
            else:
                cand = union(lex_docs, dense_docs)
                fused_docs, fused_scores = convex(
                    cand, self.lexical.score(tokens, cand), self.vectors[cand] @ q_vec, req.alpha
                )
            stages.append(f"fusion:{req.fusion}")
            t = lap("fusion", t)
        elif use_lex:
            fused_docs, fused_scores = lex_docs, lex_scores
        else:
            fused_docs, fused_scores = dense_docs, dense_scores.astype(np.float64)

        final = fused_docs
        rerank_scores: dict[int, float] = {}
        reranked = 0
        truncated = False
        if req.rerank and self.cross_encoder is not None and fused_docs.shape[0]:
            outcome = rerank(
                self.cross_encoder, req.query, fused_docs, self.catalog.passage, cfg.rerank
            )
            final = outcome.docs
            reranked = outcome.reranked
            truncated = outcome.truncated
            rerank_scores = {int(d): float(s) for d, s in zip(final, outcome.scores, strict=False)}
            stages.append("rerank")
            t = lap("rerank", t)

        fused_of = {int(d): float(s) for d, s in zip(fused_docs, fused_scores, strict=True)}
        lex_rank = {int(d): r + 1 for r, d in enumerate(lex_docs)}
        dense_rank = {int(d): r + 1 for r, d in enumerate(dense_docs)}
        hits = [
            Hit(
                doc=int(d),
                product_id=int(self.catalog.product_ids[d]),
                name=self.catalog.names[d],
                product_class=self.catalog.classes[d],
                fused_score=fused_of[int(d)],
                lexical_rank=lex_rank.get(int(d)),
                dense_rank=dense_rank.get(int(d)),
                rerank_score=rerank_scores.get(int(d)),
            )
            for d in final[: req.k]
        ]
        timings["total"] = (time.perf_counter() - t_start) * 1000.0
        return SearchResponse(
            query=req.query,
            hits=hits,
            timings_ms=timings,
            stages=stages,
            reranked=reranked,
            rerank_truncated=truncated,
            filter_matches=n_match,
            lexical_stats=lex_stats,
        )
