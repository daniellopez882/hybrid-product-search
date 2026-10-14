"""The WANDS benchmark behind `prodsearch bench`.

Sections, in order: lexical retrieval (BM25, bm25s, BM25F default and CV-tuned), dense
retrieval (exact, in-house HNSW, faiss HNSW at matched recall), fusion (RRF, CV-tuned convex),
cross-encoder rerank, lexical kernel latency, the full cascade's per-stage latency (and again
with ONNX Runtime's spinning worker threads), filtered search, and build cost. Everything lands
in one JSON file with the machine, the git state and the CPU that other processes used while
each latency section ran.

Relevance is reported on held-out folds: anything tuned (BM25F field weights, convex alpha) is
chosen on four folds' queries and scored on the fifth, so every query is scored exactly once by
a setting that never saw it.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from prodsearch import paths, sources
from prodsearch.analysis import analyze
from prodsearch.bench import cv
from prodsearch.bench.baselines import Bm25sBaseline, FaissHnswBaseline
from prodsearch.bench.machine import CpuMeter, git_state, local_now, machine
from prodsearch.bench.metrics import (
    bootstrap_mean,
    latency_summary,
    ndcg_at,
    paired_bootstrap,
    per_query,
)
from prodsearch.catalog import Catalog, Query, Wands, load_wands
from prodsearch.dense.encoders import BGE_QUERY_PREFIX, OnnxBiEncoder, OnnxCrossEncoder
from prodsearch.dense.hnsw import HNSWIndex, exact_search
from prodsearch.engine import CascadeConfig, SearchEngine, SearchRequest
from prodsearch.filters import AttributeIndex, Filter
from prodsearch.fusion import RRF_K, convex, rrf, union
from prodsearch.lexical.bm25f import (
    DEFAULT_BM25F,
    QUANT_LEVELS,
    SINGLE_FIELD_BM25,
    BM25FConfig,
    FieldedPostings,
    LexicalIndex,
)
from prodsearch.lexical.layout import PostingLayout
from prodsearch.rerank import RerankBudget
from prodsearch.store import clustered_order, field_tokens, single_field

Log = Callable[[str], None]

DEPTH = 100  # candidates per retriever; also the Recall cut-off
RERANK_DEPTH = 50
BLOCK_SIZE = 128
EF_SEARCH = 128
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
# Wide enough that both graphs reach the highest recall target on WANDS (0.98 at ef 256).
EF_SWEEP = (10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)
RECALL_TARGETS = (0.9, 0.95, 0.99)
ALPHAS = tuple(round(float(a), 2) for a in np.linspace(0.0, 1.0, 21))
RERANK_DEPTHS = (10, 20, 30, 50)
N_FOLDS = 5
FOLD_SEED = 13
N_BOOT = 10_000
SMOKE_QUERIES = 5
SMOKE_DOCS_PER_QUERY = 120
LATENCY_REPEATS = 3


def bm25f_grid() -> list[BM25FConfig]:
    """3 x 3 x 3 x 3 field-weight grid (description fixed at 1); the a-priori default first."""
    grid = [DEFAULT_BM25F]
    for name, cls, cat, feat in itertools.product(
        (1.0, 2.0, 4.0), (0.0, 1.0, 2.0), (0.0, 1.0, 2.0), (0.0, 0.5, 1.0)
    ):
        weights = {
            "name": name,
            "class": cls,
            "category": cat,
            "features": feat,
            "description": 1.0,
        }
        if weights != dict(DEFAULT_BM25F.weights):
            grid.append(BM25FConfig(weights=weights, b=DEFAULT_BM25F.b, k1=DEFAULT_BM25F.k1))
    return grid


def _clean(obj: object) -> object:
    """JSON-safe copy: numpy scalars to Python, NaN/inf to None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        value = float(obj)
        return None if math.isnan(value) or math.isinf(value) else value
    return obj


def _ms(fn: Callable[[], object], repeats: int = 1) -> tuple[float, object]:
    """Median wall time of `repeats` calls in ms, and the last result."""
    times = []
    result = None
    for _ in range(repeats):
        t = time.perf_counter_ns()
        result = fn()
        times.append((time.perf_counter_ns() - t) / 1e6)
    return float(np.median(times)), result


def _summary(per_q: dict[str, np.ndarray], seed: int) -> dict:
    return {name: bootstrap_mean(v, N_BOOT, seed) for name, v in per_q.items()}


def _vectors(catalog: Catalog, encoder: OnnxBiEncoder, log: Log) -> tuple[np.ndarray, dict]:
    """Product vectors, cached on disk by model, max length and the exact passage texts."""
    passages = [catalog.passage(d) for d in range(len(catalog))]
    digest = hashlib.sha256()
    digest.update(f"{encoder.model.repo_id}@{encoder.model.revision}:{encoder.max_length}".encode())
    for p in passages:
        digest.update(p.encode())
        digest.update(b"\x00")
    key = digest.hexdigest()[:16]
    cache = paths.data_dir() / "cache"
    vec_path, meta_path = cache / f"vectors-{key}.npy", cache / f"vectors-{key}.json"
    if vec_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        log(f"vectors   cached {vec_path.name} (embedded {meta['date']})")
        return np.load(vec_path), {**meta, "from_cache": True}
    log(f"vectors   embedding {len(passages)} products")
    t = time.perf_counter()
    vectors = encoder.encode_passages(passages)
    seconds = time.perf_counter() - t
    meta = {
        "seconds": seconds,
        "products": len(passages),
        "products_per_second": len(passages) / seconds,
        "date": local_now().date().isoformat(),
    }
    cache.mkdir(parents=True, exist_ok=True)
    np.save(vec_path, vectors)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return vectors, {**meta, "from_cache": False}


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def run(out: Path, smoke: bool, latency_queries: int, log: Log, command: str) -> dict:
    wall = time.perf_counter()
    whole_run = CpuMeter()
    load: dict[str, dict] = {}
    started = local_now()
    full = load_wands(paths.wands_dir())
    dataset = {
        "name": "WANDS (Wayfair ANnotation DataSet)",
        "source_commit": sources.WANDS_COMMIT,
        "licence": sources.WANDS_LICENCE,
        "products": len(full.catalog),
        "queries": len(full.queries),
        "judgement_rows": full.judgements.rows,
        "judged_pairs": len(full.judgements),
        "conflicting_pairs": full.judgements.conflicts,
        "queries_with_exact": sum(
            1 for q in full.queries if 2 in full.judgements.for_query(q.qid).values()
        ),
    }
    wands: Wands = full.restrict(SMOKE_QUERIES, SMOKE_DOCS_PER_QUERY) if smoke else full
    del full
    catalog, queries = wands.catalog, wands.queries
    grades = [wands.judgements.for_query(q.qid) for q in queries]
    nq = len(queries)
    folds = cv.fold_ids(nq, N_FOLDS, FOLD_SEED)
    q_tokens = [analyze(q.text) for q in queries]
    log(f"dataset   {len(catalog)} products, {nq} queries{' (smoke subset)' if smoke else ''}")

    rankings: dict[str, list[np.ndarray]] = {}
    build: dict[str, object] = {}

    # ------------------------------------------------------------------ lexical
    t = time.perf_counter()
    tokens = field_tokens(catalog)
    build["analyze_seconds"] = time.perf_counter() - t
    t = time.perf_counter()
    order = clustered_order(catalog)
    build["doc_order_seconds"] = time.perf_counter() - t
    t = time.perf_counter()
    fielded = FieldedPostings.build(tokens, order)
    build["fielded_postings_seconds"] = time.perf_counter() - t
    t = time.perf_counter()
    layout = PostingLayout.build(fielded.docs, fielded.term_ptr, fielded.n_docs, BLOCK_SIZE)
    build["layout_seconds"] = time.perf_counter() - t
    t = time.perf_counter()
    bm25f_default = LexicalIndex.build(fielded, DEFAULT_BM25F, layout=layout)
    build["impacts_seconds"] = time.perf_counter() - t
    build["bm25f_bytes"] = bm25f_default.nbytes()
    build["postings"] = layout.n_postings
    build["terms"] = layout.n_terms
    build["doc_id_bits_per_posting"] = 8 * layout.nbytes()["doc_id_words"] / layout.n_postings
    build["uncompressed_doc_id_bytes"] = 4 * layout.n_postings
    # The same index with postings in catalogue order, to show what the doc order buys.
    bm25f_catalogue_order = LexicalIndex.build(FieldedPostings.build(tokens), DEFAULT_BM25F)
    build["bm25f_catalogue_order_bytes"] = bm25f_catalogue_order.nbytes()
    build["catalogue_order_doc_id_bits_per_posting"] = (
        8 * bm25f_catalogue_order.layout.nbytes()["doc_id_words"] / layout.n_postings
    )
    log(f"lexical   {layout.n_terms} terms, {layout.n_postings} postings")

    single = single_field(tokens)
    t = time.perf_counter()
    bm25 = LexicalIndex.build(FieldedPostings.build(single, order), SINGLE_FIELD_BM25, BLOCK_SIZE)
    build["bm25_single_field_seconds"] = time.perf_counter() - t
    build["bm25_single_field_bytes"] = bm25.nbytes()
    t = time.perf_counter()
    bm25s_np = Bm25sBaseline(single["all"], backend="numpy")
    build["bm25s_index_seconds"] = time.perf_counter() - t
    build["bm25s_bytes"] = bm25s_np.nbytes()
    bm25s_nb = Bm25sBaseline(single["all"], backend="numba")

    rankings["bm25"] = [bm25.search(qt, DEPTH)[0] for qt in q_tokens]
    rankings["bm25s"] = [bm25s_np.search(qt, DEPTH)[0] for qt in q_tokens]
    rankings["bm25f_default"] = [bm25f_default.search(qt, DEPTH)[0] for qt in q_tokens]
    parity = {
        "top10_overlap": _mean(
            [
                len(set(a[:10].tolist()) & set(b[:10].tolist())) / max(1, min(10, len(a)))
                for a, b in zip(rankings["bm25"], rankings["bm25s"], strict=True)
                if len(a)
            ]
        ),
        "top100_overlap": _mean(
            [
                len(set(a.tolist()) & set(b.tolist())) / max(1, len(a))
                for a, b in zip(rankings["bm25"], rankings["bm25s"], strict=True)
                if len(a)
            ]
        ),
    }

    grid = bm25f_grid()
    grid_lists: list[list[np.ndarray]] = []
    grid_ndcg = np.empty((len(grid), nq))
    t = time.perf_counter()
    for ci, config in enumerate(grid):
        index = LexicalIndex.build(fielded, config, layout=layout)
        lists = [index.search(qt, DEPTH)[0] for qt in q_tokens]
        grid_lists.append(lists)
        grid_ndcg[ci] = [ndcg_at(lst, g) for lst, g in zip(lists, grades, strict=True)]
    grid_seconds = time.perf_counter() - t
    bm25f_sel = cv.select(grid_ndcg, folds)
    rankings["bm25f_cv"] = [grid_lists[bm25f_sel.chosen[folds[i]]][i] for i in range(nq)]
    log(f"bm25f     grid of {len(grid)} in {grid_seconds:.0f}s, folds chose {bm25f_sel.chosen}")

    # ------------------------------------------------------------------ dense
    encoder = OnnxBiEncoder()
    vectors, embed_meta = _vectors(catalog, encoder, log)
    build["embedding"] = embed_meta
    q_vecs = np.stack([encoder.encode_query(q.text) for q in queries])
    rankings["dense_exact"] = [exact_search(vectors, qv, DEPTH)[0] for qv in q_vecs]
    t = time.perf_counter()
    graph = HNSWIndex.build(vectors, HNSW_M, HNSW_EF_CONSTRUCTION)
    hnsw_build = time.perf_counter() - t
    log(f"hnsw      built in {hnsw_build:.0f}s (top layer {graph.top})")
    rankings["dense_hnsw"] = [graph.search(qv, DEPTH, EF_SEARCH)[0] for qv in q_vecs]
    t = time.perf_counter()
    faiss_index = FaissHnswBaseline(vectors, HNSW_M, HNSW_EF_CONSTRUCTION, threads=1)
    faiss_build = time.perf_counter() - t
    log(f"faiss     built in {faiss_build:.0f}s")
    meter = CpuMeter()
    ann = _ann_section(vectors, q_vecs, graph, faiss_index)
    load["ann_latency"] = meter.read()
    ann["build"] = {
        "in_house_seconds": hnsw_build,
        "faiss_seconds": faiss_build,
        "threads": 1,
        "in_house_graph_bytes": graph.nbytes(),
        "faiss_serialized_bytes": faiss_index.nbytes(),
        "vector_bytes": int(vectors.nbytes),
        "max_level": graph.top,
    }

    # ------------------------------------------------------------------ fusion
    distinct = sorted(set(bm25f_sel.chosen.tolist()))
    tuned_index = {ci: LexicalIndex.build(fielded, grid[ci], layout=layout) for ci in distinct}
    convex_lists: dict[int, list[list[np.ndarray]]] = {}
    convex_ndcg: dict[int, np.ndarray] = {}
    for ci in distinct:
        convex_lists[ci], convex_ndcg[ci] = _convex_grid(
            tuned_index[ci],
            grid_lists[ci],
            rankings["dense_hnsw"],
            vectors,
            q_vecs,
            q_tokens,
            grades,
        )
    alpha_chosen = np.empty(N_FOLDS, dtype=np.int64)
    rankings["convex_cv"] = [np.zeros(0, np.int32)] * nq
    for f in range(N_FOLDS):
        ci = int(bm25f_sel.chosen[f])
        train = folds != f
        alpha_chosen[f] = int(np.nanargmax(np.nanmean(convex_ndcg[ci][:, train], axis=1)))
        for i in np.flatnonzero(folds == f):
            rankings["convex_cv"][i] = convex_lists[ci][alpha_chosen[f]][i]
    rankings["rrf"] = [
        rrf([grid_lists[bm25f_sel.chosen[folds[i]]][i], rankings["dense_hnsw"][i]])[0]
        for i in range(nq)
    ]
    log(f"fusion    alpha per fold {[ALPHAS[a] for a in alpha_chosen]}")

    # ------------------------------------------------------------------ rerank
    cross = OnnxCrossEncoder()
    ce_cache: dict[tuple[int, int], float] = {}
    t = time.perf_counter()
    for base in ("rrf", "convex_cv"):
        rankings[f"{base}_rerank"] = [
            _rerank_cached(
                cross, ce_cache, i, queries[i].text, rankings[base][i], catalog, RERANK_DEPTH
            )
            for i in range(nq)
        ]
    ce_seconds = time.perf_counter() - t
    log(f"rerank    {len(ce_cache)} pairs in {ce_seconds:.0f}s")
    depth_sweep = {}
    for depth in RERANK_DEPTHS:
        reranked = [
            _rerank_cached(
                cross, ce_cache, i, queries[i].text, rankings["convex_cv"][i], catalog, depth
            )
            for i in range(nq)
        ]
        depth_sweep[str(depth)] = _summary({"ndcg@10": per_query(reranked, grades)["ndcg@10"]}, 1)

    # ------------------------------------------------------------------ relevance tables
    per_method = {name: per_query(lists, grades) for name, lists in rankings.items()}
    relevance = {name: _summary(pq, 1) for name, pq in per_method.items()}
    pairs = [
        ("bm25f_cv", "bm25"),
        ("bm25f_cv", "bm25f_default"),
        ("dense_hnsw", "dense_exact"),
        ("rrf", "bm25f_cv"),
        ("rrf", "dense_hnsw"),
        ("convex_cv", "rrf"),
        ("rrf_rerank", "rrf"),
        ("convex_cv_rerank", "convex_cv"),
        ("convex_cv_rerank", "rrf_rerank"),
    ]
    paired = [
        {
            "a": a,
            "b": b,
            **{
                m: paired_bootstrap(per_method[a][m], per_method[b][m], N_BOOT, 2)
                for m in ("ndcg@10", "recall@100", "mrr@10")
            },
        }
        for a, b in pairs
    ]

    # ------------------------------------------------------------------ latency
    meter = CpuMeter()
    lexical_latency = _lexical_latency(
        {
            "bm25_single_field": bm25,
            "bm25f_default": bm25f_default,
            "bm25f_default_catalogue_order": bm25f_catalogue_order,
        },
        {"bm25s_numpy": bm25s_np, "bm25s_numba": bm25s_nb},
        q_tokens,
    )
    load["lexical_latency"] = meter.read()
    # The deployed configuration is tuned on all queries; it is only used for timing.
    deploy_ci = int(np.nanargmax(np.nanmean(grid_ndcg, axis=1)))
    if deploy_ci not in convex_ndcg:
        tuned_index[deploy_ci] = LexicalIndex.build(fielded, grid[deploy_ci], layout=layout)
        _, convex_ndcg[deploy_ci] = _convex_grid(
            tuned_index[deploy_ci],
            grid_lists[deploy_ci],
            rankings["dense_hnsw"],
            vectors,
            q_vecs,
            q_tokens,
            grades,
        )
    deploy_alpha = ALPHAS[int(np.nanargmax(np.nanmean(convex_ndcg[deploy_ci], axis=1)))]
    engine = SearchEngine(
        catalog,
        tuned_index[deploy_ci],
        encoder,
        vectors,
        graph,
        cross,
        CascadeConfig(DEPTH, DEPTH, EF_SEARCH, rerank=RerankBudget(RERANK_DEPTH, None)),
    )
    timed = np.random.default_rng(5).permutation(nq)[: min(latency_queries, nq)]
    timed_texts = [queries[i].text for i in timed]
    meter = CpuMeter()
    cascade = _cascade_latency(engine, timed_texts, deploy_alpha)
    load["cascade_latency"] = meter.read()
    cascade["config"] = {"bm25f": grid[deploy_ci].to_json(), "alpha": deploy_alpha}
    # Ablation: the same cascade with ONNX Runtime's spinning worker threads (its default).
    spinning = SearchEngine(
        catalog,
        tuned_index[deploy_ci],
        OnnxBiEncoder(spin=True),
        vectors,
        graph,
        OnnxCrossEncoder(spin=True),
        engine.cascade,
    )
    meter = CpuMeter()
    cascade_spinning = _cascade_latency(spinning, timed_texts, deploy_alpha)
    load["cascade_latency_ort_spinning"] = meter.read()
    del spinning
    meter = CpuMeter()
    filtered = _filtered_section(
        catalog, queries, grades, q_tokens, q_vecs, bm25f_default, graph, vectors
    )
    load["filtered_latency"] = meter.read()

    result = {
        "benchmark": "wands-hybrid-search",
        "smoke": smoke,
        "date": started.date().isoformat(),
        "started": started.isoformat(timespec="seconds"),
        **git_state(Path.cwd()),
        "command": command,
        "machine": machine(),
        "dataset": dataset,
        "evaluated": {"products": len(catalog), "queries": nq},
        "settings": {
            "candidate_depth": DEPTH,
            "block_size": BLOCK_SIZE,
            "impact_levels": QUANT_LEVELS,
            "bm25_single_field": SINGLE_FIELD_BM25.to_json(),
            "bm25f_default": DEFAULT_BM25F.to_json(),
            "bm25f_grid_size": len(grid),
            "alpha_grid": list(ALPHAS),
            "rrf_k": RRF_K,
            "hnsw": {"M": HNSW_M, "ef_construction": HNSW_EF_CONSTRUCTION, "ef_search": EF_SEARCH},
            "rerank_depth": RERANK_DEPTH,
            "onnx_intra_op_spinning": False,
            "filtered_exact_below": CascadeConfig().exact_below,
            "service_rerank_deadline_ms": CascadeConfig().rerank.deadline_ms,
            "bi_encoder": {
                "model": f"{sources.BI_ENCODER.repo_id}@{sources.BI_ENCODER.revision}",
                "max_length": encoder.max_length,
                "query_prefix": BGE_QUERY_PREFIX,
            },
            "cross_encoder": {
                "model": f"{sources.CROSS_ENCODER.repo_id}@{sources.CROSS_ENCODER.revision}",
                "file": sources.CROSS_ENCODER.onnx_file,
                "max_length": cross.max_length,
            },
            "folds": N_FOLDS,
            "fold_seed": FOLD_SEED,
            "bootstrap_resamples": N_BOOT,
            "confidence_level": 0.95,
            "recall_targets": list(RECALL_TARGETS),
            "gains": {"Exact": 2, "Partial": 1, "Irrelevant": 0},
        },
        "relevance": relevance,
        "paired": paired,
        "cv": {
            "bm25f_chosen": [grid[c].label() for c in bm25f_sel.chosen],
            "alpha_chosen": [ALPHAS[a] for a in alpha_chosen],
            "bm25f_grid_seconds": grid_seconds,
        },
        "bm25_parity_with_bm25s": parity,
        "rerank": {
            "pairs_scored": len(ce_cache),
            "seconds": ce_seconds,
            "pairs_per_second": len(ce_cache) / ce_seconds if ce_seconds > 0 else None,
            "depth_sweep_convex_cv": depth_sweep,
        },
        "lexical_latency": lexical_latency,
        "ann": ann,
        "cascade_latency": cascade,
        "cascade_latency_ort_spinning": cascade_spinning,
        "filtered": filtered,
        "build": build,
    }
    load["whole_run"] = whole_run.read()
    result["cpu_load"] = load
    result["wall_seconds"] = time.perf_counter() - wall
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_clean(result), indent=2) + "\n", encoding="utf-8")
    log(f"wrote     {out} in {result['wall_seconds']:.0f}s")
    return result


# ---------------------------------------------------------------------- helpers
def _convex_grid(
    index: LexicalIndex,
    lex_lists: Sequence[np.ndarray],
    dense_lists: Sequence[np.ndarray],
    vectors: np.ndarray,
    q_vecs: np.ndarray,
    q_tokens: Sequence[Sequence[str]],
    grades: Sequence[dict[int, int]],
) -> tuple[list[list[np.ndarray]], np.ndarray]:
    """Convex-fused top-DEPTH lists and nDCG@10 for every alpha in the grid."""
    lists: list[list[np.ndarray]] = [[] for _ in ALPHAS]
    ndcg = np.empty((len(ALPHAS), len(q_tokens)))
    for i, qt in enumerate(q_tokens):
        cand = union(lex_lists[i], dense_lists[i])
        lex = index.score(qt, cand)
        dense = vectors[cand] @ q_vecs[i]
        for a, alpha in enumerate(ALPHAS):
            docs, _ = convex(cand, lex, dense, alpha)
            lists[a].append(docs[:DEPTH])
            ndcg[a, i] = ndcg_at(docs, grades[i])
    return lists, ndcg


def _rerank_cached(
    cross: OnnxCrossEncoder,
    cache: dict[tuple[int, int], float],
    qi: int,
    query: str,
    ranking: np.ndarray,
    catalog: Catalog,
    depth: int,
) -> np.ndarray:
    head = [int(d) for d in ranking[:depth]]
    missing = [d for d in head if (qi, d) not in cache]
    if missing:
        scores = cross.score(query, [catalog.passage(d) for d in missing])
        cache.update({(qi, d): float(s) for d, s in zip(missing, scores, strict=True)})
    logits = np.array([cache[(qi, d)] for d in head])
    order = np.argsort(-logits, kind="stable")
    return np.concatenate([np.array(head, dtype=np.int32)[order], ranking[depth:]])


def _lexical_latency(
    indexes: dict[str, LexicalIndex],
    baselines: dict[str, Bm25sBaseline],
    q_tokens: Sequence[Sequence[str]],
) -> dict:
    out: dict[str, dict] = {}
    for k in (10, DEPTH):
        rows: dict[str, dict] = {}
        for name, index in indexes.items():
            reference = [index.search(qt, k, method="exhaustive")[0] for qt in q_tokens]
            for method in ("exhaustive", "wand", "bmw"):
                index.search(q_tokens[0], k, method=method)  # compile outside the timing
                times, scored, decoded, mismatches = [], [], [], 0
                for qt, ref in zip(q_tokens, reference, strict=True):
                    ms, (docs, _, stats) = _ms(
                        lambda qt=qt, ix=index, k=k, m=method: ix.search(qt, k, method=m),
                        LATENCY_REPEATS,
                    )
                    times.append(ms)
                    scored.append(stats["scored"])
                    decoded.append(stats["blocks_decoded"])
                    mismatches += int(not np.array_equal(docs, ref))
                rows[f"{name}/{method}"] = {
                    **latency_summary(times),
                    "mean_docs_scored": _mean(scored),
                    "mean_blocks_decoded": _mean(decoded),
                    "queries_differing_from_exhaustive": mismatches,
                }
        for name, base in baselines.items():
            base.search(q_tokens[0], k)
            times = [
                _ms(lambda qt=qt, b=base, k=k: b.search(qt, k), LATENCY_REPEATS)[0]
                for qt in q_tokens
            ]
            rows[name] = latency_summary(times)
        out[f"k={k}"] = rows
    return out


def _ann_section(
    vectors: np.ndarray, q_vecs: np.ndarray, graph: HNSWIndex, faiss_index: FaissHnswBaseline
) -> dict:
    truth10 = [set(exact_search(vectors, qv, 10)[0].tolist()) for qv in q_vecs]
    truth100 = [set(exact_search(vectors, qv, DEPTH)[0].tolist()) for qv in q_vecs]
    sweep = []
    graph.search(q_vecs[0], 10, EF_SWEEP[0])
    for ef in EF_SWEEP:
        for lib in ("in_house", "faiss"):
            times, recalls = [], []
            for qv, truth in zip(q_vecs, truth10, strict=True):
                if lib == "in_house":
                    ms, ids = _ms(lambda qv=qv, ef=ef: graph.search(qv, 10, ef)[0], LATENCY_REPEATS)
                else:
                    ms, ids = _ms(
                        lambda qv=qv, ef=ef: faiss_index.search(qv, 10, ef), LATENCY_REPEATS
                    )
                times.append(ms)
                recalls.append(len(truth.intersection(ids.tolist())) / max(1, len(truth)))
            sweep.append(
                {"library": lib, "ef": ef, "recall@10": _mean(recalls), **latency_summary(times)}
            )
    matched = {}
    for target in RECALL_TARGETS:
        row = {}
        for lib in ("in_house", "faiss"):
            hits = [s for s in sweep if s["library"] == lib and s["recall@10"] >= target]
            row[lib] = min(hits, key=lambda s: s["ef"]) if hits else None
        matched[str(target)] = row
    at_depth = {}
    for lib in ("in_house", "faiss"):
        recalls = []
        for qv, truth in zip(q_vecs, truth100, strict=True):
            if lib == "in_house":
                ids = graph.search(qv, DEPTH, EF_SEARCH)[0]
            else:
                ids = faiss_index.search(qv, DEPTH, EF_SEARCH)
            recalls.append(len(truth.intersection(ids.tolist())) / max(1, len(truth)))
        at_depth[lib] = _mean(recalls)
    return {
        "sweep_k10": sweep,
        "matched_recall_k10": matched,
        f"recall@{DEPTH}_at_ef{EF_SEARCH}": at_depth,
    }


def _cascade_latency(engine: SearchEngine, texts: Sequence[str], alpha: float) -> dict:
    stages: dict[str, dict[str, list[float]]] = {"convex+rerank": {}, "rrf": {}}
    for text in texts[:3]:  # warm-up: JIT, ONNX allocations
        engine.search(SearchRequest(text, fusion="convex", alpha=alpha, rerank=True))
    for text in texts:
        for label, req in (
            ("convex+rerank", SearchRequest(text, fusion="convex", alpha=alpha, rerank=True)),
            ("rrf", SearchRequest(text, fusion="rrf", rerank=False)),
        ):
            response = engine.search(req)
            for stage, ms in response.timings_ms.items():
                stages[label].setdefault(stage, []).append(ms)
    return {
        "queries": len(texts),
        "pipelines": {
            label: {stage: latency_summary(ms) for stage, ms in per_stage.items()}
            for label, per_stage in stages.items()
        },
    }


def _filtered_section(
    catalog: Catalog,
    queries: Sequence[Query],
    grades: Sequence[dict[int, int]],
    q_tokens: Sequence[Sequence[str]],
    q_vecs: np.ndarray,
    lexical: LexicalIndex,
    graph: HNSWIndex,
    vectors: np.ndarray,
) -> dict:
    """Facet filter = the query's own WANDS query_class (an oracle facet choice)."""
    attrs = AttributeIndex(catalog)
    lex_ms, lex_exact_ms, lex_mismatch = [], [], 0
    ann_ms, ann_recall, below_fallback, matches = [], [], 0, []
    plain, faceted = [], []
    used = 0
    for i, q in enumerate(queries):
        mask = attrs.mask(Filter(classes=(q.query_class,))) if q.query_class else None
        if mask is None or not mask.any():
            continue
        used += 1
        n_match = int(mask.sum())
        matches.append(n_match)
        below_fallback += int(n_match < CascadeConfig().exact_below)
        ms, (docs, _, _) = _ms(lambda qt=q_tokens[i], m=mask: lexical.search(qt, DEPTH, m, "bmw"))
        ref_ms, (ref, _, _) = _ms(
            lambda qt=q_tokens[i], m=mask: lexical.search(qt, DEPTH, m, "exhaustive")
        )
        lex_ms.append(ms)
        lex_exact_ms.append(ref_ms)
        lex_mismatch += int(not np.array_equal(docs, ref))
        ms, ids = _ms(lambda qv=q_vecs[i], m=mask: graph.search(qv, 10, EF_SEARCH, m)[0])
        truth = exact_search(vectors, q_vecs[i], 10, mask)[0]
        ann_ms.append(ms)
        ann_recall.append(len(set(truth.tolist()) & set(ids.tolist())) / max(1, len(truth)))
        dense_f = graph.search(q_vecs[i], DEPTH, EF_SEARCH, mask)[0]
        dense_u = graph.search(q_vecs[i], DEPTH, EF_SEARCH)[0]
        lex_u = lexical.search(q_tokens[i], DEPTH)[0]
        faceted.append(ndcg_at(rrf([docs, dense_f])[0], grades[i]))
        plain.append(ndcg_at(rrf([lex_u, dense_u])[0], grades[i]))
    return {
        "queries_with_matching_class": used,
        "median_matching_products": float(np.median(matches)) if matches else None,
        "queries_below_exact_fallback": below_fallback,
        "bmw_filtered": latency_summary(lex_ms),
        "exhaustive_filtered": latency_summary(lex_exact_ms),
        "bmw_queries_differing_from_exhaustive": lex_mismatch,
        "hnsw_filtered_k10": {**latency_summary(ann_ms), "recall@10": _mean(ann_recall)},
        "rrf_bm25f_default_ndcg@10": {
            "unfiltered": bootstrap_mean(np.array(plain), N_BOOT, 3),
            "class_filtered": bootstrap_mean(np.array(faceted), N_BOOT, 3),
        },
    }
