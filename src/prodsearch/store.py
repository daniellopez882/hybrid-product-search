"""Build the on-disk index (postings, product vectors, HNSW graph) and load it into an engine."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np

from prodsearch.analysis import analyze
from prodsearch.catalog import FIELDS, Catalog, load_catalog
from prodsearch.dense.encoders import Encoder, OnnxBiEncoder, OnnxCrossEncoder
from prodsearch.dense.hnsw import HNSWIndex
from prodsearch.engine import CascadeConfig, SearchEngine
from prodsearch.lexical.bm25f import DEFAULT_BM25F, BM25FConfig, FieldedPostings, LexicalIndex
from prodsearch.sources import BI_ENCODER, CROSS_ENCODER, model_present

MANIFEST = "manifest.json"
LEXICAL = "lexical.npz"
VECTORS = "vectors.npy"
GRAPH = "hnsw.npz"


def sample_catalog_path() -> Path:
    """A 40-product catalogue in the WANDS column layout, shipped with the package."""
    return Path(str(resources.files("prodsearch") / "sample" / "products.tsv"))


def field_tokens(catalog: Catalog, fields: Sequence[str] = FIELDS) -> dict[str, list[list[str]]]:
    return {f: [analyze(catalog.field_text(d, f)) for d in range(len(catalog))] for f in fields}


def single_field(tokens: dict[str, list[list[str]]]) -> dict[str, list[list[str]]]:
    """All fields concatenated into one, for plain BM25."""
    n = len(next(iter(tokens.values())))
    return {"all": [[t for f in tokens for t in tokens[f][d]] for d in range(n)]}


def build_lexical(catalog: Catalog, config: BM25FConfig = DEFAULT_BM25F) -> LexicalIndex:
    return LexicalIndex.build(FieldedPostings.build(field_tokens(catalog)), config)


@dataclass
class HnswParams:
    M: int = 16
    ef_construction: int = 200
    seed: int = 7


def build_index(
    catalog_path: Path,
    out: Path,
    config: BM25FConfig = DEFAULT_BM25F,
    encoder: Encoder | None = None,
    hnsw: HnswParams | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog(catalog_path)
    seconds: dict[str, float] = {}
    t = time.perf_counter()
    lexical = build_lexical(catalog, config)
    seconds["lexical"] = time.perf_counter() - t
    lexical.save(out / LEXICAL)
    log(f"lexical   {len(lexical.vocab)} terms, {lexical.layout.n_postings} postings")
    manifest: dict = {
        "catalog": str(catalog_path.resolve()),
        "n_docs": len(catalog),
        "bm25f": config.to_json(),
        "dense": None,
        "hnsw": None,
    }
    if encoder is not None:
        t = time.perf_counter()
        vectors = encoder.encode_passages([catalog.passage(d) for d in range(len(catalog))])
        seconds["embed"] = time.perf_counter() - t
        np.save(out / VECTORS, vectors)
        log(f"vectors   {vectors.shape} in {seconds['embed']:.0f}s")
        params = hnsw or HnswParams()
        t = time.perf_counter()
        graph = HNSWIndex.build(vectors, params.M, params.ef_construction, params.seed)
        seconds["hnsw"] = time.perf_counter() - t
        graph.save(out / GRAPH)
        log(f"hnsw      M={params.M} efC={params.ef_construction} in {seconds['hnsw']:.0f}s")
        manifest["dense"] = {
            "model": f"{BI_ENCODER.repo_id}@{BI_ENCODER.revision}",
            "max_length": getattr(encoder, "max_length", None),
        }
        manifest["hnsw"] = vars(params)
    manifest["build_seconds"] = {k: round(v, 3) for k, v in seconds.items()}
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_engine(
    index: Path, rerank: bool = True, cascade: CascadeConfig | None = None
) -> SearchEngine:
    manifest = json.loads((index / MANIFEST).read_text(encoding="utf-8"))
    catalog = load_catalog(Path(manifest["catalog"]))
    lexical = LexicalIndex.load(index / LEXICAL)
    encoder = vectors = graph = None
    if manifest.get("dense"):
        encoder = OnnxBiEncoder(max_length=manifest["dense"].get("max_length") or 128)
        vectors = np.load(index / VECTORS)
        graph = HNSWIndex.load(index / GRAPH, vectors)
    cross = OnnxCrossEncoder() if rerank and model_present(CROSS_ENCODER) else None
    return SearchEngine(catalog, lexical, encoder, vectors, graph, cross, cascade)


def sample_engine() -> SearchEngine:
    """Lexical-only engine over the bundled sample catalogue; needs no download."""
    catalog = load_catalog(sample_catalog_path())
    return SearchEngine(catalog, build_lexical(catalog))
