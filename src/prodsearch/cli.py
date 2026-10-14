"""Command line: fetch, index, search, serve, bench."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from prodsearch import paths, sources

if TYPE_CHECKING:
    from prodsearch.filters import Filter
    from prodsearch.lexical.bm25f import BM25FConfig

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)


def _weights(spec: str | None) -> BM25FConfig:
    from prodsearch.lexical.bm25f import DEFAULT_BM25F, BM25FConfig

    if not spec:
        return DEFAULT_BM25F
    weights = dict(DEFAULT_BM25F.weights)
    for item in spec.split(","):
        name, _, value = item.partition("=")
        if name.strip() not in weights:
            raise typer.BadParameter(f"unknown field {name!r}; fields: {', '.join(weights)}")
        weights[name.strip()] = float(value)
    return BM25FConfig(weights=weights, b=DEFAULT_BM25F.b, k1=DEFAULT_BM25F.k1)


@app.command()
def fetch(
    models: Annotated[bool, typer.Option(help="Also fetch the two ONNX models.")] = True,
) -> None:
    """Download WANDS and the ONNX models, verifying every file's sha256."""
    target = sources.fetch_wands(typer.echo)
    typer.echo(f"WANDS     {target}")
    if models:
        for model in sources.MODELS:
            typer.echo(f"{model.key:<13} {sources.fetch_model(model, typer.echo)}")


@app.command()
def index(
    out: Annotated[Path | None, typer.Option(help="Index directory.")] = None,
    catalog: Annotated[Path | None, typer.Option(help="Catalogue TSV (WANDS layout).")] = None,
    dense: Annotated[bool, typer.Option(help="Embed products and build HNSW.")] = True,
    weights: Annotated[
        str | None, typer.Option(help="BM25F field weights, e.g. name=3,description=0.5")
    ] = None,
    m: Annotated[int, typer.Option("--m", help="HNSW links per node (2M on layer 0).")] = 16,
    ef_construction: Annotated[int, typer.Option(help="HNSW build beam width.")] = 200,
) -> None:
    """Build BM25F postings, product vectors and the HNSW graph."""
    from prodsearch.dense.encoders import OnnxBiEncoder
    from prodsearch.store import HnswParams, build_index

    catalog = catalog or paths.wands_dir() / "product.csv"
    if not catalog.is_file():
        raise typer.BadParameter(f"{catalog} not found; run `prodsearch fetch` first")
    encoder = None
    if dense:
        if not sources.model_present(sources.BI_ENCODER):
            raise typer.BadParameter("bi-encoder missing; run `prodsearch fetch` or use --no-dense")
        encoder = OnnxBiEncoder()
    manifest = build_index(
        catalog,
        out or paths.index_dir(),
        _weights(weights),
        encoder,
        HnswParams(M=m, ef_construction=ef_construction),
        typer.echo,
    )
    typer.echo(json.dumps(manifest["build_seconds"]))


def _filters(
    classes: list[str] | None,
    category: str | None,
    min_rating: float | None,
    attr: list[str] | None,
) -> Filter:
    from prodsearch.filters import Filter

    pairs = []
    for item in attr or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise typer.BadParameter("--attr takes key=value")
        pairs.append((key, value))
    return Filter(
        classes=tuple(classes or ()),
        category=category,
        min_rating=min_rating,
        attributes=tuple(pairs),
    )


@app.command()
def search(
    query: str,
    index_dir: Annotated[Path | None, typer.Option("--index", help="Index directory.")] = None,
    k: int = 10,
    mode: Annotated[str, typer.Option(help="hybrid, lexical or dense.")] = "hybrid",
    fusion: Annotated[str, typer.Option(help="rrf or convex.")] = "rrf",
    alpha: Annotated[float, typer.Option(help="Lexical weight for convex fusion.")] = 0.5,
    rerank: Annotated[bool, typer.Option(help="Cross-encoder rerank of the fused head.")] = True,
    product_class: Annotated[list[str] | None, typer.Option("--class")] = None,
    category: Annotated[str | None, typer.Option(help="Category path prefix.")] = None,
    min_rating: float | None = None,
    attr: Annotated[list[str] | None, typer.Option(help="Feature filter key=value.")] = None,
    sample: Annotated[bool, typer.Option(help="Search the bundled sample catalogue.")] = False,
) -> None:
    """Run one query and print the hits with per-stage timings."""
    from prodsearch.engine import SearchRequest
    from prodsearch.store import load_engine, sample_engine

    engine = sample_engine() if sample else load_engine(index_dir or paths.index_dir(), rerank)
    response = engine.search(
        SearchRequest(
            query=query,
            k=k,
            mode=mode,  # type: ignore[arg-type]
            fusion=fusion,  # type: ignore[arg-type]
            alpha=alpha,
            rerank=rerank,
            filters=_filters(product_class, category, min_rating, attr),
        )
    )
    for rank, hit in enumerate(response.hits, 1):
        typer.echo(f"{rank:>2}. [{hit.product_id}] {hit.name}  ({hit.product_class})")
    timings = "  ".join(f"{name}={ms:.1f}ms" for name, ms in response.timings_ms.items())
    typer.echo(f"stages: {' > '.join(response.stages)}")
    typer.echo(f"timings: {timings}")


@app.command()
def serve(
    index_dir: Annotated[Path | None, typer.Option("--index", help="Index directory.")] = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    rerank: bool = True,
    sample: Annotated[bool, typer.Option(help="Serve the bundled sample catalogue.")] = False,
) -> None:
    """Serve /search and /healthz over HTTP (Falcon ASGI on uvicorn)."""
    import uvicorn

    from prodsearch.service import create_app
    from prodsearch.store import load_engine, sample_engine

    engine = sample_engine() if sample else load_engine(index_dir or paths.index_dir(), rerank)
    uvicorn.run(create_app(engine), host=host, port=port, log_level="info")


@app.command()
def bench(
    out: Annotated[Path, typer.Option(help="Results JSON.")] = Path("bench/results/wands.json"),
    smoke: Annotated[
        bool, typer.Option(help="Tiny subset (first 12 queries, their judged products).")
    ] = False,
    latency_queries: Annotated[
        int, typer.Option(help="Queries timed through the full cascade.")
    ] = 200,
) -> None:
    """Run the WANDS benchmark (relevance with 5-fold CV, baselines, latency, build cost)."""
    from prodsearch.bench.runner import run

    if not sources.wands_present():
        raise typer.BadParameter("WANDS missing; run `prodsearch fetch` first")
    for model in sources.MODELS:
        if not sources.model_present(model):
            raise typer.BadParameter(f"{model.repo_id} missing; run `prodsearch fetch` first")
    command = f"uv run prodsearch bench --out {out.as_posix()}"
    if smoke:
        command += " --smoke"
    if latency_queries != 200:
        command += f" --latency-queries {latency_queries}"
    run(out=out, smoke=smoke, latency_queries=latency_queries, log=typer.echo, command=command)


if __name__ == "__main__":
    app()
