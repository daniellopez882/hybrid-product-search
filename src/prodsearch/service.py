"""Falcon ASGI service: GET/POST /search with per-stage timings, GET /healthz.

Searches run on a small thread pool so the event loop keeps answering health checks while a
query is in the cross-encoder; the numba kernels release the GIL and onnxruntime does too.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import falcon
import falcon.asgi

from prodsearch.engine import SearchEngine, SearchRequest
from prodsearch.filters import Filter

MAX_K = 100
MODES = ("hybrid", "lexical", "dense")
FUSIONS = ("rrf", "convex")


def _bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    raise falcon.HTTPBadRequest(title="invalid parameter", description=f"{name} must be a boolean")


def _number(value: Any, name: str, kind: type, lo: float, hi: float) -> Any:
    try:
        number = kind(value)
    except (TypeError, ValueError):
        raise falcon.HTTPBadRequest(
            title="invalid parameter", description=f"{name} must be a {kind.__name__}"
        ) from None
    if not lo <= number <= hi:
        raise falcon.HTTPBadRequest(
            title="invalid parameter", description=f"{name} must be between {lo} and {hi}"
        )
    return number


def _list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def parse_request(params: dict[str, Any]) -> SearchRequest:
    query = str(params.get("q") or params.get("query") or "").strip()
    if not query:
        raise falcon.HTTPBadRequest(title="missing parameter", description="q is required")
    mode = str(params.get("mode", "hybrid"))
    if mode not in MODES:
        raise falcon.HTTPBadRequest(title="invalid parameter", description=f"mode: one of {MODES}")
    fusion = str(params.get("fusion", "rrf"))
    if fusion not in FUSIONS:
        raise falcon.HTTPBadRequest(
            title="invalid parameter", description=f"fusion: one of {FUSIONS}"
        )
    attributes = []
    for item in _list(params.get("attr")):
        key, sep, value = item.partition("=")
        if not sep or not key.strip() or not value.strip():
            raise falcon.HTTPBadRequest(
                title="invalid parameter", description="attr must look like key=value"
            )
        attributes.append((key.strip(), value.strip()))
    min_rating = params.get("min_rating")
    min_reviews = params.get("min_reviews")
    return SearchRequest(
        query=query,
        k=_number(params.get("k", 10), "k", int, 1, MAX_K),
        mode=mode,  # type: ignore[arg-type]
        fusion=fusion,  # type: ignore[arg-type]
        alpha=_number(params.get("alpha", 0.5), "alpha", float, 0.0, 1.0),
        rerank=_bool(params.get("rerank", True), "rerank"),
        filters=Filter(
            classes=tuple(_list(params.get("class"))),
            category=params.get("category") or None,
            min_rating=None
            if min_rating is None
            else _number(min_rating, "min_rating", float, 0.0, 5.0),
            min_reviews=None
            if min_reviews is None
            else _number(min_reviews, "min_reviews", float, 0.0, 1e9),
            attributes=tuple(attributes),
        ),
    )


class _Health:
    def __init__(self, engine: SearchEngine) -> None:
        self.engine = engine

    async def on_get(self, req: falcon.asgi.Request, resp: falcon.asgi.Response) -> None:
        resp.media = {
            "status": "ok",
            "products": len(self.engine.catalog),
            "capabilities": self.engine.capabilities(),
        }


class _Search:
    def __init__(self, engine: SearchEngine, pool: ThreadPoolExecutor) -> None:
        self.engine = engine
        self.pool = pool

    async def _run(self, params: dict[str, Any], resp: falcon.asgi.Response) -> None:
        request = parse_request(params)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self.pool, self.engine.search, request)
        resp.media = result.to_json()

    async def on_get(self, req: falcon.asgi.Request, resp: falcon.asgi.Response) -> None:
        params: dict[str, Any] = dict(req.params)
        await self._run(params, resp)

    async def on_post(self, req: falcon.asgi.Request, resp: falcon.asgi.Response) -> None:
        body = await req.get_media()
        if not isinstance(body, dict):
            raise falcon.HTTPBadRequest(title="invalid body", description="expected a JSON object")
        await self._run(body, resp)


def create_app(engine: SearchEngine, workers: int = 2) -> falcon.asgi.App:
    app = falcon.asgi.App()
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="search")
    app.add_route("/healthz", _Health(engine))
    app.add_route("/search", _Search(engine, pool))
    return app
