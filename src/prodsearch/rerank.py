"""Cross-encoder rerank of the fused head under a depth and a wall-clock budget.

The head is scored in fused order, one batch at a time. When the deadline passes between batches
the cascade stops: the documents scored so far are reordered by cross-encoder score and
everything else keeps its fused order behind them. A late query therefore degrades towards the
fusion ranking instead of blowing its latency budget.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from prodsearch.dense.encoders import PairScorer


@dataclass(frozen=True)
class RerankBudget:
    depth: int = 50
    deadline_ms: float | None = None
    batch_size: int = 10


@dataclass(frozen=True)
class RerankOutcome:
    docs: np.ndarray
    scores: np.ndarray  # cross-encoder logits for the reranked prefix, in output order
    reranked: int
    truncated: bool


def rerank(
    scorer: PairScorer,
    query: str,
    docs: np.ndarray,
    passage: Callable[[int], str],
    budget: RerankBudget,
    clock: Callable[[], float] = time.perf_counter,
) -> RerankOutcome:
    head = np.asarray(docs[: budget.depth], dtype=np.int32)
    start = clock()
    scored: list[np.ndarray] = []
    done = 0
    truncated = False
    while done < head.shape[0]:
        if (
            budget.deadline_ms is not None
            and done > 0
            and (clock() - start) * 1000.0 >= budget.deadline_ms
        ):
            truncated = True
            break
        batch = head[done : done + budget.batch_size]
        scored.append(scorer.score(query, [passage(int(d)) for d in batch]))
        done += batch.shape[0]
    logits = np.concatenate(scored) if scored else np.zeros(0, np.float32)
    # Stable sort: equal logits keep their fused order.
    order = np.argsort(-logits, kind="stable")
    out = np.concatenate([head[:done][order], np.asarray(docs[done:], dtype=np.int32)])
    return RerankOutcome(docs=out, scores=logits[order], reranked=done, truncated=truncated)
